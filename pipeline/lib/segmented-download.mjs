#!/usr/bin/env node
// Downloads a big file over a bad connection. The file is fetched as fixed-size
// HTTP Range segments, a few at a time, each retried with backoff until it
// lands. Progress lives in `<dest>.part.json`, so a crash, a dropped Wi-Fi or a
// reboot resumes at the first missing segment instead of starting over.
//
// The source is pinned by its ETag (or Last-Modified) through If-Range: if it
// changes between runs, the download starts over rather than splicing two files
// together. A server without Range support falls back to one stream. The whole
// file is hashed before the atomic rename into place, and running it again on a
// finished file is a no-op.
//
// CLI: node scripts/lib/segmented-download.mjs <url> <dest>
//        [--sha1 <hex> | --sha256 <hex>] [--conn 2] [--segment-mb 4]
//        [--rate <KB/s>] [--header "Name: value"]... [--log <file>]

import { createHash } from 'node:crypto';
import { createReadStream, existsSync } from 'node:fs';
import { appendFile, mkdir, open, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MiB = 1024 * 1024;

export class DownloadError extends Error {
  constructor(message, { fatal = false } = {}) {
    super(message);
    this.fatal = fatal;
  }
}

/** Statuses worth retrying: the network, the server or a rate limit, never the request itself. */
const TRANSIENT = new Set([408, 425, 429, 500, 502, 503, 504]);

/** A token bucket shared by every connection, so `--rate` caps the whole download. */
function rateLimiter(bytesPerSec, sleep) {
  if (!bytesPerSec) return async () => {};
  let tokens = bytesPerSec;
  let last = Date.now();
  return async (n) => {
    for (;;) {
      const now = Date.now();
      tokens = Math.min(bytesPerSec, tokens + ((now - last) / 1000) * bytesPerSec);
      last = now;
      if (tokens >= n || tokens >= bytesPerSec) {
        tokens -= n;
        return;
      }
      await sleep(Math.ceil(((n - tokens) / bytesPerSec) * 1000));
    }
  };
}

async function hashFile(file, algo) {
  const h = createHash(algo);
  for await (const chunk of createReadStream(file, { highWaterMark: 4 * MiB })) h.update(chunk);
  return h.digest('hex');
}

/**
 * @param {object} o
 * @param {string} o.url
 * @param {string} o.dest
 * @param {{ algo: 'sha1' | 'sha256', hex: string }} [o.hash]
 * @param {number} [o.segmentBytes] default 4 MiB
 * @param {number} [o.connections] default 2
 * @param {number} [o.rateBytesPerSec] optional cap for the whole download
 * @param {Record<string, string>} [o.headers]
 * @param {() => Promise<string>} [o.resolveUrl] re-resolves an expiring signed URL after a 401/403
 * @param {AbortSignal} [o.signal]
 * @param {(line: string) => void} [o.log]
 * @param {number} [o.stallMs] a segment with no bytes for this long is retried. Default 30 s
 * @param {number} [o.backoffBaseMs] default 1000; the backoff doubles up to 60 s, with jitter
 * @param {number} [o.maxRetries] per segment; default unlimited
 * @param {typeof fetch} [o.fetchImpl]
 * @param {(ms: number) => Promise<void>} [o.sleep]
 * @returns {Promise<{ size: number, resumed: boolean, requests: number }>}
 */
export async function download(o) {
  const segmentBytes = o.segmentBytes ?? 4 * MiB;
  const connections = Math.max(1, o.connections ?? 2);
  const fetchImpl = o.fetchImpl ?? fetch;
  const sleep = o.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
  const log = o.log ?? (() => {});
  const stallMs = o.stallMs ?? 30_000;
  const backoffBase = o.backoffBaseMs ?? 1000;
  const maxRetries = o.maxRetries ?? Infinity;
  const part = `${o.dest}.part`;
  const stateFile = `${o.dest}.part.json`;
  const take = rateLimiter(o.rateBytesPerSec, sleep);
  let url = o.url;
  let requests = 0;

  if (o.hash && existsSync(o.dest) && (await hashFile(o.dest, o.hash.algo)) === o.hash.hex.toLowerCase()) {
    log(`already have ${o.dest}`);
    return { size: (await stat(o.dest)).size, resumed: false, requests };
  }
  await mkdir(path.dirname(path.resolve(o.dest)), { recursive: true });

  const backoff = (attempt) => {
    const ms = Math.min(60_000, backoffBase * 2 ** Math.min(attempt, 16));
    return ms / 2 + Math.random() * (ms / 2);
  };

  /** One request, with a stall timer that aborts it when the bytes stop. */
  async function request(headers) {
    // An abort that landed between two requests fired no event for this one.
    if (o.signal?.aborted) throw o.signal.reason ?? new DownloadError('aborted', { fatal: true });
    requests++;
    const ctl = new AbortController();
    const onAbort = () => ctl.abort(o.signal.reason);
    o.signal?.addEventListener('abort', onAbort, { once: true });
    let timer = setTimeout(() => ctl.abort(new DownloadError('stalled')), stallMs);
    const poke = () => {
      clearTimeout(timer);
      timer = setTimeout(() => ctl.abort(new DownloadError('stalled')), stallMs);
    };
    const done = () => {
      clearTimeout(timer);
      o.signal?.removeEventListener('abort', onAbort);
    };
    try {
      const res = await fetchImpl(url, { headers: { ...o.headers, ...headers }, signal: ctl.signal, redirect: 'follow' });
      poke();
      return { res, poke, done, ctl };
    } catch (e) {
      done();
      throw e;
    }
  }

  /** Classifies a failed response: retry, re-resolve the URL, or give up. */
  async function check(res) {
    if (res.ok) return;
    await res.body?.cancel().catch(() => {});
    if ((res.status === 401 || res.status === 403) && o.resolveUrl) {
      url = await o.resolveUrl();
      throw new DownloadError(`HTTP ${res.status}; the URL was re-resolved`);
    }
    throw new DownloadError(`HTTP ${res.status}`, { fatal: !TRANSIENT.has(res.status) });
  }

  // Probe: total size, the validator, and whether the server honours Range.
  let probe;
  for (let attempt = 0; ; attempt++) {
    try {
      const { res, done } = await request({ Range: 'bytes=0-0' });
      await check(res);
      await res.body?.cancel().catch(() => {});
      done();
      probe = res;
      break;
    } catch (e) {
      if (o.signal?.aborted) throw o.signal.reason ?? e;
      if (e.fatal || attempt >= maxRetries) throw e;
      log(`probe failed (${e.message}); retrying`);
      await sleep(backoff(attempt));
    }
  }
  const validator = probe.headers.get('etag') ?? probe.headers.get('last-modified') ?? '';
  const range = probe.status === 206 ? /\/(\d+)\s*$/.exec(probe.headers.get('content-range') ?? '') : null;

  if (!range) {
    log('no Range support; one stream, restarted from the top on failure');
    return { ...(await single()), resumed: false, requests };
  }
  const size = Number(range[1]);
  const count = Math.ceil(size / segmentBytes);

  let state = null;
  try {
    state = JSON.parse(await readFile(stateFile, 'utf8'));
  } catch {}
  const resumed = !!(state && state.size === size && state.validator === validator && state.segmentBytes === segmentBytes && existsSync(part));
  if (!resumed) state = { url: o.url, size, validator, segmentBytes, done: [] };
  const done = new Set(state.done);
  if (resumed) log(`resuming: ${done.size}/${count} segments already here`);

  const fh = await open(part, existsSync(part) && resumed ? 'r+' : 'w+');
  await fh.truncate(size);

  let saving = Promise.resolve();
  const save = () => {
    saving = saving.then(async () => {
      const tmp = `${stateFile}.tmp`;
      await writeFile(tmp, JSON.stringify({ ...state, done: [...done].sort((a, b) => a - b) }));
      await rename(tmp, stateFile);
    });
    return saving;
  };
  await save();

  let got = [...done].reduce((n, i) => n + Math.min(segmentBytes, size - i * segmentBytes), 0);
  const started = Date.now();
  const startBytes = got;
  let lastReport = 0;
  const report = (force = false) => {
    const now = Date.now();
    if (!force && now - lastReport < 5000) return;
    lastReport = now;
    const speed = (got - startBytes) / Math.max(1, (now - started) / 1000);
    const eta = speed > 0 ? Math.round((size - got) / speed) : Infinity;
    log(`${((got / size) * 100).toFixed(1)}%  ${(got / MiB).toFixed(1)}/${(size / MiB).toFixed(1)} MiB  ${(speed / 1024).toFixed(0)} KB/s  eta ${Number.isFinite(eta) ? `${eta}s` : '?'}`);
  };

  let changed = false;
  let halt = false;
  const queue = [];
  for (let i = 0; i < count; i++) if (!done.has(i)) queue.push(i);

  async function segment(i) {
    const start = i * segmentBytes;
    const end = Math.min(size, start + segmentBytes) - 1;
    for (let attempt = 0; ; attempt++) {
      let wrote = 0;
      try {
        const { res, poke, done: finish } = await request({ Range: `bytes=${start}-${end}`, ...(validator ? { 'If-Range': validator } : {}) });
        try {
          await check(res);
          if (res.status !== 206) {
            // If-Range failed: the file changed under us. Start the whole download again.
            await res.body?.cancel().catch(() => {});
            changed = true;
            throw new DownloadError('the source changed', { fatal: true });
          }
          for await (const chunk of res.body) {
            poke();
            await take(chunk.length);
            if (wrote + chunk.length > end - start + 1) throw new DownloadError('segment longer than asked');
            await fh.write(chunk, 0, chunk.length, start + wrote);
            wrote += chunk.length;
            got += chunk.length;
            report();
          }
        } finally {
          finish();
        }
        if (wrote !== end - start + 1) throw new DownloadError(`segment ${i} short: ${wrote}/${end - start + 1}`);
        done.add(i);
        await save();
        return;
      } catch (e) {
        got -= wrote;
        if (halt) throw e;
        if (o.signal?.aborted) throw o.signal.reason ?? e;
        if (e.fatal || attempt >= maxRetries) throw e;
        log(`segment ${i} failed (${e.message ?? e}); retry ${attempt + 1}`);
        await sleep(backoff(attempt));
      }
    }
  }

  // Every worker settles before the file is closed, so nothing writes to a closed or deleted file.
  let failure = null;
  await Promise.all(
    Array.from({ length: Math.min(connections, queue.length || 1) }, async () => {
      try {
        while (queue.length && !changed && !halt) await segment(queue.shift());
      } catch (e) {
        halt = true;
        failure ??= e;
      }
    }),
  );
  if (failure) {
    await fh.close();
    await saving.catch(() => {});
    if (changed) {
      log('the source changed between runs; starting over');
      await rm(stateFile, { force: true });
      await rm(part, { force: true });
      return download(o);
    }
    throw failure;
  }
  await fh.sync();
  await fh.close();
  await saving;
  report(true);
  await finishFile();
  await rm(stateFile, { force: true });
  return { size, resumed, requests };

  async function finishFile() {
    if (o.hash) {
      const hex = await hashFile(part, o.hash.algo);
      if (hex !== o.hash.hex.toLowerCase()) {
        await rm(part, { force: true });
        await rm(stateFile, { force: true });
        throw new DownloadError(`${o.hash.algo} mismatch: got ${hex}, want ${o.hash.hex}`, { fatal: true });
      }
    }
    await rename(part, o.dest);
  }

  async function single() {
    for (let attempt = 0; ; attempt++) {
      let fh2;
      try {
        const { res, poke, done: finish } = await request({});
        try {
          await check(res);
          fh2 = await open(part, 'w');
          let n = 0;
          for await (const chunk of res.body) {
            poke();
            await take(chunk.length);
            await fh2.write(chunk);
            n += chunk.length;
          }
          await fh2.close();
          fh2 = null;
          await finishFile();
          return { size: n };
        } finally {
          finish();
        }
      } catch (e) {
        await fh2?.close().catch(() => {});
        if (o.signal?.aborted) throw o.signal.reason ?? e;
        if (e.fatal || attempt >= maxRetries) throw e;
        log(`stream failed (${e.message}); retry ${attempt + 1}`);
        await sleep(backoff(attempt));
      }
    }
  }
}

async function main(argv) {
  const args = { headers: {} };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--sha1') args.hash = { algo: 'sha1', hex: next() };
    else if (a === '--sha256') args.hash = { algo: 'sha256', hex: next() };
    else if (a === '--conn') args.connections = Number(next());
    else if (a === '--segment-mb') args.segmentBytes = Number(next()) * MiB;
    else if (a === '--rate') args.rateBytesPerSec = Number(next()) * 1024;
    else if (a === '--header') {
      const [k, ...v] = next().split(':');
      args.headers[k.trim()] = v.join(':').trim();
    } else if (a === '--log') args.logFile = next();
    else rest.push(a);
  }
  const [url, dest] = rest;
  if (!url || !dest) {
    console.error('usage: segmented-download.mjs <url> <dest> [--sha1 X|--sha256 X] [--conn N] [--segment-mb N] [--rate KB/s] [--header "K: V"] [--log file]');
    process.exit(2);
  }
  const log = (line) => {
    const stamped = `${new Date().toISOString()} ${line}`;
    console.log(stamped);
    if (args.logFile) appendFile(args.logFile, `${stamped}\n`).catch(() => {});
  };
  const t = Date.now();
  const r = await download({ url, dest, ...args, log });
  log(`done: ${dest} (${(r.size / MiB).toFixed(1)} MiB, ${r.requests} requests, ${((Date.now() - t) / 1000).toFixed(0)}s${r.resumed ? ', resumed' : ''})`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((e) => {
    console.error(`download failed: ${e.message ?? e}`);
    // Not process.exit, which crashes Node on Windows while fetch's sockets close.
    process.exitCode = 1;
  });
}
