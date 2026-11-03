#!/usr/bin/env node
// Fetches every input in pipeline/inputs.json into <out>/, each from the first mirror that answers, in
// HTTP Range segments with retries (pipeline/lib/segmented-download.mjs), verifies it (the publisher's
// sha256 or sha1 where there is one; every zip is tested whole), and writes <out>/SHA256SUMS plus a listing
// of every zip and the authors and licences the CC-BY packs name. Runs in CI only (vendor.yml).
//
//   node pipeline/vendor.mjs <out>
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { download } from './lib/segmented-download.mjs';

const out = path.resolve(process.argv[2] ?? 'inputs');
const spec = JSON.parse(fs.readFileSync(new URL('./inputs.json', import.meta.url), 'utf8'));
fs.mkdirSync(path.join(out, 'listing'), { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sha256 = (file) => {
  const h = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  const buf = Buffer.alloc(1 << 22);
  let n;
  while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) h.update(buf.subarray(0, n));
  fs.closeSync(fd);
  return h.digest('hex');
};

async function text(url) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(30000) });
      if (res.ok) return (await res.text()).trim();
      if (res.status < 500 && res.status !== 429 && res.status !== 408) throw new Error(`HTTP ${res.status} for ${url}`);
    } catch (e) {
      if (attempt > 20) throw e;
    }
    await sleep(Math.min(60000, 1000 * 2 ** attempt));
  }
}

async function fetchOne(item) {
  const dest = path.join(out, item.name);
  if (fs.existsSync(dest)) return;
  let hash;
  if (item.sha256) hash = { algo: 'sha256', hex: item.sha256 };
  else if (item.sha1url) hash = { algo: 'sha1', hex: (await text(item.sha1url)).split(/\s+/)[0] };
  // Mirrors are tried in turn, round after round, while the errors are transient. Each mirror keeps its
  // own partial file and progress, so coming back to it resumes where it stopped.
  for (let round = 0; ; round++) {
    for (const [i, url] of item.urls.entries()) {
      const tmp = path.join(out, `.m${i}`, item.name);
      fs.mkdirSync(path.dirname(tmp), { recursive: true });
      try {
        const r = await download({
          url,
          dest: tmp,
          hash,
          connections: 4,
          segmentBytes: 8 * 1024 * 1024,
          maxRetries: 6,
          stallMs: 45000,
          log: (line) => console.log(`[${item.name} m${i}] ${line}`),
        });
        fs.renameSync(tmp, dest);
        console.log(`[${item.name}] done from ${url} (${(r.size / 1e6).toFixed(1)} MB, ${r.requests} requests)`);
        return;
      } catch (e) {
        console.log(`::warning::${item.name}: ${url} failed: ${e.message}`);
        if (e.message?.includes('mismatch')) fs.rmSync(tmp, { force: true });
      }
    }
    await sleep(Math.min(120000, 5000 * 2 ** round));
  }
}

// A few at a time: the MakeHuman host is slow and its second mirror flaky.
const queue = [...spec.inputs];
await Promise.all(
  Array.from({ length: 3 }, async () => {
    for (let item; (item = queue.shift()); ) await fetchOne(item);
  }),
);
for (const d of fs.readdirSync(out)) if (/^\.m\d+$/.test(d)) fs.rmSync(path.join(out, d), { recursive: true, force: true });

// Whole-file checks, listings and credits.
const credits = [];
for (const item of spec.inputs) {
  const file = path.join(out, item.name);
  if (item.name.endsWith('.zip')) {
    execFileSync('unzip', ['-tqq', file], { stdio: 'inherit' });
    const listing = execFileSync('unzip', ['-Z1', file], { maxBuffer: 64 << 20 }).toString();
    fs.writeFileSync(path.join(out, 'listing', `${item.name}.txt`), listing);
  }
  if (item.license === 'CC-BY') {
    // MakeHuman assets carry their author and licence in their own text files (mhclo, mhmat).
    let meta = '';
    try {
      meta = execFileSync('sh', ['-c', `unzip -p "${file}" '*.mhclo' '*.mhmat' '*.thumb.txt' '*.txt' 2>/dev/null | grep -a -i -E '^(name|author|license|homepage) ' | sort -u`], { maxBuffer: 64 << 20 }).toString();
    } catch {}
    credits.push(`## ${item.name}\n\n${item.urls[0]}\n\n\`\`\`\n${meta.trim()}\n\`\`\`\n`);
  }
}
fs.writeFileSync(path.join(out, 'listing', 'credits.md'), credits.join('\n'));

const names = spec.inputs.map((i) => i.name).sort();
const sums = names.map((n) => `${sha256(path.join(out, n))}  ${n}`).join('\n') + '\n';
fs.writeFileSync(path.join(out, 'SHA256SUMS'), sums);
console.log(sums);
