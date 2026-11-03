#!/usr/bin/env node
// Assembles the pack from the fighters' builds (and, later, stages and music): the shipped tree, its
// manifest.json, one deterministic tarball cut in 10 MiB parts, the part list, and the release layout.
// Pack id: m1-<first 8 hex of sha256 over the manifest's file list> (name, size and sha256 of every file).
//
//   node pipeline/publish.mjs --in <dir> --rel <dir> [--previews <dir>] [--repo nizarmasri/eggy-mortal-assets]
// <dir> holds the shipped tree: fighters/<id>.glb|.rig.json|.json|.mask_crack.ktx2, common/..., stages/..., music/...
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';

const argv = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : fallback;
};
const tree = path.resolve(opt('in'));
const rel = path.resolve(opt('rel', 'rel'));
const previews = opt('previews') ? path.resolve(opt('previews')) : null;
const repo = opt('repo', 'nizarmasri/eggy-mortal-assets');
const PART = 10 * 1024 * 1024;

const shaFile = (f) => crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
fs.rmSync(rel, { recursive: true, force: true });
fs.mkdirSync(rel, { recursive: true });

// ---- The manifest: every shipped file with its set ----
const files = [];
(function walk(dir) {
  for (const e of fs.readdirSync(path.join(tree, dir), { withFileTypes: true })) {
    const p = dir ? `${dir}/${e.name}` : e.name;
    if (e.isDirectory()) walk(p);
    else if (p !== 'manifest.json') files.push(p);
  }
})('');
files.sort();
const setOf = (p) => {
  let m;
  if ((m = /^fighters\/([a-z0-9]+)\./.exec(p))) return `fighter:${m[1]}`;
  if ((m = /^stages\/([a-z0-9]+)[./]/.exec(p))) return `stage:${m[1]}`;
  if (p.startsWith('music/')) return 'music';
  return 'common';
};
const entries = files.map((name) => {
  const f = path.join(tree, name);
  return { name, size: fs.statSync(f).size, sha256: shaFile(f), set: setOf(name) };
});
const listHash = crypto.createHash('sha256').update(entries.map((e) => `${e.name} ${e.size} ${e.sha256}\n`).join('')).digest('hex');
const pack = `m1-${listHash.slice(0, 8)}`;
const manifest = { pack, files: entries };
fs.writeFileSync(path.join(tree, 'manifest.json'), JSON.stringify(manifest, null, 1) + '\n');

// ---- The tarball: same bytes every run (sorted names, no owners, no times) ----
const list = path.join(rel, '.files.txt');
fs.writeFileSync(list, ['manifest.json', ...files].join('\n') + '\n');
const file = `mortal-${pack}.tar`;
const tarPath = path.join(rel, '.' + file);
execFileSync('tar', ['--create', '--file', tarPath, '--directory', tree, '--format=gnu', '--owner=0', '--group=0', '--numeric-owner',
  '--mtime=@0', '--mode=a+rX,u+w,go-w', '--no-recursion', '--files-from', list]);
const tarSha = shaFile(tarPath);
const bytes = fs.statSync(tarPath).size;
const parts = [];
{
  const fd = fs.openSync(tarPath, 'r');
  const buf = Buffer.alloc(PART);
  for (let i = 0, off = 0; off < bytes; i++, off += PART) {
    const n = fs.readSync(fd, buf, 0, Math.min(PART, bytes - off), off);
    const name = `${file}.${String(i + 1).padStart(2, '0')}`;
    fs.writeFileSync(path.join(rel, name), buf.subarray(0, n));
    parts.push({ name, bytes: n, sha256: crypto.createHash('sha256').update(buf.subarray(0, n)).digest('hex') });
  }
  fs.closeSync(fd);
}
fs.rmSync(tarPath);
fs.rmSync(list);
const lock = { pack, file, sha256: tarSha, bytes, url: `https://github.com/${repo}/releases/download/${pack}/`, parts };
fs.writeFileSync(path.join(rel, 'parts.json'), JSON.stringify(lock, null, 2) + '\n');
fs.copyFileSync(path.join(tree, 'manifest.json'), path.join(rel, 'manifest.json'));

// ---- Every file on its own too (flat names), and the review renders (not shipped) ----
for (const e of entries) fs.copyFileSync(path.join(tree, e.name), path.join(rel, e.name.replace(/\//g, '-')));
if (previews && fs.existsSync(previews)) {
  for (const f of fs.readdirSync(previews)) if (f.endsWith('.png')) fs.copyFileSync(path.join(previews, f), path.join(rel, `preview-${f}`));
}
const names = fs.readdirSync(rel).filter((f) => f !== 'SHA256SUMS' && !f.startsWith('.')).sort();
fs.writeFileSync(path.join(rel, 'SHA256SUMS'), names.map((f) => `${shaFile(path.join(rel, f))}  ${f}`).join('\n') + '\n');

const bySet = {};
for (const e of entries) bySet[e.set] = (bySet[e.set] ?? 0) + e.size;
console.log(`${pack}: ${(bytes / 1e6).toFixed(2)} MB in ${parts.length} parts`);
console.log(JSON.stringify(bySet, null, 1));
if (process.env.GITHUB_OUTPUT) fs.appendFileSync(process.env.GITHUB_OUTPUT, `pack=${pack}\n`);
if (process.env.GITHUB_STEP_SUMMARY) {
  fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, [`### ${pack}`, '', '| set | bytes |', '|---|---|', ...Object.entries(bySet).map(([k, v]) => `| ${k} | ${v} |`), ''].join('\n'));
}
