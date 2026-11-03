#!/usr/bin/env node
// From Blender's raw GLB to the shipped fighter: attaches the ORM maps, forces opaque materials and metallic 0
// (metal only where a mask's ORM says so), adds the body's _RESTPOS attribute, compresses textures to KTX2
// (UASTC for normal maps, ETC1S for colour and ORM) and geometry with meshopt + quantisation, converts the
// mask's crack map to KTX2, checks the skin (53 joints, UE4 names) and writes rig.json and the meta file.
//
//   node pipeline/post.mjs <workdir> <id>
// reads  <workdir>/raw/<id>.glb, <workdir>/tex/*.png, <workdir>/<id>.bones.json, <workdir>/<id>.stats.json
// writes <workdir>/dist/fighters/<id>.glb, <id>.rig.json, <id>.json, <id>.mask_crack.ktx2
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { dedup, prune, quantize, reorder } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';

const require = createRequire(import.meta.url);
const work = path.resolve(process.argv[2]);
const id = process.argv[3];
const dist = path.join(work, 'dist', 'fighters');
fs.mkdirSync(dist, { recursive: true });
const tmp = path.join(work, 'post');
fs.mkdirSync(tmp, { recursive: true });

export const JOINTS = [
  'Root', 'pelvis', 'spine_01', 'spine_02', 'spine_03', 'neck_01', 'head',
  ...['l', 'r'].flatMap((s) => ['clavicle', 'upperarm', 'lowerarm', 'hand',
    'thumb_01', 'thumb_02', 'thumb_03', 'index_01', 'index_02', 'index_03', 'middle_01', 'middle_02', 'middle_03',
    'ring_01', 'ring_02', 'ring_03', 'pinky_01', 'pinky_02', 'pinky_03', 'thigh', 'calf', 'foot', 'ball'].map((n) => `${n}_${s}`)),
];

await MeshoptEncoder.ready;
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'meshopt.encoder': MeshoptEncoder,
  'meshopt.decoder': MeshoptDecoder,
});

const sha = (buf) => crypto.createHash('sha256').update(buf).digest('hex');
const r6 = (x) => Math.round(x * 1e6) / 1e6;
const cli = path.join(path.dirname(require.resolve('@gltf-transform/cli/package.json')), 'bin', 'cli.js');
const run = (args) => execFileSync(process.execPath, [cli, ...args], { stdio: 'inherit' });

// ---- 1. Materials, attributes ----
const doc = await io.read(path.join(work, 'raw', `${id}.glb`));
const root = doc.getRoot();
const ORM = { skin: 'body_orm.png', gear: 'gear_orm.png', mask: 'mask_orm.png' };
const METAL = { mask: 1 };
for (const mat of root.listMaterials()) {
  const name = mat.getName();
  mat.setAlphaMode('OPAQUE');
  mat.setAlphaCutoff(0.5);
  mat.setEmissiveFactor([0, 0, 0]);
  const file = ORM[name];
  if (file) {
    const tex = doc.createTexture(`${name}_orm`).setImage(fs.readFileSync(path.join(work, 'tex', file))).setMimeType('image/png').setURI(`${name}_orm.png`);
    mat.setMetallicRoughnessTexture(tex);
    mat.setOcclusionTexture(tex);
    mat.setOcclusionStrength(1);
    mat.setRoughnessFactor(1);
  } else {
    mat.setMetallicRoughnessTexture(null);
    mat.setOcclusionTexture(null);
  }
  mat.setMetallicFactor(METAL[name] ?? 0);
  if (name === 'gear') mat.setDoubleSided(true);
}
for (const node of root.listNodes()) {
  if (node.getName() !== 'body' || !node.getMesh()) continue;
  for (const prim of node.getMesh().listPrimitives()) {
    const pos = prim.getAttribute('POSITION');
    const rest = doc.createAccessor('restPos').setType('VEC3').setArray(new Float32Array(pos.getArray())).setBuffer(pos.getBuffer());
    prim.setAttribute('_RESTPOS', rest);
  }
}
await doc.transform(dedup(), prune({ keepLeaves: true, keepAttributes: true }));
const step1 = path.join(tmp, `${id}.1.glb`);
await io.write(step1, doc);

// ---- 2. Textures (KTX-Software's ktx, through gltf-transform) ----
const step2 = path.join(tmp, `${id}.2.glb`);
const step3 = path.join(tmp, `${id}.3.glb`);
run(['uastc', step1, step2, '--slots', 'normalTexture', '--level', '2', '--rdo', '--rdo-lambda', '2', '--zstd', '18']);
run(['etc1s', step2, step3, '--slots', '{baseColorTexture,metallicRoughnessTexture,occlusionTexture}', '--quality', '200']);

// ---- 3. Geometry: reorder, quantise (not _RESTPOS: it stays float, exact), meshopt ----
const doc3 = await io.read(step3);
await doc3.transform(
  reorder({ encoder: MeshoptEncoder }),
  quantize({ pattern: /^(POSITION|NORMAL|TANGENT|TEXCOORD_\d+|JOINTS_\d+|WEIGHTS_\d+|COLOR_\d+)$/, quantizePosition: 14, quantizeNormal: 10, quantizeTexcoord: 12 }),
);
doc3.createExtension(EXTMeshoptCompression).setRequired(true).setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.FILTER });
const glbPath = path.join(dist, `${id}.glb`);
await io.write(glbPath, doc3);

// ---- 4. The skin must be the 53-bone UE4-style skeleton ----
const r3 = doc3.getRoot();
const skins = r3.listSkins();
if (skins.length < 1) throw new Error('no skin');
const joints = skins[0].listJoints();
const names = joints.map((j) => j.getName());
const missing = JOINTS.filter((n) => !names.includes(n));
if (joints.length !== 53 || missing.length) throw new Error(`skin has ${joints.length} joints, missing ${missing}`);
for (const s of skins) if (s.listJoints().length !== 53) throw new Error(`a second skin with ${s.listJoints().length} joints`);

// ---- 5. rig.json ----
const side = JSON.parse(fs.readFileSync(path.join(work, `${id}.bones.json`), 'utf8'));
const length = Object.fromEntries(side.bones.map((b) => [b.name, b.length]));
const bones = joints.map((j) => {
  const parent = j.getParentNode();
  const w = j.getWorldMatrix();
  return {
    name: j.getName(),
    parent: parent && names.includes(parent.getName()) ? parent.getName() : null,
    t: j.getTranslation().map(r6),
    q: j.getRotation().map(r6),
    s: j.getScale().map(r6),
    world: [w[12], w[13], w[14]].map(r6),
    length: r6(length[j.getName()] ?? 0),
  };
});
const topology = sha(JSON.stringify(bones.map((b) => [b.name, b.parent])));
const hash = sha(JSON.stringify(bones.map((b) => [b.name, b.parent, b.t, b.q, b.s, b.length])));
const rig = {
  v: 1,
  fighter: id,
  space: 'glTF: metres, Y up, the fighter facing +Z; t/q/s are each joint\'s rest transform relative to its parent, world its rest position',
  order: names,
  bones,
  topology,
  hash,
};
fs.writeFileSync(path.join(dist, `${id}.rig.json`), JSON.stringify(rig, null, 1) + '\n');

// ---- 6. The crack map ----
const crackPng = path.join(work, 'tex', 'mask_crack.png');
const crackKtx = path.join(dist, `${id}.mask_crack.ktx2`);
execFileSync('ktx', ['create', '--format', 'R8G8B8_UNORM', '--assign-tf', 'linear', '--assign-primaries', 'none',
  '--encode', 'uastc', '--uastc-quality', '2', '--zstd', '18', '--generate-mipmap', crackPng, crackKtx], { stdio: 'inherit' });

// ---- 7. Meta ----
const stats = JSON.parse(fs.readFileSync(path.join(work, `${id}.stats.json`), 'utf8'));
const tris = {};
for (const mesh of r3.listMeshes()) {
  let n = 0;
  for (const p of mesh.listPrimitives()) n += (p.getIndices()?.getCount() ?? p.getAttribute('POSITION').getCount()) / 3;
  tris[mesh.getName()] = n;
}
const textures = r3.listTextures().map((t) => ({ name: t.getName(), mime: t.getMimeType(), bytes: t.getImage()?.byteLength ?? 0, size: t.getSize() }));
const nodes = r3.listNodes().filter((n) => !names.includes(n.getName())).map((n) => ({
  name: n.getName(),
  mesh: n.getMesh()?.getName() ?? null,
  skinned: !!n.getSkin(),
  parent: n.getParentNode()?.getName() ?? null,
}));
const shardNames = nodes.filter((n) => n.name.startsWith('mask_shard_')).map((n) => n.name);
const visible = ['body', 'eyes', 'teeth', 'gear', 'mask'].reduce((s, k) => s + (tris[k] ?? 0), 0);
const meta = {
  v: 1,
  id,
  height_m: stats.height_m,
  triangles: { ...tris, visible },
  shards: shardNames.length,
  textures,
  nodes,
  files: {
    glb: { name: `${id}.glb`, bytes: fs.statSync(glbPath).size },
    rig: { name: `${id}.rig.json`, hash },
    crack: { name: `${id}.mask_crack.ktx2`, bytes: fs.statSync(crackKtx).size },
  },
  skeleton: { joints: 53, topology },
};
fs.writeFileSync(path.join(dist, `${id}.json`), JSON.stringify(meta, null, 1) + '\n');
console.log(JSON.stringify({ glb: meta.files.glb.bytes, crack: meta.files.crack.bytes, triangles: meta.triangles, shards: meta.shards, textures }, null, 1));
