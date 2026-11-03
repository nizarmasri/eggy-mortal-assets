# The Mortal Eggy asset pack

What `build-fighters.yml` publishes, and everything the game looks up by name. One pack is one GitHub
release named after its id, `m1-<8 hex>`; nothing in a released pack ever changes (a new build is a new id).

## Release layout

A release `m1-xxxxxxxx` holds:

| Asset | What |
|---|---|
| `mortal-m1-xxxxxxxx.tar.01` … `.NN` | The shipped tree as one tar, cut in 10 MiB parts (the last one shorter). Concatenated in order they are the tar byte for byte. |
| `parts.json` | The part list (below). Fetch this first. |
| `manifest.json` | Every shipped file with its size, sha256 and set (below). Also inside the tar, first. |
| `fighters-<id>.glb`, `fighters-<id>.rig.json`, … | Each shipped file on its own, its path with `/` → `-`, for fetching one fighter without the tar. |
| `common-basis_transcoder.js`, `common-basis_transcoder.wasm` | Likewise (the common set). |
| `preview-<id>_*.png` | Review renders. Not shipped, not in the manifest or the tar. |
| `SHA256SUMS` | sha256 of every asset above. |

`parts.json`:

```json
{
  "pack": "m1-xxxxxxxx",
  "file": "mortal-m1-xxxxxxxx.tar",
  "sha256": "<sha256 of the whole tar>",
  "bytes": 12345678,
  "url": "https://github.com/nizarmasri/eggy-mortal-assets/releases/download/m1-xxxxxxxx/",
  "parts": [{ "name": "mortal-m1-xxxxxxxx.tar.01", "bytes": 10485760, "sha256": "…" }]
}
```

Each part is fetched from `url + name` (in Range segments, resumable, each part verified against its sha256
before it is kept); the joined tar is verified against `sha256`.

The tar is deterministic: GNU format, owner and group 0, mtime 0, mode `a+rX,u+w,go-w`, `manifest.json` first,
then every file in byte order of its path. The same tree always makes the same tar and the same parts.

`manifest.json`:

```json
{
  "pack": "m1-xxxxxxxx",
  "files": [{ "name": "fighters/reiji.glb", "size": 3031234, "sha256": "…", "set": "fighter:reiji" }]
}
```

`pack` is `m1-` + the first 8 hex of sha256 over the lines `<name> <size> <sha256>\n` of every file, in
manifest order. Sets: `common` (needed by everything), `fighter:<id>`, `stage:<id>`, `music`. A phone can
fetch only the sets a match needs, file by file from the per-file assets, and check each against the manifest.

## The shipped tree

```
manifest.json
common/basis_transcoder.js        three r186's Basis transcoder (KTX2Loader.setTranscoderPath)
common/basis_transcoder.wasm
fighters/<id>.glb                 the fighter
fighters/<id>.rig.json            its skeleton at rest
fighters/<id>.json                its meta: triangles, textures, nodes, file sizes
fighters/<id>.mask_crack.ktx2     the mask's crack map
```

Fighter ids: `reiji`, `mara`, `elyema`, `bia`, `duke`, `sorn`, `han`, `gu`.

## A fighter's GLB

Loading needs `GLTFLoader` with `MeshoptDecoder` (`EXT_meshopt_compression` is required) and `KTX2Loader`
(`KHR_texture_basisu`; colour and ORM maps are ETC1S, normal maps UASTC). Positions, normals and UVs are
quantised (`KHR_mesh_quantization`); `_RESTPOS` is not. No animations, no morph targets, no cameras or lights.

Space: metres, Y up, the fighter facing +Z, standing on y = 0, centred on x = z = 0. Left is +X. The rest
(bind) pose is MakeHuman's: upper arms out and down about 45°, elbows bent about 45° forward so the hands
sit in front of the hips, legs straight and a little apart. Every fighter shares it.

### Skeleton

One skin, 53 joints, UE4 mannequin names, shared by every fighter and every animation:

```
Root
└ pelvis
  ├ spine_01 ─ spine_02 ─ spine_03
  │   ├ neck_01 ─ head
  │   └ clavicle_{l,r} ─ upperarm ─ lowerarm ─ hand
  │                         ├ thumb_01 ─ thumb_02 ─ thumb_03
  │                         ├ index_01 ─ index_02 ─ index_03
  │                         ├ middle_01 ─ middle_02 ─ middle_03
  │                         ├ ring_01 ─ ring_02 ─ ring_03
  │                         └ pinky_01 ─ pinky_02 ─ pinky_03
  └ thigh_{l,r} ─ calf ─ foot ─ ball
```

(Every arm and leg joint carries its side suffix: `upperarm_l`, `index_02_r`, `ball_l` …) Bone rolls are
normalised to Reiji's, so one animation clip drives every fighter: a rotation means the same thing on each.
Proportions differ; retarget by copying rotations and scaling the `pelvis` translation by the rig's
`pelvis` height ratio. The head has no jaw, eye or face bones (the mask covers the face).

Every skinned mesh uses at most 4 influences per vertex.

### Nodes

| Node | Mesh | Skinned | Parent | Material | Shown |
|---|---|---|---|---|---|
| `Armature` | – | – | scene root | – | – |
| the 53 joints | – | – | each other, `Root` under `Armature` | – | – |
| `body` | skin, ~18k tris | yes | `Armature` | `skin` | yes |
| `eyes` | low-poly eyes | yes | `Armature` | `eyes` | yes (behind the mask's slits) |
| `teeth` | a tiny set behind the lips | yes | `Armature` | `teeth` | yes |
| `gear` | every garment, the hair and the mask strap (El Yema: his hood and its laces, no strap) in one mesh | yes | `Armature` | `gear` | yes |
| `cape` | El Yema's cape | yes | `Armature` | `gear` | yes (El Yema only) |
| `fitas` | Bia's mask ribbons | yes | `Armature` | `gear` | yes (Bia only) |
| `gourd` | Old Gu's gourd and its cord | yes | `Armature` | `gear` | yes (Gu only) |
| `mask` | the whole egg mask | yes, all on `head` | `Armature` | `mask` | yes |
| `mask_shard_00` … `_NN` | the mask broken in 25–40 pieces | no | joint `head` | `mask` | no |
| `cap_<cut>_p`, `cap_<cut>_d` | gore caps (below) | yes | `Armature` | `gore` | no |
| `skeleton` | stylised skeleton, ~7k tris | yes, rigid | `Armature` | `gore` | no |
| `bone_frac_<part>_<n>` | pre-broken bone pieces | no | their joint | `gore` | no |

Every node that starts hidden has `extras.hidden = true`; the game sets `visible = false` on load.

**Mask shards.** `mask_shard_NN` together are exactly the mask (same outside surface, same UVs, same
material) cut into Voronoi cells, each a closed solid with its own break faces. In their rest place they
sit on the `head` joint like the whole mask does, so a break is: hide `mask`, show the shards, detach each
from `head` keeping its world transform, and throw it. The break faces use the bisque part of the texture.

**Chain pieces.** `cape` and `fitas` hang from a spring chain the game simulates. Each vertex carries
`_CHAIN` (float, 0 where the piece is fixed → 1 at its free end) and `_CHAINU` (float, −1 … 1 across it);
`fitas` also has `_STRIP` (float, which ribbon: 0, 1, 2). At rest they are skinned and look right without the
simulation: the cape with the weights of the back under it (the collar on `spine_03` and the clavicles,
the rest on `spine_03`), each ribbon on `head` at the knot, blending through `neck_01` to `spine_03` along
its length. The `gourd` hangs
100% on `pelvis`; it has no chain attributes.

### Gore

Cuts: `neck`, `waist`, `shoulder_l`, `shoulder_r`, `elbow_l`, `elbow_r`, `hip_l`, `hip_r`, `knee_l`,
`knee_r`. At each, `body` and `gear` are split along a loop across the bone, and both copies of the loop are
weighted 100% to that cut's bone, so the seam never opens while the limb is attached.

`body` has a vertex attribute `_SEGMENT` (float, whole numbers) saying which piece each vertex belongs to:

| id | segment | id | segment |
|---|---|---|---|
| 0 | torso | 6 | thigh_l |
| 1 | head | 7 | shin_l |
| 2 | upperarm_l | 8 | thigh_r |
| 3 | forearm_l | 9 | shin_r |
| 4 | upperarm_r | 10 | pelvis |
| 5 | forearm_r | | |

`gear` carries the same attribute (the separate `cape`, `fitas` and `gourd` do not: they hang from the
torso, the head and the pelvis and go with whichever piece holds those). To sever at a cut, hide (in the shader, by `_SEGMENT`) the segments on the
far side on the original mesh and show them on a detached copy, then show `cap_<cut>_p` (the stump's cap,
facing away from the body) and `cap_<cut>_d` (the severed piece's cap). The caps fit the loop exactly.

| Cut | Bone | Stump keeps | Severed piece |
|---|---|---|---|
| `neck` | `neck_01` | torso | head (1) |
| `waist` | `spine_01` | pelvis (10) | torso and everything above (0–5) |
| `shoulder_l/r` | `upperarm_l/r` | torso | upperarm, forearm (2–3 / 4–5) |
| `elbow_l/r` | `lowerarm_l/r` | upperarm | forearm (3 / 5) |
| `hip_l/r` | `thigh_l/r` | pelvis | thigh, shin (6–7 / 8–9) |
| `knee_l/r` | `calf_l/r` | thigh | shin (7 / 9) |

`body` also has `_RESTPOS` (float VEC3, not quantised): every vertex's rest position, for wound decals
and blood placed in the body's own space.

The `gore` material's texture is a 256×256 strip: skin rim, fat, muscle, bone and marrow in bands along u.

`skeleton`: every bone a rigid piece on its joint (skull, jaw, spine, ribs, pelvis, arm and leg bones,
hands and feet). `bone_frac_<part>_<n>` are the pieces a break shows instead, each parented to its joint:

| Part | Pieces | Joint |
|---|---|---|
| `skull` | 4 | `head` |
| `jaw` | 2 | `head` |
| `ribs` | 6 | `spine_02`, `spine_03` |
| `spine` | 3 | `spine_01`, `spine_03`, `neck_01` |
| `forearm_l`, `forearm_r` | 2 each | `lowerarm_l/r` |
| `shin_l`, `shin_r` | 2 each | `calf_l/r` |

### Materials

All opaque, metallic 0 unless stated. Textures are 1024² unless stated.

| Material | baseColor | metallicRoughness and occlusion (one ORM map) | normal |
|---|---|---|---|
| `skin` | skin albedo (the scalp fade painted in) | R occlusion, G roughness, B cavity (not metal: metallicFactor is 0) | baked body detail |
| `gear` | the garment atlas | R occlusion, G roughness, B 0 | fabric, leather and tape detail |
| `mask` | the painted design | R 1, G roughness (the glaze), B metalness; metallicFactor 1 so B is the metal | 512², the paint relief, chips and dents |
| `gore` | 256² gore strip | R 1, G roughness | – |
| `eyes` | 256² eye texture | – | – |
| `teeth` | flat colour | – | – |

The `skin` ORM's B is cavity: a game shader can darken pores and creases with it; three.js ignores it.

### The crack map

`fighters/<id>.mask_crack.ktx2`: 512², UASTC with mipmaps, linear (not sRGB), in the mask's UV space.

- R: distance from the texel to the nearest shard wall, in units of 20 mm, clipped to 1. `R < 0.05` is a
  hairline crack on the surface. Grow cracks over time by raising the threshold.
- G: a random value per shard cell (0–1), for staggering which shards crack first.
- B: the cell's index: `(index + 0.5) / 64`, so `floor(B * 64)` is the shard number (`mask_shard_NN`).

Texels off the egg have R = 1.

### Mask UVs

One map, three regions: the outside of the egg on v 0.25 … 1 (a metric projection around the egg, the face
centred at u = 0.5), the inside on u 0 … 0.62, v 0 … 0.24, and the rim, slit walls and break faces on
u 0.65 … 1, v 0 … 0.24 (plain bisque).

## rig.json

```json
{
  "v": 1,
  "fighter": "reiji",
  "space": "glTF: metres, Y up, the fighter facing +Z; …",
  "order": ["Root", "pelvis", "…"],
  "bones": [{
    "name": "pelvis",
    "parent": "Root",
    "t": [0, 0.98, 0.01], "q": [x, y, z, w], "s": [1, 1, 1],
    "world": [0, 0.98, 0.01],
    "length": 0.11
  }],
  "topology": "<sha256 of [name, parent] in order>",
  "hash": "<sha256 of [name, parent, t, q, s, length] in order>"
}
```

`order` is the skin's joint order (the index a `JOINTS_0` value points at). `t`, `q`, `s` are each joint's
rest transform relative to its parent, `world` its rest position, `length` the Blender bone's length (head
to tail, metres). Numbers are rounded to 6 decimals. `topology` is the same for every fighter; `hash`
changes when the fighter's proportions do.

## fighters/<id>.json

The meta file: `height_m`, `triangles` (per mesh, and `visible`: body + eyes + teeth + gear + mask),
`shards`, `textures` (name, mime, bytes, size), `nodes` (name, mesh, skinned, parent), `files` (the GLB's
bytes, the rig's hash, the crack map's bytes) and `skeleton` (`joints: 53`, `topology`).

## In the game

The game repo pins one pack id. It downloads `parts.json` from the release, fetches the parts it needs
with the segmented downloader (or, on a phone, the per-file assets of the sets the match needs, through the
chunk loader in `client/kit/pack/`), verifies every file against `manifest.json`, and points
`KTX2Loader.setTranscoderPath` at the `common` set's transcoder.
