# Credits

Every input is CC0 except two garments taken from the CC-BY packs below. Their authors and licences are
read from the garments' own `.mhclo` headers by the build (`credits` in each fighter's `stats.json`).

| Garment | Pack | Author | Licence | Used for |
|---|---|---|---|---|
| `elvs_male_swim_shorts1` ("elvs_male_shorts1") | MakeHuman community `pants03` | Elvaerwyn | CC-BY | Duke's trunks: refitted, recoloured and repainted |
| `punkduck_kill_bill_shoes` ("martialartshoes") | MakeHuman community `shoes02` | punkduck | CC BY 3.0 | Old Gu's shoes: refitted and recoloured |

Both come from the MakeHuman community asset library (http://www.makehuman.org/). Only their meshes are
used: the textures are baked new by this pipeline.

CC0 inputs: the MakeHuman system assets (base mesh, targets, rigs, eyes), `skins01`, `skins02`
(the skin textures and their normal and specular maps, including the Mindfront skins), `shirts01`,
`pants01`, `gloves01`, `masks01`. All are MakeHuman community asset packs, mirrored in release `inputs-v1`.
Every mask design, garment built from the body, pattern and the gore set are original to this pipeline.

Tools (used in CI, not shipped): Blender 4.5 (GPL), MPFB 2 (GPL code, CC0 data), Cell Fracture (GPL),
KTX-Software (Apache-2.0), glTF-Transform (MIT), meshoptimizer (MIT). Shipped in the `common` set: three.js
r186's Basis Universal transcoder (`basis_transcoder.js`, `.wasm`; Apache-2.0, from three's MIT package).
