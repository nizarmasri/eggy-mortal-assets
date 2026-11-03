# eggy-mortal-assets

Fighter and stage assets for Mortal Eggy, Eggy's Tekken-style 3D fighter. Everything is built by GitHub
Actions from CC0 inputs (two credited CC-BY packs); nothing is made by hand and nothing heavy runs on a
laptop.

| Workflow | What |
|---|---|
| `vendor.yml` | Downloads every input once (Blender 4.5.15, MPFB 2.0.18, Cell Fracture 0.2.1, KTX-Software 4.4.2, the MakeHuman asset packs), verifies each and publishes them as release `inputs-v1` with `SHA256SUMS`. Builds read only from there. |
| `build-fighters.yml` | One job per fighter (`fighters`: ids or `all`): body, gear, mask, shards and gore parts in Blender (`pipeline/blender/fighter.py`), then compression and checks (`pipeline/post.mjs`). Review renders go up as the `previews` artifact. With `publish`, the pack is released as `m1-<hash>` (`pipeline/publish.mjs`). |

The pack's layout, every node name the renderer looks up and the release format are in [PACK.md](PACK.md).
Fighter presets are `fighters/<id>.json`. Credits: [CREDITS.md](CREDITS.md).
