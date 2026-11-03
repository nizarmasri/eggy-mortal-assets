"""Extract MakeHuman asset packs into MPFB's user data, the way MPFB's own installer does.

    blender -b -P pipeline/blender/extract_packs.py -- <pack.zip>...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mp  # noqa: E402

zips = sys.argv[sys.argv.index("--") + 1:]
target = mp.LocationService.get_user_data()
os.makedirs(target, exist_ok=True)
for z in zips:
    err = mp.AssetService.fix_and_extract_asset_pack_zip(z, target)
    assert err is None, f"{z}: {err}"
    print("[mortal] extracted", os.path.basename(z), "into", target, flush=True)
for sub in ("skins", "clothes", "eyes", "proxymeshes", "hair"):
    d = os.path.join(target, sub)
    print("[mortal]", sub, len(os.listdir(d)) if os.path.isdir(d) else 0, flush=True)
packs = os.path.join(target, "packs")
if os.path.isdir(packs):
    for f in sorted(os.listdir(packs)):
        if f.startswith(("pants03", "shoes02")):
            with open(os.path.join(packs, f)) as fh:
                print("[mortal] pack meta", f, json.dumps(json.load(fh))[:3000], flush=True)
