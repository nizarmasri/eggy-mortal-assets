#!/usr/bin/env bash
# CI only: Blender 4.5, MPFB and Cell Fracture as user extensions, the MakeHuman packs in MPFB's user data and
# KTX-Software, all from our own release inputs-v1 (verified against its SHA256SUMS). Skips what the cache
# already restored.
set -euo pipefail
REPO="${GITHUB_REPOSITORY:-nizarmasri/eggy-mortal-assets}"
REL=inputs-v1
B="$HOME/blender"
HERE="$(cd "$(dirname "$0")" && pwd)"

sudo apt-get update -qq
sudo apt-get install -y -qq libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libxkbcommon0 libsm6 libice6 >/dev/null

if [ ! -x "$B/blender" ] || [ ! -x "$HOME/ktx/bin/ktx" ]; then
  rm -rf "$B" "$HOME/ktx" "$HOME/.config/blender"
  mkdir -p "$HOME/in"
  cd "$HOME/in"
  for i in 1 2 3 4 5 6; do
    if gh release download "$REL" -R "$REPO" --clobber -p SHA256SUMS -p 'blender-*' -p 'add-on-*' -p 'KTX-*' -p '*_cc0.zip' -p '*_ccby.zip'; then break; fi
    echo "download failed, retry $i"; sleep $((i * 10))
  done
  sha256sum -c SHA256SUMS
  tar -xJf blender-4.5.15-linux-x64.tar.xz -C "$HOME"
  mv "$HOME/blender-4.5.15-linux-x64" "$B"
  mkdir -p "$HOME/ktx"
  tar -xjf KTX-Software-4.4.2-Linux-x86_64.tar.bz2 -C "$HOME/ktx" --strip-components=1
  "$B/blender" --command extension install-file -r user_default -e "$HOME/in/add-on-mpfb-v2.0.18.zip"
  "$B/blender" --command extension install-file -r user_default -e "$HOME/in/add-on-cell-fracture-v0.2.1.zip"
  "$B/blender" -b --python-exit-code 1 -P "$HERE/blender/extract_packs.py" -- "$HOME"/in/*_cc0.zip "$HOME"/in/*_ccby.zip
  cd "$HOME"
  rm -rf "$HOME/in"
fi
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$HOME/ktx/bin" >> "$GITHUB_PATH"
  echo "LD_LIBRARY_PATH=$HOME/ktx/lib:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
fi
"$B/blender" --version | head -1
LD_LIBRARY_PATH="$HOME/ktx/lib" "$HOME/ktx/bin/ktx" --version
