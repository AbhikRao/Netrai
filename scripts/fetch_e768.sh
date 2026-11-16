#!/usr/bin/env bash
# Download the E_768 research weights (143 MB) from the project's GitHub Release and verify them against the SHA-256
# recorded in the committed deployment bundle. Usage: scripts/fetch_e768.sh [--with-gradcam] [release-tag]   (default tag: e768-v1)
# --with-gradcam also installs the two PyTorch member files (2 x 75 MB, hash-checked against torch_members.json) that the
# Python pipeline needs for E_768 Grad-CAM (the ONNX graph has no gradients). MATLAB computes Grad-CAM from the ONNX file itself.
set -euo pipefail
with_gradcam=0
if [ "${1:-}" = "--with-gradcam" ]; then with_gradcam=1; shift; fi
tag="${1:-e768-v1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/matlab/variant_R3_E_768/netrai_fold0.onnx"
bundle="$root/matlab/variant_R3_E_768/netrai_bundle.json"
url="${E768_URL:-https://github.com/AbhikRao/Netrai/releases/download/${tag}/netrai_E_768.onnx}"   # E768_URL overrides (testing)
expected="$(python3 -c "import json,sys; print(next(a['sha256'] for a in json.load(open(sys.argv[1]))['artifacts'] if a['role']=='model'))" "$bundle")"
if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$expected" ]; then
  echo "E_768 weights already present and verified: $dest"     # only the download is skipped; --with-gradcam still runs below
else
  tmp="$(mktemp "${dest}.XXXXXX.part")"
  trap 'rm -f "$tmp"' EXIT
  echo "Downloading $url"
  curl --fail --location --retry 3 --output "$tmp" "$url"
  actual="$(sha256sum "$tmp" | cut -d' ' -f1)"
  if [ "$actual" != "$expected" ]; then
    echo "SHA-256 mismatch: expected $expected, got $actual - not installing." >&2; exit 1
  fi
  mv "$tmp" "$dest"; trap - EXIT
  echo "Installed and verified: $dest"
fi
echo "Read the weights' terms in docs/MODEL_CARD.md before use."
if [ "$with_gradcam" = 1 ]; then
  manifest="$root/matlab/variant_R3_E_768/torch_members.json"
  mkdir -p "$root/matlab/variant_R3_E_768/torch"
  python3 -c "import json,sys; [print(m['file'], m['sha256']) for m in json.load(open(sys.argv[1]))['members'].values()]" "$manifest" |
  while read -r file sha; do
    target="$root/matlab/variant_R3_E_768/torch/$file"
    if [ -f "$target" ] && [ "$(sha256sum "$target" | cut -d' ' -f1)" = "$sha" ]; then echo "already verified: $file"; continue; fi
    part="$(mktemp "${target}.XXXXXX.part")"
    curl --fail --location --retry 3 --output "$part" "${E768_MEMBER_URL_BASE:-https://github.com/AbhikRao/Netrai/releases/download/${tag}}/$file"
    if [ "$(sha256sum "$part" | cut -d' ' -f1)" != "$sha" ]; then rm -f "$part"; echo "SHA-256 mismatch for $file - not installing." >&2; exit 1; fi
    mv "$part" "$target"; echo "Installed and verified: $target"
  done
fi
