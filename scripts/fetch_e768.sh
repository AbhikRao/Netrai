#!/usr/bin/env bash
# Download the E_768 research weights (143 MB) from the project's GitHub Release and verify them against the SHA-256
# recorded in the committed deployment bundle. Usage: scripts/fetch_e768.sh [release-tag]   (default: e768-v1)
set -euo pipefail
tag="${1:-e768-v1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/matlab/variant_R3_E_768/netrai_fold0.onnx"
bundle="$root/matlab/variant_R3_E_768/netrai_bundle.json"
url="${E768_URL:-https://github.com/AbhikRao/Netrai/releases/download/${tag}/netrai_E_768.onnx}"   # E768_URL overrides (testing)
expected="$(python3 -c "import json,sys; print(next(a['sha256'] for a in json.load(open(sys.argv[1]))['artifacts'] if a['role']=='model'))" "$bundle")"
if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$expected" ]; then
  echo "E_768 weights already present and verified: $dest"; exit 0
fi
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
echo "Read the weights' terms in docs/MODEL_CARD.md before use."
