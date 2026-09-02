#!/usr/bin/env bash
set -euo pipefail

FILE_ID="1q5vP5Xkhw5yXdUJ-5ySNeoEvEsZgpNuO"
EXPECTED_SHA256="caeb77c353fa0f9e489cda0442943faf2043ee8225605777b2d3aac2c0779bca"
CHECKPOINT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/checkpoints"
OUTPUT="${CHECKPOINT_DIR}/jetclass_transformer.pt"
TEMP="${OUTPUT}.download"

mkdir -p "$CHECKPOINT_DIR"
rm -f "$TEMP"

uvx --from gdown gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$TEMP"

if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256="$(sha256sum "$TEMP" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    actual_sha256="$(shasum -a 256 "$TEMP" | awk '{print $1}')"
else
    echo "ERROR: neither sha256sum nor shasum is available" >&2
    rm -f "$TEMP"
    exit 1
fi

if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: downloaded model checksum does not match" >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $actual_sha256" >&2
    rm -f "$TEMP"
    exit 1
fi

mv "$TEMP" "$OUTPUT"
echo "Downloaded and verified: $OUTPUT"
