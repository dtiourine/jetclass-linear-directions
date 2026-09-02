#!/bin/bash
BASE="https://zenodo.org/records/6619768/files"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

mkdir -p "$DATA_DIR"

download() {
    local name="$1"
    aria2c -x 16 -s 16 -d "$DATA_DIR" -o "${name}" \
        "${BASE}/${name}?download=1"
}

# Training set (100M jets, split across 10 parts)
for i in $(seq 0 9); do
    download "JetClass_Pythia_train_100M_part${i}.tar"
done

# Validation set (5M jets)
download "JetClass_Pythia_val_5M.tar"

# Test set (20M jets)
download "JetClass_Pythia_test_20M.tar"
