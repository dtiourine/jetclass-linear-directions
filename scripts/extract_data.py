"""
This script assumes the data directory (e.g., /data in root) contains the raw zip files downloaded directly from JetClass:

    JetClass_Pythia_train_100M_part0.tar ... part9.tar
    JetClass_Pythia_val_5M.tar
    JetClass_Pythia_test_20M.tar

It extracts these raw JetClass .tar downloads and places them in train/val/test directories in ./data:

    data/train/*.root
    data/val/*.root
    data/test/*.root

The data directory is read from jetdirections.config.ROOT_DATA_DIR.

Usage:
    python scripts/extract_data.py
    python scripts/extract_data.py --delete-tars
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

from jetdirections.config import ROOT_DATA_DIR


def extract_tar(tar_path: Path, target_dir: Path) -> None:
    """Flatten every .root file in the archive into target_dir by basename."""
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar:
        extracted = 0
        for member in tar:
            if not (member.isfile() and member.name.endswith(".root")):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            dest = target_dir / Path(member.name).name
            with src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            extracted += 1
            if extracted % 5 == 0:
                print(f"    ... {extracted} files", end="\r", flush=True)
        print(f"    extracted {extracted} files            ")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract raw JetClass .tar files into the layout the code expects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--delete-tars",
        action="store_true",
        help="Delete each .tar after it extracts successfully (frees disk).",
    )
    args = parser.parse_args()

    data_dir: Path = ROOT_DATA_DIR.resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    print(f"Data dir  : {data_dir}")
    print(f"Targets   : {train_dir} | {val_dir} | {test_dir}\n")

    # archive name -> target split dir; train is 10 parts, all flattened together
    jobs = [(f"JetClass_Pythia_train_100M_part{i}.tar", train_dir) for i in range(10)]
    jobs += [
        ("JetClass_Pythia_val_5M.tar", val_dir),
        ("JetClass_Pythia_test_20M.tar", test_dir),
    ]

    for tar_name, target_dir in jobs:
        tar_path = data_dir / tar_name
        if not tar_path.exists():
            print(f"WARNING: missing archive {tar_name} (skipping)")
            continue

        print(f"[{tar_name}] extracting -> {target_dir}")
        extract_tar(tar_path, target_dir)
        if args.delete_tars:
            tar_path.unlink()
            print(f"[{tar_name}] deleted")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
