#!/usr/bin/env python3
"""
Remove ONNX models whose metadata has "epoch": 1.

Usage:
    python scripts/remove_epoch1_models.py [--checkpoints-dir backend/checkpoints] [--delete] [--git-rm]

By default the script performs a dry run and only prints the files that would be removed.
Pass `--delete` to actually unlink files. Pass `--git-rm` together with `--delete` to also run
`git rm` on the removed files so they are staged for commit.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple


logger = logging.getLogger("remove_epoch1_models")


def find_metadata_files(checkpoints_dir: Path) -> Iterable[Path]:
    return sorted(checkpoints_dir.glob("*.onnx_metadata.json"))


def read_epoch(meta_path: Path) -> int | None:
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("epoch")
    except Exception as e:
        logger.warning(f"Failed to read {meta_path}: {e}")
        return None


def corresponding_onnx_path(meta_path: Path) -> Path:
    # convert 'foo.onnx_metadata.json' -> 'foo.onnx'
    name = meta_path.name
    if name.endswith(".onnx_metadata.json"):
        onnx_name = name.replace(".onnx_metadata.json", ".onnx")
    else:
        onnx_name = name + ".onnx"
    return meta_path.parent / onnx_name


def collect_epoch1_files(checkpoints_dir: Path) -> List[Tuple[Path, Path]]:
    found: List[Tuple[Path, Path]] = []
    for meta in find_metadata_files(checkpoints_dir):
        epoch = read_epoch(meta)
        if epoch == 1:
            onnx = corresponding_onnx_path(meta)
            found.append((meta, onnx))
    return found


def delete_files(
    pairs: List[Tuple[Path, Path]],
) -> int:
    removed = 0
    for meta, onnx in pairs:
        if not onnx.exists():
            logger.info(f"Model file does not exist (skipped): {onnx}")
            continue
        try:
            onnx.unlink()
            meta.unlink()
            removed += 1
        except Exception as e:
            logger.error(f"Failed to remove files: {e}")

    return removed


def main() -> int:

    checkpoints_dir = Path("backend/checkpoints")
    if not checkpoints_dir.exists():
        logger.error(f"Checkpoints directory does not exist: {checkpoints_dir}")
        return 2

    pairs = collect_epoch1_files(checkpoints_dir)

    if not pairs:
        logger.info("No metadata files with 'epoch': 1 found.")
        return 0

    logger.info(f"Found {len(pairs)} metadata file(s) with epoch=1:")
    for meta, onnx in pairs:
        logger.info(f"  - {meta}  (model: {onnx})")

    removed = delete_files(pairs)

    logger.info(f"Removed metadata files: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
