#!/usr/bin/env python3
"""Extract the eight rectus-muscle MCSA and volume measures from generated masks."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from morphometry import MUSCLE_LABELS, extract_mask_morphometry


def extract_task(task: tuple[str, str]) -> dict[str, object] | None:
    file_id, mask_path = task
    try:
        return extract_mask_morphometry(mask_path, file_id)
    except Exception as exc:
        print(f"[Warning] Failed {file_id} ({mask_path}): {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract JMS morphometry from generated TotalSegmentator masks."
    )
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_paths = sorted(args.mask_dir.glob("*.nii.gz"))
    if not mask_paths:
        raise FileNotFoundError(f"No .nii.gz masks found in {args.mask_dir}")

    tasks = [(path.name.split("_", 1)[0].split(".", 1)[0], str(path)) for path in mask_paths]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(
                executor.map(extract_task, tasks),
                total=len(tasks),
                desc="Generated-mask morphometry",
            )
        )

    frame = pd.DataFrame(result for result in results if result is not None)
    if frame.empty:
        raise RuntimeError("No generated-mask measurements were extracted")
    if frame["file_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["file_id"].duplicated(), "file_id"].tolist()
        raise ValueError(f"Duplicate patient IDs: {duplicate_ids}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"[Done] Saved {len(frame)} patients to {args.output_csv}")
    for muscle in MUSCLE_LABELS:
        valid = int(frame[f"{muscle}_vol"].notna().sum())
        print(f"{muscle}: {valid}/{len(frame)} ({valid / len(frame) * 100.0:.2f}%)")


if __name__ == "__main__":
    main()
