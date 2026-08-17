#!/usr/bin/env python3
"""Extract reference-CT rectus-muscle MCSA and volume from dataset masks."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from morphometry import MUSCLE_LABELS, extract_mask_morphometry


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "testing", "test", "evaluation"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported dataset JSON structure: {path}")


def extract_task(task: tuple[str, str]) -> dict[str, object] | None:
    file_id, mask_path = task
    try:
        return extract_mask_morphometry(mask_path, file_id)
    except Exception as exc:
        print(f"[Warning] Failed {file_id} ({mask_path}): {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract JMS morphometry from reference CT-derived masks."
    )
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--id-key", default="anon_id")
    parser.add_argument("--mask-key", default="label")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items(args.dataset_json)
    tasks: list[tuple[str, str]] = []
    for item in items:
        if args.id_key not in item or args.mask_key not in item:
            raise KeyError(
                f"Each dataset item must contain {args.id_key!r} and {args.mask_key!r}"
            )
        tasks.append((str(item[args.id_key]), str(item[args.mask_key])))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(
                executor.map(extract_task, tasks),
                total=len(tasks),
                desc="Reference-mask morphometry",
            )
        )

    frame = pd.DataFrame(result for result in results if result is not None)
    if frame.empty:
        raise RuntimeError("No reference-mask measurements were extracted")
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
