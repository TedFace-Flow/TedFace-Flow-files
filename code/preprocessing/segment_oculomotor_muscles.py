#!/usr/bin/env python3
"""Segment and align orbital structures in generated CT proxy volumes."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
)
from tqdm import tqdm


TARGET_SPACING = (0.5, 0.5, 1.0)
TARGET_SIZE = (512, 512, 128)
EXPECTED_TOTALSEGMENTATOR_VERSION = "2.15.0"


@dataclass
class Result:
    input_ct: str
    output_mask: str
    status: str
    error: str = ""


def align_z_axis_top(image: torch.Tensor, target_z: int) -> torch.Tensor:
    _, _, _, current_z = image.shape
    if current_z == target_z:
        return image
    if current_z < target_z:
        return torch.nn.functional.pad(
            image,
            (target_z - current_z, 0, 0, 0, 0, 0),
            mode="constant",
            value=0,
        )
    return image[..., -target_z:]


def paired_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=TARGET_SPACING,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-300,
                a_max=300,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(TARGET_SIZE[0], TARGET_SIZE[1], -1),
                mode="constant",
            ),
            Lambdad(
                keys=["image", "label"],
                func=lambda value: align_z_axis_top(value, target_z=TARGET_SIZE[2]),
            ),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )


def validate_ct(path: Path) -> None:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI image, found shape {image.shape}")
    if any(size <= 0 for size in image.shape):
        raise ValueError(f"Invalid image shape {image.shape}")
    if image.shape[2] < 8:
        raise ValueError(f"Too few axial slices: {image.shape[2]}")


def output_name(ct_path: Path) -> str:
    name = ct_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_CT"):
        name = name[:-3]
    return f"{name}_mask.nii.gz"


def align_mask(ct_path: Path, raw_mask_path: Path, output_path: Path) -> None:
    transformed = paired_transforms()(
        {"image": str(ct_path), "label": str(raw_mask_path)}
    )
    mask = np.rint(transformed["label"].squeeze(0).cpu().numpy()).astype(np.int16)
    affine = transformed["label"].meta.get("affine", np.eye(4))
    if isinstance(affine, torch.Tensor):
        affine = affine.cpu().numpy()
    nib.save(nib.Nifti1Image(mask, affine=np.asarray(affine)), str(output_path))


def process_one(task: tuple[str, str, str, bool, str]) -> Result:
    ct_string, output_string, temp_dir_string, overwrite, device = task
    ct_path = Path(ct_string)
    output_path = Path(output_string)
    if output_path.is_file() and output_path.stat().st_size > 100 and not overwrite:
        return Result(str(ct_path), str(output_path), "skipped")

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("nnUNet_n_proc_final_predict", "1")
    os.environ.setdefault("nnUNet_compile", "f")

    try:
        validate_ct(ct_path)
        from totalsegmentator.python_api import totalsegmentator

        temp_path = Path(temp_dir_string) / f"{os.getpid()}_{output_path.name}"
        totalsegmentator(
            str(ct_path),
            str(temp_path),
            task="oculomotor_muscles",
            ml=True,
            quiet=True,
            device=device,
        )
        align_mask(ct_path, temp_path, output_path)
        temp_path.unlink(missing_ok=True)
        if not output_path.is_file() or output_path.stat().st_size <= 100:
            raise RuntimeError("Aligned mask was not written correctly")
        return Result(str(ct_path), str(output_path), "success")
    except Exception as exc:
        return Result(str(ct_path), str(output_path), "failed", str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    installed_version = version("TotalSegmentator")
    if installed_version != EXPECTED_TOTALSEGMENTATOR_VERSION:
        raise RuntimeError(
            "Expected TotalSegmentator "
            f"{EXPECTED_TOTALSEGMENTATOR_VERSION}, found {installed_version}"
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    ct_paths = sorted(
        path
        for path in args.input_dir.rglob("*.nii.gz")
        if not path.name.startswith("._")
    )
    if not ct_paths:
        raise FileNotFoundError(f"No .nii.gz files found under {args.input_dir}")

    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp_dir:
        tasks = [
            (
                str(path),
                str(args.output_dir / output_name(path)),
                temp_dir,
                args.overwrite,
                args.device,
            )
            for path in ct_paths
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_one, tasks),
                    total=len(tasks),
                    desc="Oculomotor segmentation",
                )
            )

    with args.summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    counts = {
        status: sum(result.status == status for result in results)
        for status in ("success", "skipped", "failed")
    }
    print(
        f"Processed {len(results)} cases: "
        f"success={counts['success']}, skipped={counts['skipped']}, "
        f"failed={counts['failed']}"
    )
    print(f"Summary: {args.summary_csv}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
