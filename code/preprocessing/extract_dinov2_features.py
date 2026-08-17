#!/usr/bin/env python3
"""Extract the 3072-D DINOv2 descriptors consumed by TedFace-Flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IMAGE_SIZE = (224, 448)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval().to(device)
    transform = v2.Compose(
        [
            v2.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    for image_path in tqdm(image_paths, desc="DINOv2 ViT-L/14"):
        relative_path = image_path.relative_to(args.input_dir).with_suffix(".pt")
        output_path = args.output_dir / relative_path
        if output_path.is_file() and not args.overwrite:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            tokens = model.forward_features(tensor)
            class_token = tokens["x_norm_clstoken"]
            patch_tokens = tokens["x_norm_patchtokens"]
            descriptor = torch.cat(
                [
                    class_token,
                    patch_tokens.mean(dim=1),
                    patch_tokens.amax(dim=1),
                ],
                dim=1,
            ).squeeze(0)
        if descriptor.shape != (3072,):
            raise RuntimeError(
                f"Expected a 3072-D descriptor, found {tuple(descriptor.shape)}"
            )
        torch.save(descriptor.cpu(), output_path)

    print(f"[Done] Features saved under {args.output_dir}")


if __name__ == "__main__":
    main()
