#!/usr/bin/env python3
"""Fixed-atlas conditioning-control inference for TedFace-Flow.

Face controls:
  matched : use the paired DINO feature.
  null    : remove the face residual and keep only the CT modality embedding.
  shuffle : deterministically rotate DINO features across test subjects.
  flipped : load a precomputed horizontally flipped feature with the same basename.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import monai
import nibabel as nib
import torch
import torch.distributed as dist
from monai.data import MetaTensor, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import SaveImage
from monai.utils import set_determinism
from torch.cuda.amp import autocast
from tqdm import tqdm

from diff_model_setting import load_config
from utils import (
    binarize_labels,
    define_instance,
    dynamic_infer,
    prepare_maisi_controlnet_json_dataloader,
    setup_ddp,
)


def append_csv_row(path: str | os.PathLike, row: dict[str, object], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_nifti_label_tensor(path: str | os.PathLike, device: torch.device) -> tuple[torch.Tensor, tuple[int, ...]]:
    image = nib.load(str(path))
    labels = torch.from_numpy(image.get_fdata()).unsqueeze(0).unsqueeze(0).to(torch.uint8).to(device)
    return labels, tuple(labels.shape[2:])


def torch_load_feature(path: str | os.PathLike, device: torch.device) -> torch.Tensor:
    feature = torch.load(str(path), map_location=device, weights_only=True)
    if isinstance(feature, dict):
        for key in ("feature", "features", "embedding", "dino_feature"):
            if key in feature:
                feature = feature[key]
                break
    if not torch.is_tensor(feature):
        feature = torch.as_tensor(feature)
    return feature.squeeze().to(device=device, dtype=torch.float32)


class FaceResidualAdapter(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, out_dim)
        self.norm = torch.nn.LayerNorm(out_dim)
        self.silu = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.norm(x)
        x = self.silu(x)
        return self.linear2(x)


class EmbeddingBypass(torch.nn.Module):
    def __init__(self, original_embedding: torch.nn.Module):
        super().__init__()
        self.original_embedding = original_embedding
        self.embedding_dim = original_embedding.embedding_dim
        self.num_embeddings = original_embedding.num_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in [torch.long, torch.int]:
            return self.original_embedding(x)
        return x.clone()


class ReconModel(torch.nn.Module):
    def __init__(self, autoencoder: torch.nn.Module, scale_factor: torch.Tensor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def init_device() -> tuple[int, int, torch.device]:
    """Support both torchrun/DDP and single-process manual runs."""

    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        torch.cuda.set_device(device)
        return rank, world_size, device

    rank = 0
    world_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return rank, world_size, device


def barrier_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_dist_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> None:
    rank, world_size, device = init_device()

    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    logger = logging.getLogger("maisi.ted.fixed_atlas_conditioning_controls")

    set_determinism(seed=args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.env, args.config, args.net)

    face_path_map: dict[str, str] = {}
    if args.face_mode == "shuffle":
        with open(cfg.json_data_list, "r", encoding="utf-8") as f:
            test_json = json.load(f)
        if isinstance(test_json, dict):
            items = test_json.get("data", [])
        elif isinstance(test_json, list):
            items = test_json
        else:
            items = []
        feature_paths = sorted(str(item["dino_feature_path"]) for item in items if item.get("dino_feature_path"))
        if len(feature_paths) < 2:
            raise RuntimeError("Shuffled-face control requires at least two test DINO features.")
        shifted_paths = feature_paths[1:] + feature_paths[:1]
        face_path_map = dict(zip(feature_paths, shifted_paths))

    autoencoder = define_instance(cfg, "autoencoder_def").to(device)
    checkpoint_ae = torch.load(cfg.trained_autoencoder_path, map_location=device)
    ae_state = checkpoint_ae["unet_state_dict"] if isinstance(checkpoint_ae, dict) and "unet_state_dict" in checkpoint_ae else checkpoint_ae
    autoencoder.load_state_dict(ae_state, strict=False)

    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    checkpoint_unet = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint_unet["unet_state_dict"] if "unet_state_dict" in checkpoint_unet else checkpoint_unet, strict=False)
    scale_factor = checkpoint_unet["scale_factor"].to(device)

    face_adapter = FaceResidualAdapter(3072, unet.class_embedding.embedding_dim).to(device)
    unet.class_embedding = EmbeddingBypass(unet.class_embedding)

    controlnet = define_instance(cfg, "controlnet_def").to(device)
    monai.networks.utils.copy_model_state(controlnet, unet.state_dict())
    controlnet.class_embedding = EmbeddingBypass(controlnet.class_embedding)

    checkpoint_ted = torch.load(cfg.trained_controlnet_path, map_location=device, weights_only=True)
    controlnet.load_state_dict(checkpoint_ted["controlnet_state_dict"], strict=False)
    face_adapter.load_state_dict(checkpoint_ted["face_adapter_state_dict"])

    noise_scheduler = define_instance(cfg, "noise_scheduler")
    with open(cfg.modality_mapping_path, "r", encoding="utf-8") as f:
        modality_mapping = json.load(f)

    autoencoder.eval()
    controlnet.eval()
    unet.eval()
    face_adapter.eval()

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        logger.info("Using fixed atlas mask: %s", args.mask)
    barrier_if_needed()

    _, test_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=cfg.json_data_list,
        data_base_dir=cfg.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=1,
        fold=-1,
        modality_mapping=modality_mapping,
        json_key="data",
    )

    log_path = args.selection_log
    if world_size > 1:
        root, ext = os.path.splitext(log_path)
        log_path = f"{root}_rank{rank}{ext or '.csv'}"

    log_fields = [
        "rank",
        "test_id",
        "test_dino_path",
        "face_dino_path",
        "face_mode",
        "guidance_mode",
        "atlas_mask_path",
        "output_size",
    ]

    if not args.mask:
        raise ValueError("--mask is required for fixed-atlas inference.")
    fixed_label_tensor, _ = load_nifti_label_tensor(args.mask, device=device)

    processed = 0
    for batch in test_loader:
        if args.max_cases is not None and processed >= args.max_cases:
            break
        processed += 1

        batch_data = decollate_batch(batch)[0]
        subject_id = batch_data.get(
            "anon_id",
            os.path.basename(batch_data["label"].meta["filename_or_obj"]).split(".")[0].replace("_mask", ""),
        )
        spacing_tensor = batch["spacing"].to(device).float()

        dino_path = str(batch["dino_feature_path"][0])
        face_dino_path = dino_path
        if args.face_mode == "shuffle":
            if dino_path not in face_path_map:
                raise KeyError(f"Cannot find test DINO feature in shuffle map: {dino_path}")
            face_dino_path = face_path_map[dino_path]
        elif args.face_mode == "flipped":
            if not args.alternate_feature_dir:
                raise ValueError("--alternate-feature-dir is required for --face-mode flipped.")
            face_dino_path = str(Path(args.alternate_feature_dir) / Path(dino_path).name)
            if not Path(face_dino_path).is_file():
                raise FileNotFoundError(face_dino_path)

        raw_face_feat = torch_load_feature(face_dino_path, device=device)
        if raw_face_feat.ndim == 1:
            raw_face_feat = raw_face_feat.unsqueeze(0)

        label_tensor = fixed_label_tensor

        output_size = tuple(label_tensor.shape[2:])
        latent_shape = (
            cfg.latent_channels,
            output_size[0] // 4,
            output_size[1] // 4,
            output_size[2] // 4,
        )

        append_csv_row(
            log_path,
            {
                "rank": rank,
                "test_id": subject_id,
                "test_dino_path": dino_path,
                "face_dino_path": face_dino_path,
                "face_mode": args.face_mode,
                "guidance_mode": args.guidance_mode,
                "atlas_mask_path": args.mask,
                "output_size": "x".join(str(v) for v in output_size),
            },
            log_fields,
        )

        with torch.no_grad(), autocast(enabled=args.amp):
            projected_face_feat = face_adapter(raw_face_feat)
            modality_id = torch.tensor([cfg.controlnet_infer["modality"]], device=device)
            base_ct_emb = unet.class_embedding.original_embedding(modality_id)
            fused_embedding = base_ct_emb if args.face_mode == "null" else projected_face_feat + base_ct_emb

            controlnet_cond_vis = binarize_labels(label_tensor.long()).half()
            latents = torch.randn([1] + list(latent_shape)).half().to(device)

            if isinstance(noise_scheduler, RFlowScheduler):
                noise_scheduler.set_timesteps(
                    num_inference_steps=cfg.controlnet_infer["num_inference_steps"],
                    input_img_size_numel=torch.prod(torch.tensor(latents.shape[-3:])),
                )
            else:
                noise_scheduler.set_timesteps(num_inference_steps=cfg.controlnet_infer["num_inference_steps"])

            all_timesteps = noise_scheduler.timesteps
            all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
            progress_bar = tqdm(
                zip(all_timesteps, all_next_timesteps),
                total=len(all_timesteps),
                desc=f"fixed-atlas {subject_id}",
                disable=rank != 0,
            )

            for t, next_t in progress_bar:
                latent_model_input = torch.cat([latents] * 2)
                empty_mask = torch.zeros_like(controlnet_cond_vis)
                controlnet_cond_input = torch.cat([controlnet_cond_vis, empty_mask])
                if args.guidance_mode == "standard":
                    class_labels_input = torch.cat([fused_embedding, base_ct_emb])
                else:
                    class_labels_input = torch.cat([fused_embedding, fused_embedding])

                down_res, mid_res = controlnet(
                    x=latent_model_input,
                    timesteps=torch.Tensor((t,)).to(device).repeat(2),
                    controlnet_cond=controlnet_cond_input,
                    class_labels=class_labels_input,
                )

                unet_out = unet(
                    x=latent_model_input,
                    timesteps=torch.Tensor((t,)).to(device).repeat(2),
                    spacing_tensor=spacing_tensor.repeat(2, 1),
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    class_labels=class_labels_input,
                )

                model_cond, model_uncond = unet_out.chunk(2)
                noise_pred = model_uncond + args.cfg_scale * (model_cond - model_uncond)

                if isinstance(noise_scheduler, RFlowScheduler):
                    latents, _ = noise_scheduler.step(noise_pred, t, latents, next_t)
                else:
                    latents, _ = noise_scheduler.step(noise_pred, t, latents)

            recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
            inferer = SlidingWindowInferer(
                roi_size=cfg.controlnet_infer["autoencoder_sliding_window_infer_size"],
                sw_batch_size=1,
                progress=False,
                mode="gaussian",
                overlap=cfg.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
                sw_device=device,
                device=torch.device("cpu"),
            )
            synthetic_images = dynamic_infer(inferer, recon_model, latents)
            synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
            synthetic_images = synthetic_images * 600 - 300

        torch.cuda.empty_cache()

        meta_dict = batch_data["label"].meta.copy()
        meta_dict["filename_or_obj"] = f"{subject_id}.nii.gz"
        output_postfix = (
            f"fixed_{args.face_mode}_{args.guidance_mode}_"
            f"cfg{args.cfg_scale:g}_gen"
        )
        synthetic_images_final = MetaTensor(synthetic_images.squeeze(0), meta=meta_dict)
        SaveImage(output_dir=args.out, output_postfix=output_postfix, separate_folder=False)(synthetic_images_final)

        if rank == 0:
            print(
                f"[Done] {subject_id} | mode=fixed_atlas | "
                f"face={args.face_mode} | guidance={args.guidance_mode}"
            )

    destroy_dist_if_needed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-atlas conditioning-control inference for TedFace-Flow.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask", required=True, help="Fixed atlas mask path.")
    parser.add_argument("--selection-log", default="template_selection.csv")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--face-mode", choices=["matched", "null", "shuffle", "flipped"], default="matched")
    parser.add_argument("--guidance-mode", choices=["decoupled", "standard"], default="decoupled")
    parser.add_argument("--alternate-feature-dir", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
