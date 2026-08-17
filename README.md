# TedFace-Flow

TedFace-Flow is a face-conditioned three-dimensional orbital morphology proxy
generation framework for thyroid eye disease (TED). It combines a frozen DINOv2
facial encoder, a MAISI-v2 rectified-flow backbone, anatomical ControlNet
conditioning, a Face Residual Adapter, label-priority-preserving mask perturbation,
progressive mask dropout, RSCL-face, and decoupled classifier-free guidance.

The generated volume is an atlas-normalized research proxy. It is not a reconstruction
of an individual patient's CT and is not intended for diagnosis or clinical decision
making.

## Repository layout

```text
assets/                       Fixed orbital atlas and metadata
code/                         Cross-modal training and inference
code/preprocessing/           DINOv2 extraction and orbital segmentation
code/analysis/                Morphometry and statistical analysis
configs/                      Network, phase-specific, and environment files
examples/                     De-identified manifest schemas
models/                       Model-weight placement instructions
```

## Installation

Python 3.10 and a CUDA-enabled PyTorch environment are recommended.

```bash
pip install -r requirements.txt
```

The orbital-segmentation pipeline uses TotalSegmentator 2.15.0 and its distributed
`oculomotor_muscles` weights. The MAISI-v2 autoencoder and rectified-flow backbone
weights must be obtained separately under their original distribution terms.

## Model weights

The final inference checkpoint used for the reported analyses is available in the
[TedFace-Flow v1.0.0 release](https://github.com/TedFace-Flow/TedFace-Flow-files/releases/tag/v1.0.0).

- [Download `tedface_flow_final.pt`](https://github.com/TedFace-Flow/TedFace-Flow-files/releases/download/v1.0.0/tedface_flow_final.pt)
- SHA-256: `153768cb0cb837e8ddccf17e54a6cbe5b1f02c3cb4dff50c0a2c4ee21c10fce7`

After downloading, place the checkpoint at:

```text
models/tedface_flow_final.pt

## Facial descriptors

The model consumes a 3072-dimensional DINOv2 ViT-L/14 descriptor formed by
concatenating the normalized class token, mean-pooled patch tokens, and max-pooled
patch tokens. Input images to the released extractor must already follow the study's
privacy-preserving periocular preprocessing: frontal primary gaze, iris alignment,
nasal-bridge/eyebrow crop, and ocular occlusion band.

```bash
python code/preprocessing/extract_dinov2_features.py \
  --input-dir data/preprocessed_faces \
  --output-dir data/dinov2_features \
  --device cuda
```

## Training schedule

The manuscript uses two high-level development stages:

1. Orbital-domain adaptation initializes the generator from CT-mask pairs.
2. Cross-modal face-conditioning introduces the matched periocular descriptor.

The initial four-phase schedule comprised 120 cross-modal training epochs.
It was followed by an additional 60-epoch refinement stage. The released
checkpoint is the endpoint of this refinement stage.

| Cross-modal phase | Epochs | Initial LR | Mask perturbation | Mask dropout | RSCL-face weight |
|---|---:|---:|---:|---:|---:|
| 1 | 30 | 1e-5 | 0.50 | 0.15 | 0.010 |
| 2 | 30 | 5e-6 | 0.50 | 0.20 | 0.001 |
| 3 | 30 | 2e-6 | 0.50 | 0.30 | 0.001 |
| 4 | 30 | 1e-6 | 0.50 | 0.50 | 0.001 |

RSCL-face is always enabled for the complete model. Its phase-specific weight is
defined in `code/train_tedface_flow.py`, not in the JSON files. Each phase loads the
preceding phase's `*_current.pt` checkpoint. Phase 1 loads the orbital-domain
checkpoint described in `models/README.md`.

```bash
torchrun --nproc_per_node=8 code/train_tedface_flow.py \
  -e configs/environment_train_phase1.example.json \
  -c configs/config_crossmodal_phase1.json \
  -t configs/config_network_rflow.json -g 8 --crossmodal-phase 1

torchrun --nproc_per_node=8 code/train_tedface_flow.py \
  -e configs/environment_train_phase2.example.json \
  -c configs/config_crossmodal_phase2.json \
  -t configs/config_network_rflow.json -g 8 --crossmodal-phase 2

torchrun --nproc_per_node=8 code/train_tedface_flow.py \
  -e configs/environment_train_phase3.example.json \
  -c configs/config_crossmodal_phase3.json \
  -t configs/config_network_rflow.json -g 8 --crossmodal-phase 3

torchrun --nproc_per_node=8 code/train_tedface_flow.py \
  -e configs/environment_train_phase4.example.json \
  -c configs/config_crossmodal_phase4.json \
  -t configs/config_network_rflow.json -g 8 --crossmodal-phase 4
```

## Fixed-atlas inference

Edit `configs/environment_inference.example.json` and provide a manifest following
`examples/inference_manifest.example.json`. Run commands from the repository root.

```bash
torchrun --nproc_per_node=1 code/infer_fixed_atlas.py \
  --env configs/environment_inference.example.json \
  --config configs/config_crossmodal_phase4.json \
  --net configs/config_network_rflow.json \
  --mask assets/fixed_orbital_atlas.nii.gz \
  --out outputs/fixed_atlas \
  --seed 2026 \
  --cfg-scale 3
```

The reported repeated matched-face analysis used 60 rectified-flow sampling steps and
seeds 2026, 42, and 1024. Run inference once for each seed.

## Orbital segmentation

```bash
python code/preprocessing/segment_oculomotor_muscles.py \
  --input-dir outputs/fixed_atlas \
  --output-dir outputs/fixed_atlas_masks \
  --summary-csv outputs/fixed_atlas_segmentation_summary.csv \
  --workers 1 \
  --device gpu
```

Increase `--workers` only after checking available GPU memory. The script exits with a
nonzero status if any case fails and records every case in the summary CSV. The
quantitative analysis workflow is documented in `code/analysis/README.md`.

## Data availability

Patient-level CT images, photographs, masks, derived facial features, and clinical
records are not included because of patient privacy, ethical restrictions, and
institutional data-governance requirements. Example manifests contain synthetic
identifiers and paths only. The released fixed atlas is a labelled, intensity-free,
de-identified derivative; its construction is documented in
`assets/fixed_orbital_atlas.json`.

## License and citation

Code is released under the Apache License 2.0. Third-party checkpoints and software
remain subject to their original licenses. Citation information will be updated after
publication.
