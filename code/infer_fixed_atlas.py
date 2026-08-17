# Copyright (c) 2026 TED-Project 
import argparse
import json
import logging
import os
import torch
import torch.distributed as dist
import monai
import nibabel as nib
from monai.data import MetaTensor, decollate_batch
from monai.transforms import SaveImage
from monai.utils import RankFilter, set_determinism
from torch.cuda.amp import autocast
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer

from diff_model_setting import load_config
from utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp, binarize_labels, dynamic_infer
from monai.networks.schedulers import RFlowScheduler

# --- [对齐核心：定义类结构确保权重匹配] ---
class FaceResidualAdapter(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, out_dim)
        self.norm = torch.nn.LayerNorm(out_dim)
        self.silu = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(out_dim, out_dim)
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.silu(x)
        return self.linear2(x)

class EmbeddingBypass(torch.nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.original_embedding = original_embedding
        self.embedding_dim = original_embedding.embedding_dim
        self.num_embeddings = original_embedding.num_embeddings
    def forward(self, x):
        if x.dtype in [torch.long, torch.int]: return self.original_embedding(x)
        return x.clone()

class ReconModel(torch.nn.Module):
    """用于 VAE 最终解码还原 CT 图像的包裹类"""
    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor
    def forward(self, z):
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda

@torch.inference_mode()
def run_template_testset_inference(
    env_config,
    model_config,
    model_def,
    template_mask_path,
    base_output_dir,
    seed=2026,
    cfg_scale=3.0,
):
    # DDP 初始化
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = setup_ddp(rank, world_size)
    torch.cuda.set_device(device)
    
    logger = logging.getLogger("maisi.ted.template_infer")
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
        logger.info(f">>> 启动标准模板+测试集特征推理 | 终极解耦 CFG 模式 | 显卡数: {world_size}")

    set_determinism(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if rank == 0:
        logger.info(f">>> inference seed fixed to {seed} <<<")

    cfg = load_config(env_config, model_config, model_def)

    # 模型加载 
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
    with open(cfg.modality_mapping_path) as f: modality_mapping = json.load(f)
    autoencoder.eval(); controlnet.eval(); unet.eval(); face_adapter.eval()

    # 加载固定的标准模板 Mask
    tm_img = nib.load(template_mask_path)
    standard_labels = MetaTensor(
        torch.from_numpy(tm_img.get_fdata()).unsqueeze(0).unsqueeze(0),
        meta={'filename_or_obj': template_mask_path} 
    ).to(device).to(torch.uint8)

    # 加载测试集特征 (fold=-1)
    _, test_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=cfg.json_data_list, data_base_dir=cfg.data_base_dir,
        rank=rank, world_size=world_size, batch_size=1, fold=-1, 
        modality_mapping=modality_mapping, json_key="data"
    )

    if rank == 0: os.makedirs(base_output_dir, exist_ok=True)
    dist.barrier()

    cfg_val = cfg_scale

    for batch in test_loader:
        batch_data = decollate_batch(batch)[0]
        subject_id = batch_data.get("anon_id", os.path.basename(batch_data["label"].meta["filename_or_obj"]).split('.')[0].replace("_mask", ""))
        spacing_tensor = batch["spacing"].to(device).float()
        
        # 尺寸对齐
        output_size = standard_labels.shape[2:]
        latent_shape = (cfg.latent_channels, output_size[0]//4, output_size[1]//4, output_size[2]//4)

        # 提取受试者本人的 DINO 面部特征
        dino_path = batch["dino_feature_path"][0]
        raw_face_feat = torch.load(dino_path, map_location=device, weights_only=True).squeeze().to(torch.float32)
        if raw_face_feat.ndim == 1: raw_face_feat = raw_face_feat.unsqueeze(0)

        # 开启 AMP 与显存回收
        with torch.no_grad(), autocast(enabled=True):
            
            # 1. 分离出 Base CT 特征 和 融合后的特征
            projected_face_feat = face_adapter(raw_face_feat)
            modality_id = torch.tensor([cfg.controlnet_infer["modality"]], device=device)
            base_ct_emb = unet.class_embedding.original_embedding(modality_id)
            fused_embedding = projected_face_feat + base_ct_emb

            # 2. 掩码二值化处理
            controlnet_cond_vis = binarize_labels(standard_labels.as_tensor().long()).half()

            # 3. 生成完全固定的初始噪声 (基于 Seed=42)
            latents = torch.randn([1] + list(latent_shape)).half().to(device)

            # 4. 配置调度器
            if isinstance(noise_scheduler, RFlowScheduler):
                noise_scheduler.set_timesteps(num_inference_steps=cfg.controlnet_infer["num_inference_steps"], input_img_size_numel=torch.prod(torch.tensor(latents.shape[-3:])))
            else:
                noise_scheduler.set_timesteps(num_inference_steps=cfg.controlnet_infer["num_inference_steps"])

            all_timesteps = noise_scheduler.timesteps
            all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
            progress_bar = tqdm(zip(all_timesteps, all_next_timesteps), total=len(all_timesteps), desc=f"Infer {subject_id}")

            # 5. 去噪循环
            for t, next_t in progress_bar:
                # 复制潜变量以构建 Cond 和 Uncond 双支路
                latent_model_input = torch.cat([latents] * 2)
                
                # =======================================================
                # 终极正确 CFG：只对 Mask 进行抽真空（因为训练时只 Dropout 了 Mask）
                # =======================================================
                # Cond: 标准模板 Mask
                # Uncond: 全 0 Mask (触发清晰度锐化)
                empty_mask = torch.zeros_like(controlnet_cond_vis)
                controlnet_cond_input = torch.cat([controlnet_cond_vis, empty_mask])
                
                # =======================================================
                # 两个支路都必须携带 DINO 特征！
                # 绝对不要用 base_ct_emb！
                # =======================================================
                class_labels_input = torch.cat([fused_embedding, fused_embedding])

                # ControlNet 前向
                down_res, mid_res = controlnet(
                    x=latent_model_input,
                    timesteps=torch.Tensor((t,)).to(device).repeat(2),
                    controlnet_cond=controlnet_cond_input,
                    class_labels=class_labels_input
                )

                # UNet 前向
                unet_out = unet(
                    x=latent_model_input,
                    timesteps=torch.Tensor((t,)).to(device).repeat(2),
                    spacing_tensor=spacing_tensor.repeat(2, 1),
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    class_labels=class_labels_input
                )

                # 计算 CFG
                model_cond, model_uncond = unet_out.chunk(2)
                
                # 此时放大的差值 = 仅有 Mask 带来的清晰度锐化
                # DINO 信号贯穿始终，提供绝对的病理指导！不会触发潜空间爆炸
                noise_pred = model_uncond + cfg_val * (model_cond - model_uncond)

                # Scheduler Step
                if isinstance(noise_scheduler, RFlowScheduler):
                    latents, _ = noise_scheduler.step(noise_pred, t, latents, next_t)
                else:
                    latents, _ = noise_scheduler.step(noise_pred, t, latents)

            # 6. VAE 空间还原
            recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
            inferer = SlidingWindowInferer(
                roi_size=cfg.controlnet_infer["autoencoder_sliding_window_infer_size"],
                sw_batch_size=1, progress=False, mode="gaussian",
                overlap=cfg.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
                sw_device=device, device=torch.device("cpu")
            )
            synthetic_images = dynamic_infer(inferer, recon_model, latents)

            # 7. 后处理还原 HU 范围[-300, 300]
            synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
            synthetic_images = synthetic_images * 600 - 300
        
        torch.cuda.empty_cache()

        # 覆写 meta 信息保存
        meta_dict = batch_data["label"].meta.copy()
        if "anon_id" in batch_data:
            meta_dict["filename_or_obj"] = f"{batch_data['anon_id']}.nii.gz"
            
        synthetic_images_final = MetaTensor(synthetic_images.squeeze(0), meta=meta_dict)
        SaveImage(output_dir=base_output_dir, output_postfix=f"cfg{cfg_val}_gen", separate_folder=False)(synthetic_images_final)
        
        if rank == 0: print(f"   - [Rank 0] 受试者 {subject_id} 生成完成 (Mask: 模板, CFG: {cfg_val})")

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    pa = parser.parse_args()
    
    run_template_testset_inference(pa.env, pa.config, pa.net, pa.mask, pa.out, pa.seed, pa.cfg_scale)
