# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
import contextlib
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.utils import copy_model_state
from monai.transforms.utils_morphological_ops import dilate
from monai.utils import RankFilter
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.tensorboard import SummaryWriter

from augmentation import remove_tumors
from diff_model_setting import load_config
from utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp


# RSCL-face is part of the complete TedFace-Flow objective. Its phase-specific
# weight is controlled here, independently of the JSON configuration.
RSCL_FACE_MAX_RESPONSE = 2.0
RSCL_FACE_WEIGHT_BY_PHASE = {
    1: 0.010,
    2: 0.001,
    3: 0.001,
    4: 0.001,
}


def _binary_dilate(mask, iteration):
    if iteration <= 0:
        return mask
    k_size = 1 + 2 * iteration
    return F.max_pool3d(mask.float(), kernel_size=k_size, stride=1, padding=iteration) > 0


def _binary_erode(mask, iteration):
    if iteration <= 0:
        return mask
    k_size = 1 + 2 * iteration
    inv = 1.0 - mask.float()
    eroded_inv = F.max_pool3d(inv, kernel_size=k_size, stride=1, padding=iteration)
    return eroded_inv <= 0


def perturb_labels(labels, roi_ids, prob=0.5):
    """
    Label-wise mask perturbation used during cross-modal training.

    Dilation writes only into original background voxels, so existing anatomical
    labels are protected. If two labels compete for the same newly dilated
    background voxel, the lower sorted ROI id gets priority deterministically.
    """
    if torch.rand(1).item() > prob:
        return labels 

    iteration = torch.randint(1, 4, (1,)).item()
    mode = "dilate" if torch.rand(1).item() > 0.5 else "erode"
    roi_ids = sorted({int(rid) for rid in roi_ids})
    
    if mode == "dilate":
        labels_perturbed = labels.clone()
        original_foreground = labels != 0
        claimed = torch.zeros_like(labels, dtype=torch.bool)

        for rid in roi_ids:
            label_mask = labels == rid
            if not bool(label_mask.any()):
                continue
            dilated = _binary_dilate(label_mask, iteration)
            write_mask = dilated & (~original_foreground) & (~claimed)
            labels_perturbed[write_mask] = rid
            claimed |= write_mask
    else: 
        labels_perturbed = labels.clone()
        for rid in roi_ids:
            label_mask = labels == rid
            if not bool(label_mask.any()):
                continue
            eroded = _binary_erode(label_mask, iteration)
            labels_perturbed[label_mask & (~eroded)] = 0
    
    return labels_perturbed

class EmbeddingBypass(torch.nn.Module):
    """
    TED-Project 专用插件：
    允许模型同时接受 类别索引(Long) 和 预计算特征(Float)。
    """
    def __init__(self, original_embedding):
        super().__init__()
        self.original_embedding = original_embedding
        # 继承原始属性，防止模型其它部分访问时报错
        self.embedding_dim = original_embedding.embedding_dim
        self.num_embeddings = original_embedding.num_embeddings

    def forward(self, x):
        # 如果输入是整数 ID，执行原始查表逻辑
        if x.dtype in [torch.long, torch.int]:
            return self.original_embedding(x)
        # 如果输入已经是 FloatTensor（我们的融合向量），直接返回
        return x.clone()

def remove_roi(labels):
    """
    TED 项目专用：在计算 RSCL 扰动掩码时，将眼肌和视神经标签抹除。
    """
    # 你的 ROI 标签列表：包括左右所有肌肉和视神经
    roi_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    # 构造扰动掩码：将 ROI 替换为背景标签 0（代表脂肪）
    labels_roi_free = labels.clone()
    for rid in roi_ids:
        labels_roi_free[labels == rid] = 0
        
    return labels_roi_free


def compute_region_contrastive_loss_ted(
    model_output_active,    # 支路 A: 有面部特征 (DINO + Modality)
    model_output_null,      # 支路 B: 无面部特征 (Null + Modality)
    roi_mask,               # 真实对齐的 ROI Mask
    roi_bg_mask,            # 真实对齐的背景 Mask
    max_region_contrastive_loss=RSCL_FACE_MAX_RESPONSE,
):
    """
    基于 MAISI-v2 的 RSCL 思想 [cite: 148]，强制 DINO 信号仅在眼眶区域生效。
    """
    # 空间对齐：将 Mask 缩放到 Latent 空间尺寸
    m = F.interpolate(roi_mask.float(), size=model_output_active.shape[2:], mode="nearest")
    m_bg = F.interpolate(roi_bg_mask.float(), size=model_output_active.shape[2:], mode="nearest")
    
    # 1. ROI 敏感性：计算有无 DINO 导致的生成差异 [cite: 146]
    # 我们希望 DINO 信号在眼肌区域引起显著变化
    diff = torch.abs(model_output_active - model_output_null)
    loss_roi = - (diff * m).sum() / (m.sum() + 1e-5)
    # 使用 delta 截断防止梯度爆炸 [cite: 160, 165]
    loss_roi = F.relu(loss_roi + max_region_contrastive_loss) - max_region_contrastive_loss

    # 2. 背景一致性：确保 DINO 不干扰眼眶外区域 [cite: 147]
    # 强制 DINO 信号在背景区域的贡献为 0
    loss_bg = (diff * m_bg).sum() / (m_bg.sum() + 1e-5)
    
    return loss_roi, loss_bg


def compute_model_output(
    images,
    labels,
    noise,
    timesteps,
    noise_scheduler,
    controlnet,
    unet,
    spacing_tensor,
    face_features=None,
    modality_tensor=None,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    return_controlnet_blocks=False,
):
    """
    TED-Project 深度定制版：支持面部特征与模态 Embedding 的向量级融合。
    """
    include_body_region = (top_region_index_tensor is not None) and (bottom_region_index_tensor is not None)

    # 1. 构造 ControlNet 掩码条件
    # 将多标签 Mask 转为二值位图编码
    controlnet_cond = binarize_labels(labels.to(torch.long)).float()

    # 2. 生成带噪 Latent (Rectified Flow 直线加噪)
    # RFlow 逻辑：x_t = (1-t)*images + t*noise 
    noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

    # 3. 深度特征融合 (核心修改点)
    # 我们需要将 3072D 投影后的向量与模型自带的 Modality Embedding 相加
    if face_features is not None:
        unet_model = unet.module if hasattr(unet, "module") else unet
        # 获取模态向量（如果存在）
        m_emb = unet_model.class_embedding(modality_tensor) if modality_tensor is not None else 0
        # 支路 A: 执行残差注入
        final_class_labels = face_features + m_emb
    else:
        # 支路 B: 直接返回 LongTensor ID，触发 EmbeddingBypass 原始查表
        final_class_labels = modality_tensor

    # 4. 前向传播 ControlNet
    # 注入融合后的 final_class_labels，让 ControlNet 感知面部病理特征
    controlnet_inputs = {
        "x": noisy_latent,
        "timesteps": timesteps,
        "controlnet_cond": controlnet_cond,
        "class_labels": final_class_labels
    }
    down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

    # 5. 前向传播 UNet
    # 预测速度场 (Velocity Field) [cite: 72, 75]
    unet_inputs = {
        "x": noisy_latent,
        "timesteps": timesteps,
        "spacing_tensor": spacing_tensor,
        "down_block_additional_residuals": down_block_res_samples,
        "mid_block_additional_residual": mid_block_res_sample,
        "class_labels": final_class_labels
    }
    
    if include_body_region:
        unet_inputs.update({
            "top_region_index_tensor": top_region_index_tensor,
            "bottom_region_index_tensor": bottom_region_index_tensor,
        })

    model_output = unet(**unet_inputs)
    
    if return_controlnet_blocks:
        return model_output, down_block_res_samples, mid_block_res_sample
    else:
        return model_output, None, None


def train_controlnet(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_gpus: int,
    crossmodal_phase: int,
) -> None:
    # 开启异常检测模式
    #torch.autograd.set_detect_anomaly(True)
    # Step 0: configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()] # 确保输出到终端
    )
    logger = logging.getLogger("maisi.controlnet.training")
    logger.setLevel(logging.INFO) # 显式设置级别
    # whether to use distributed data parallel
    use_ddp = num_gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    args = load_config(env_config_path, model_config_path, model_def_path)
    if crossmodal_phase not in RSCL_FACE_WEIGHT_BY_PHASE:
        raise ValueError("--crossmodal-phase must be one of 1, 2, 3, or 4")
    rscl_face_weight = RSCL_FACE_WEIGHT_BY_PHASE[crossmodal_phase]
    logger.info(
        "Cross-modal phase %d: RSCL-face enabled in code "
        "(max_response=%.3f, weight=%.6f)",
        crossmodal_phase,
        RSCL_FACE_MAX_RESPONSE,
        rscl_face_weight,
    )

    # initialize tensorboard writer
    if rank == 0:
        tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        
        Path(args.model_dir).mkdir(parents=True, exist_ok=True) 
        logger.info(f"Directory for checkpoints ensured at: {args.model_dir}")

    # --- TED-Project: 模型定义与适配器初始化 ---
    # 定义基础 UNet
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # TED 项目建议：使用残差结构的 Adapter，使 DINO 信号更容易作为增量注入
    class FaceResidualAdapter(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear1 = torch.nn.Linear(in_dim, out_dim)
            self.norm = torch.nn.LayerNorm(out_dim)
            self.silu = torch.nn.SiLU()
            self.linear2 = torch.nn.Linear(out_dim, out_dim)
            
        def forward(self, x):
            # 将 3072D 投影到目标维度
            x = self.linear1(x)
            x = self.norm(x)
            x = self.silu(x)
            # 这里的输出将作为对 modality_embedding 的残差修正
            return self.linear2(x)

    # 1. 初始化 Face Adapter
    face_feature_dim = 3072
    embed_dim = unet.class_embedding.embedding_dim if hasattr(unet, 'class_embedding') else (args.diffusion_unet_def["num_channels"][0] * 4)
    face_adapter = FaceResidualAdapter(face_feature_dim, embed_dim).to(device)

    # 2. 加载基座 UNet 权重 (此时未打补丁，Key 匹配原始 MAISI 权重)
    if args.trained_diffusion_path is not None:
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"], strict=False)
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Loaded base unet. scale_factor: {scale_factor}")
    else:
        raise ValueError("'trained_diffusion_path' is required.")

    # 3. 初始化 ControlNet 并从 UNet 同步原始权重
    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())

    # 4. 【核心修复】为两者注入插件，使其结构与第一阶段 Checkpoint 对齐
    unet.class_embedding = EmbeddingBypass(unet.class_embedding)
    controlnet.class_embedding = EmbeddingBypass(controlnet.class_embedding)
    logger.info("Successfully patched UNet and ControlNet with EmbeddingBypass.")

    # 5. 设置训练属性
    if hasattr(controlnet, "use_checkpointing"):
        controlnet.use_checkpointing = True
    if hasattr(unet, "use_checkpointing"):
        unet.use_checkpointing = True
    logger.info("Enabled Activation Checkpointing.")
    
    # 6. Load the orbital-domain or preceding cross-modal checkpoint.
    if args.existing_ckpt_filepath is not None:
        checkpoint = torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=True)
        # 加载微调后的 ControlNet
        controlnet.load_state_dict(checkpoint["controlnet_state_dict"])
        # 加载微调后的 Face Adapter
        if "face_adapter_state_dict" in checkpoint:
            face_adapter.load_state_dict(checkpoint["face_adapter_state_dict"])
            logger.info("Loaded face_adapter weights from initialization checkpoint.")
        logger.info(f"Loaded initialization checkpoint from {args.existing_ckpt_filepath}")
    
    
    # 冻结 UNet
    for p in unet.parameters():
        p.requires_grad = False
    unet.eval()
    
    # --- TED-Project: 强制禁用原地操作，防止 RSCL 梯度冲突 ---
    for name, m in unet.named_modules():
        if hasattr(m, 'inplace'):
            m.inplace = False
    for name, m in controlnet.named_modules():
        if hasattr(m, 'inplace'):
            m.inplace = False
    logger.info("Strictly disabled all recursive inplace operations.")
    
    
    
    # 初始化调度器 (MAISI-v2 使用 Rectified Flow)
    noise_scheduler = define_instance(args, "noise_scheduler")

    # 分布式并行处理
    if use_ddp:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)
        # 增加 find_unused_parameters=True，防止 CFG 丢弃时报梯度不同步
        face_adapter = DDP(face_adapter, device_ids=[device], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        params=list(controlnet.parameters()) + list(face_adapter.parameters()), 
        lr=args.controlnet_train["lr"]
    )
    
    # set data loader
    if include_modality:
        if args.modality_mapping_path is not None:
            if not os.path.exists(args.modality_mapping_path):
                raise ValueError(f"Please check if {args.modality_mapping_path} exist.")
        else:
            raise ValueError(f"'modality_mapping_path' in {env_config_path} cannot be null")
        with open(args.modality_mapping_path) as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    train_loader, _ = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
        modality_mapping=args.modality_mapping,
        drop_last=True # 强制丢弃最后那个分不匀的 Batch
    )

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    mask_perturb_prob = float(args.controlnet_train["mask_perturb_prob"])
    mask_dropout_prob = float(args.controlnet_train["mask_dropout_prob"])
    for name, value in {
        "mask_perturb_prob": mask_perturb_prob,
        "mask_dropout_prob": mask_dropout_prob,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if int(args.controlnet_train["n_epochs"]) != 30:
        raise ValueError(
            "Each reported cross-modal phase contains exactly 30 epochs; "
            f"received n_epochs={args.controlnet_train['n_epochs']}"
        )
    total_steps = args.controlnet_train["n_epochs"] * len(train_loader)
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler("cuda")
    total_step = 0
    best_loss = 1e4

    if weighted_loss > 1.0:
        logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(n_epochs):
        logger.info(
            "Phase %d, epoch %d mask probabilities: perturb=%.4f, dropout=%.4f",
            crossmodal_phase,
            epoch + 1,
            mask_perturb_prob,
            mask_dropout_prob,
        )
        epoch_loss_ = 0
        epoch_l_roi_ = 0
        epoch_l_bg_ = 0
        controlnet.train()
        face_adapter.train() # 确保适配器在训练模式
        
        with (controlnet.join() if use_ddp else contextlib.nullcontext()), \
             (face_adapter.join() if use_ddp else contextlib.nullcontext()):
        
            for step, batch in enumerate(train_loader):
                # 1. 加载图像与标签
                images = batch["image"].as_tensor().to(device) * scale_factor
                labels_gt = batch["label"].as_tensor().to(device)
                # 动态获取配置里的标签列表，实现“改配置即改模型”
                current_roi_ids = args.controlnet_train["weighted_loss_label"]
                labels_for_cond = perturb_labels(labels_gt, current_roi_ids, prob=mask_perturb_prob)
                                
                # --- TED-Project: 面部特征加载与投影 (增强版) ---
                # 从 batch 中获取 DINO 特征的路径列表
                dino_paths = batch["dino_feature_path"] 
                
                face_feats_list = []
                for p in dino_paths:
                    # 显式使用 weights_only=True 提高安全性，并确保加载到当前设备
                    f = torch.load(p, map_location=device, weights_only=True).squeeze()
                    # 强制转换为 float32，避免因保存时的精度问题导致与 face_adapter 类型不匹配
                    face_feats_list.append(f.to(torch.float32))
                
                # 拼合 Batch [B, 3072]
                raw_face_feat = torch.stack(face_feats_list).to(device)
                
                # 通过适配器投影到模型潜空间维度 [cite: 122, 168]
                # 这里是强监督信号注入的起点
                #projected_face_feat = face_adapter(raw_face_feat)
                # -----------------------------------
                # 必须通过广播确保所有 GPU 步调一致，防止 Batch 数量不齐时死锁
                # 保证在整个 50 Epoch 内，每一步的种子都是唯一且同步的
                current_seed = int(epoch * 100000 + step)
                g = torch.Generator() # 不传参数，默认就是 CPU
                g.manual_seed(current_seed)
                
                # 获取其他辅助条件
                spacing_tensor = batch["spacing"].to(device)
                top_region_index_tensor = batch.get("top_region_index", None)
                if top_region_index_tensor is not None: top_region_index_tensor = top_region_index_tensor.to(device)
                bottom_region_index_tensor = batch.get("bottom_region_index", None)
                if bottom_region_index_tensor is not None: bottom_region_index_tensor = bottom_region_index_tensor.to(device)
                modality_tensor = batch.get("modality", None)
                if modality_tensor is not None: modality_tensor = modality_tensor.to(device)

                optimizer.zero_grad(set_to_none=True)
                
                if rank == 0:
                    logger.info(f"--- Starting Step {step + 1} Forward Pass A ---")

                
                # 1. 准备 Rectified Flow 直线采样轨迹 [cite: 78, 125]
                noise = torch.randn_like(images).to(device)
                timesteps = noise_scheduler.sample_timesteps(images)
                # x_t = (1-t)*x_0 + t*x_1 (x_0 是噪声，x_1 是真实 Latent) [cite: 78]
                noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                model_gt = images - noise # R-Flow 的目标是预测速度场 v [cite: 77, 125]

                # 2. 准备 DINO 条件对：Active (真实特征) vs Null (零向量)
                dino_paths = batch["dino_feature_path"] 
                face_feats_list = [torch.load(p, map_location=device, weights_only=True).to(torch.float32) for p in dino_paths]
                raw_face_feat = torch.stack(face_feats_list).to(device)
                null_face_feat = torch.zeros_like(raw_face_feat) 

                # --- 新增：Mask Dropout 机制，培养 DINO 的独立预测能力 ---
                is_mask_dropped = torch.rand(1).item() < mask_dropout_prob
                
                # 决定当前 Step 输入 ControlNet 的掩码
                input_labels = torch.zeros_like(labels_gt) if is_mask_dropped else labels_for_cond

                with autocast("cuda", enabled=True):
                    # --- 支路 A: 结合当前输入 (DINO + 模态 + 现有掩码) ---
                    feat_a = face_adapter(raw_face_feat)
                    model_output_active, _, _ = compute_model_output(
                        images, input_labels, noise, timesteps, noise_scheduler,
                        controlnet, unet, spacing_tensor,
                        face_features=feat_a,
                        modality_tensor=modality_tensor,
                        top_region_index_tensor=top_region_index_tensor,
                        bottom_region_index_tensor=bottom_region_index_tensor,
                    )
                    
                    # 基础生成损失：确保不管有没有 Mask，生成的 CT 质量都要过关
                    loss_recons = F.l1_loss(model_output_active.float(), model_gt.float())

                    # --- 支路 B: 对照组 (仅模态 + 现有掩码) ---
                    model_output_null, _, _ = compute_model_output(
                        images, input_labels, noise, timesteps, noise_scheduler,
                        controlnet, unet, spacing_tensor,
                        face_features=None, # 强制支路 B 不含面部特征
                        modality_tensor=modality_tensor,
                        top_region_index_tensor=top_region_index_tensor,
                        bottom_region_index_tensor=bottom_region_index_tensor,
                    )

                    # 3. 计算 RSCL：无论输入是否 Dropout，对比基准永远看真实对齐的 labels_gt ---
                    roi_mask = (labels_gt >= 2).to(torch.uint8)
                    dilated_mask = F.max_pool3d(roi_mask.float(), kernel_size=5, stride=1, padding=2)
                    roi_bg_mask = (1 - (dilated_mask > 0).to(torch.uint8))

                    # 强制 DINO 信号在解剖学正确的位置产生差异 [cite: 146, 147]
                    l_roi, l_bg = compute_region_contrastive_loss_ted(
                        model_output_active,
                        model_output_null,
                        roi_mask,
                        roi_bg_mask,
                        max_region_contrastive_loss=RSCL_FACE_MAX_RESPONSE,
                    )
                    
                    # 总损失叠加
                    total_loss = loss_recons + rscl_face_weight * (l_roi + l_bg)

                # 4. 统一反向传播 (不再分两步，提高数值稳定性)
                scaler.scale(total_loss).backward()
                    
                # 记录总损失用于后续指标计算
                loss = total_loss.detach()
                    
                # 3. 唯一次反向传播
                # 这样可以保证所有 DDP 钩子只触发一次，且没有中间变量被修改的风险
                if rank == 0:
                    logger.info(f"--- [Step {step + 1}] 所有计算完成，准备同步梯度并更新 ---")
                scaler.step(optimizer)
                scaler.update()

                loss = total_loss
                
                lr_scheduler.step()
                total_step += 1

                if rank == 0:
                    # write train loss for each batch into tensorboard
                    tensorboard_writer.add_scalar("train/train_controlnet_loss_iter", loss.detach().cpu().item(), total_step)
                    # --- 新增代码：记录强监督关键指标 ---
                    # l_roi approaches -2 as facial sensitivity in the orbit increases;
                    # l_bg approaches zero as off-target background effects decrease.
                    tensorboard_writer.add_scalar("train/loss_roi_sensitivity_iter", l_roi.detach().cpu().item(), total_step)
                    tensorboard_writer.add_scalar("train/loss_bg_consistency_iter", l_bg.detach().cpu().item(), total_step)
                    #tensorboard_writer.flush()
            # -----------------------------------
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                # --- TED-Project 增强版日志打印：输出所有分项损失 ---
                loss_str = f"[Total Loss: {loss.detach().cpu().item():.4f}] "
                loss_str += f"[Rec Loss: {loss_recons.detach().cpu().item():.4f}] "
                
                # 如果开启了强监督，打印 ROI 和 BG 损失
                loss_str += f"[ROI Sens: {l_roi.detach().cpu().item():.4f}] "
                loss_str += f"[BG Consist: {l_bg.detach().cpu().item():.4f}] "

                logger.info(
                    f"\r[Epoch {epoch + 1}/{n_epochs}] [Batch {step + 1}/{len(train_loader)}] "
                    f"[LR: {lr_scheduler.get_last_lr()[0]:.8f}] {loss_str} ETA: {time_left} "
                )
                epoch_loss_ += loss.detach()
                epoch_l_roi_ += l_roi.detach()
                epoch_l_bg_ += l_bg.detach()

        epoch_loss = epoch_loss_ / (step + 1)
        # --- 计算平均值 ---
        avg_l_roi = epoch_l_roi_ / (step + 1)
        avg_l_bg = epoch_l_bg_ / (step + 1)
        # ------------------

        if use_ddp:
            # 移除 dist.barrier()，因为它经常是死锁的元凶
            # 使用更稳健的同步方式
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.SUM)
            epoch_loss /= world_size
            
            dist.all_reduce(avg_l_roi, op=torch.distributed.ReduceOp.SUM)
            avg_l_roi /= world_size
            dist.all_reduce(avg_l_bg, op=torch.distributed.ReduceOp.SUM)
            avg_l_bg /= world_size
            # -----------------

        if rank == 0:
            import shutil  # 建议在函数开头 import，或者直接在这里引入
            
            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss.cpu().item(), total_step)
            # --- 记录到 TensorBoard ---
            tensorboard_writer.add_scalar("train/loss_roi_sensitivity_epoch", avg_l_roi.cpu().item(), total_step)
            tensorboard_writer.add_scalar("train/loss_bg_consistency_epoch", avg_l_bg.cpu().item(), total_step)
            
            # 1. 提取状态字典 (保持你原有的 DDP 判断逻辑)
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
            adapter_state_dict = face_adapter.module.state_dict() if world_size > 1 else face_adapter.state_dict()
            
            checkpoint_data = {
                "epoch": epoch + 1,
                "crossmodal_phase": crossmodal_phase,
                "loss": epoch_loss,
                "controlnet_state_dict": controlnet_state_dict,
                "face_adapter_state_dict": adapter_state_dict,
            }
            
            # 2. 【全场唯一一次写入】先保存为 current.pt
            current_path = f"{args.model_dir}/{args.exp_name}_current.pt"
            torch.save(checkpoint_data, current_path)
            logger.info(f"Successfully saved current model to {current_path}")

            # 3. 【方案 B 核心：拷贝生成周期档】
            if (epoch + 1) % 10 == 0:
                epoch_ckpt_path = f"{args.model_dir}/{args.exp_name}_epoch_{epoch + 1}.pt"
                try:
                    shutil.copy2(current_path, epoch_ckpt_path) # 直接拷贝 current 文件
                    logger.info(f"--- Periodic Checkpoint Copied: {epoch_ckpt_path} ---")
                except Exception as e:
                    logger.warning(f"Failed to copy periodic checkpoint: {e}")

            # 4. 【方案 B 核心：拷贝生成最优档】
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f"New best loss achieved: {best_loss:.4f}")
                
                best_path = f"{args.model_dir}/{args.exp_name}_best.pt"
                try:
                    shutil.copy2(current_path, best_path) # 同样使用拷贝，不再重复 torch.save
                    logger.info(f"Best model updated (copied) to {best_path}")
                except Exception as e:
                    logger.error(f"Failed to update best model via copy: {e}")

        torch.cuda.empty_cache()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet Model Training")
    parser.add_argument(
        "-e",
        "--env_config_path",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "-c",
        "--model_config_path",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument("-t", "--model_def_path", type=str, default="./configs/config_maisi.json", help="Path to model definition file")
    parser.add_argument("-g", "--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument(
        "--crossmodal-phase",
        type=int,
        choices=(1, 2, 3, 4),
        required=True,
        help="One of the four reported 30-epoch cross-modal phases.",
    )

    args = parser.parse_args()
    train_controlnet(
        args.env_config_path,
        args.model_config_path,
        args.model_def_path,
        args.num_gpus,
        args.crossmodal_phase,
    )
