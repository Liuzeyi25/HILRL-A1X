"""
训练脚本：基于 ResNet-18 + MLP 的任务进度估计模型

P(o, s) → 进度值 ∈ [0, 1]

输入：
  - side_img  : (256, 256, 3) 侧面摄像头图像
  - wrist_img : (128, 128, 3) 腕部摄像头图像
  - state     : (7,) 机器人关节状态 (TCP位置xyz + 欧拉角rpy + 夹爪)

两阶段流程：
  Phase 1: 用冻结的 ResNet-18 预提取所有图像特征并缓存到磁盘
  Phase 2: 仅训练 MLP head，使用 MC / TD / 进度 三种损失
  Phase 3: 评估 buffer 中 30 条成功轨迹并可视化

数据：segmentation/liuzeyi_data/insert_block/demo_data/20260228
        （10 条纯人类 demo，全部成功，末尾 reward=1.0）
评估：segmentation/liuzeyi_data/insert_block/buffer/buffer
        （取前 30 条 reward>=1.0 的成功轨迹）
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────────────
# 常量 & 工具
# ──────────────────────────────────────────────────────────────────────

SIDE_KEY  = "side_policy_256"
WRIST_KEY = "wrist_1"
STATE_DIM = 7
FEAT_DIM  = 512   # ResNet-18 最终池化层输出维度（side + wrist 各 512，拼接后 1024）


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ──────────────────────────────────────────────────────────────────────
# ResNet-18 特征提取器（冻结，仅用作特征提取）
# ──────────────────────────────────────────────────────────────────────

class ResNetExtractor(nn.Module):
    """截断到全局平均池化层之前的 ResNet-18，输出 512 维特征。"""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 去掉最后的 fc 层，保留到 avgpool
        self.feature = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, 512)"""
        return self.feature(x).flatten(1)


# 图像预处理（ResNet-18 标准归一化）
_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def img_array_to_pil(arr: np.ndarray) -> Image.Image:
    """将 (H, W, 3) uint8 → PIL RGB Image"""
    if arr.ndim == 4:
        arr = arr[0]   # 取 row1（当前帧）
    elif arr.ndim == 2:
        arr = arr[np.newaxis]
    return Image.fromarray(arr.astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────
# MLP 进度头
# ──────────────────────────────────────────────────────────────────────

class ProgressHead(nn.Module):
    """
    输入：
      图像差值: side_feat - side_feat_0  (512)
              wrist_feat - wrist_feat_0 (512)
      状态差值: state - state_0           (7)  → state_encoder(64)
    输出： 进度标量 ∈ [0, 1]

    结构：
      state_encoder : 7 → 64  （将结构䯵信息升维再拼接，避免被 1024 维图像湿没）
      fusion_net   : 1024+64 → hidden → hidden//2 → 1
    """

    def __init__(self, feat_dim: int = 1024, state_dim: int = STATE_DIM,
                 hidden: int = 256, state_enc_dim: int = 64):
        super().__init__()
        # 状态历独立升维支路
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_enc_dim),
            nn.LayerNorm(state_enc_dim),
            nn.ReLU(),
        )
        in_dim = feat_dim + state_enc_dim  # 1024 + 64 = 1088
        self.fusion_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        d_side: torch.Tensor,   # (B, 512)  side_feat  - side_feat_0
        d_wrist: torch.Tensor,  # (B, 512)  wrist_feat - wrist_feat_0
        d_state: torch.Tensor,  # (B, 7)    state      - state_0
    ) -> torch.Tensor:
        """→ (B,) 进度值"""
        s_enc = self.state_encoder(d_state)            # (B, 64)
        x = torch.cat([d_side, d_wrist, s_enc], dim=-1)  # (B, 1088)
        return self.fusion_net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Phase 1：特征提取 & 缓存
# ──────────────────────────────────────────────────────────────────────

def extract_and_cache_demo_features(
    demo_dir: str,
    cache_path: str,
    device: str,
    batch_size: int = 32,
) -> dict:
    """
    遍历 demo_dir 下所有 traj_*.pkl，提取每个 transition 的
    side/wrist ResNet-18 特征、state、progress 标签、
    MC 回报（gamma=0.99）和 done。

    缓存结构（torch 张量）：
      side_feats   : (N, 512)
      wrist_feats  : (N, 512)
      states       : (N, 7)
      next_side_feats  : (N, 512)
      next_wrist_feats : (N, 512)
      next_states  : (N, 7)
      progress     : (N,)   —— t / (T-1)，最后一步=1.0
      mc_returns   : (N,)   —— γ^(T-1-t)（稀疏末端奖励）
      rewards      : (N,)   —— 0/1 稀疏奖励
      dones        : (N,)   —— 是否最后一步
      traj_id      : (N,)   —— 所属轨迹索引
      step_id      : (N,)   —— 轨迹内步骤索引
      traj_len     : (N,)   —— 所属轨迹总长度
    """
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        if "init_side_feats" in cache:
            log(f"[cache] 加载已有缓存: {cache_path} ({cache['side_feats'].shape[0]} transitions)")
            return cache
        else:
            log(f"[cache] 旧格式缓存（无 init 字段），重新生成: {cache_path}")

    log(f"[phase1] 开始提取特征，缓存到: {cache_path}")
    extractor = ResNetExtractor().to(device).eval()

    files = sorted(glob.glob(os.path.join(demo_dir, "traj_*.pkl")))
    if not files:
        raise FileNotFoundError(f"未找到 demo 文件: {demo_dir}/traj_*.pkl")
    log(f"[phase1] 找到 {len(files)} 条 demo")

    gamma = 0.99

    # 收集所有图像用于批量提取特征
    all_side_pils: list[Image.Image] = []
    all_wrist_pils: list[Image.Image] = []
    all_next_side_pils: list[Image.Image] = []
    all_next_wrist_pils: list[Image.Image] = []
    all_init_side_pils: list[Image.Image] = []   # t=0 帧（按 transition 广播）
    all_init_wrist_pils: list[Image.Image] = []

    # 标量数据逐步收集
    all_states, all_next_states, all_init_states = [], [], []
    all_progress, all_mc, all_rewards, all_dones = [], [], [], []
    all_traj_id, all_step_id, all_traj_len = [], [], []

    for traj_id, fpath in enumerate(files):
        with open(fpath, "rb") as fh:
            transitions = pickle.load(fh)
        T = len(transitions)
        if T == 0:
            continue

        # 构造稀疏奖励（末端=1.0）和 MC 回报
        rewards = np.zeros(T, dtype=np.float32)
        rewards[-1] = 1.0  # demo 全部成功
        mc_returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for i in range(T - 1, -1, -1):
            running = rewards[i] + gamma * running
            mc_returns[i] = running

        # 线性进度标签
        denom = max(T - 1, 1)
        progress = np.array([i / denom for i in range(T)], dtype=np.float32)

        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0

        # 轨迹 t=0 的观测（用于计算差值）
        init_obs = transitions[0]["observations"]
        init_side_pil  = img_array_to_pil(np.array(init_obs[SIDE_KEY])[1])
        init_wrist_pil = img_array_to_pil(np.array(init_obs[WRIST_KEY])[1])
        init_state_arr = np.array(init_obs["state"])[1].astype(np.float32)

        for step, tr in enumerate(transitions):
            obs = tr["observations"]
            next_obs = tr["next_observations"]

            # 当前观测图像
            side_arr  = np.array(obs[SIDE_KEY])
            wrist_arr = np.array(obs[WRIST_KEY])
            # next 观测图像（最后一步用自身）
            next_side_arr  = np.array(next_obs[SIDE_KEY])
            next_wrist_arr = np.array(next_obs[WRIST_KEY])

            # shape=(2,H,W,3) → 取 row1（索引 1 = 当前帧）
            all_side_pils.append(img_array_to_pil(side_arr[1]))
            all_wrist_pils.append(img_array_to_pil(wrist_arr[1]))
            all_next_side_pils.append(img_array_to_pil(next_side_arr[1]))
            all_next_wrist_pils.append(img_array_to_pil(next_wrist_arr[1]))
            # t=0 帧图像（每个 transition 广播同一个）
            all_init_side_pils.append(init_side_pil)
            all_init_wrist_pils.append(init_wrist_pil)

            # state: (2,7) → 取 row1
            state = np.array(obs["state"])[1]
            next_state = np.array(next_obs["state"])[1]
            all_states.append(state)
            all_next_states.append(next_state)
            all_init_states.append(init_state_arr)  # 轨迹起始状态

            all_progress.append(progress[step])
            all_mc.append(mc_returns[step])
            all_rewards.append(rewards[step])
            all_dones.append(dones[step])
            all_traj_id.append(traj_id)
            all_step_id.append(step)
            all_traj_len.append(T)

        log(f"  [phase1] traj {traj_id+1}/{len(files)}: T={T}  "
            f"mc[0]={mc_returns[0]:.4f}")

    log(f"[phase1] 共 {len(all_side_pils)} 个 transition，开始批量提取特征...")

    def batch_extract(pils: list[Image.Image]) -> torch.Tensor:
        feats = []
        for start in range(0, len(pils), batch_size):
            batch = pils[start:start + batch_size]
            tensors = torch.stack([_resnet_transform(p) for p in batch]).to(device)
            with torch.no_grad():
                f = extractor(tensors).cpu()
            feats.append(f)
        return torch.cat(feats, dim=0)

    side_feats       = batch_extract(all_side_pils)
    wrist_feats      = batch_extract(all_wrist_pils)
    next_side_feats  = batch_extract(all_next_side_pils)
    next_wrist_feats = batch_extract(all_next_wrist_pils)
    init_side_feats  = batch_extract(all_init_side_pils)
    init_wrist_feats = batch_extract(all_init_wrist_pils)
    log(f"[phase1] 特征提取完成: side={side_feats.shape}")

    cache = {
        "side_feats":        side_feats,
        "wrist_feats":       wrist_feats,
        "next_side_feats":   next_side_feats,
        "next_wrist_feats":  next_wrist_feats,
        "init_side_feats":   init_side_feats,   # (N, 512) 轨迹 t=0 帧特征
        "init_wrist_feats":  init_wrist_feats,  # (N, 512)
        "states":      torch.tensor(np.array(all_states),       dtype=torch.float32),
        "next_states": torch.tensor(np.array(all_next_states),  dtype=torch.float32),
        "init_states": torch.tensor(np.array(all_init_states),  dtype=torch.float32),  # (N, 7)
        "progress":    torch.tensor(all_progress, dtype=torch.float32),
        "mc_returns":  torch.tensor(all_mc,       dtype=torch.float32),
        "rewards":     torch.tensor(all_rewards,  dtype=torch.float32),
        "dones":       torch.tensor(all_dones,    dtype=torch.float32),
        "traj_id":     torch.tensor(all_traj_id,  dtype=torch.long),
        "step_id":     torch.tensor(all_step_id,  dtype=torch.long),
        "traj_len":    torch.tensor(all_traj_len, dtype=torch.long),
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(cache, cache_path)
    log(f"[phase1] 缓存已保存: {cache_path}")
    return cache


# ──────────────────────────────────────────────────────────────────────
# 数据集 & 归一化
# ──────────────────────────────────────────────────────────────────────

class ProgressDataset(Dataset):
    def __init__(self, cache: dict):
        self.cache = cache
        self.N = cache["side_feats"].shape[0]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        c = self.cache
        # state 差值归一化: (s_t - s_0 - mean) / std
        d_state      = (c["states"][idx]      - c["init_states"][idx] - c["state_mean"]) / c["state_std"]
        next_d_state = (c["next_states"][idx] - c["init_states"][idx] - c["state_mean"]) / c["state_std"]
        return {
            # 返回差值: feat_t - feat_0
            "d_side_feat":       c["side_feats"][idx]  - c["init_side_feats"][idx],
            "d_wrist_feat":      c["wrist_feats"][idx] - c["init_wrist_feats"][idx],
            "d_state":           d_state,
            # next 帧差值
            "next_d_side_feat":  c["next_side_feats"][idx]  - c["init_side_feats"][idx],
            "next_d_wrist_feat": c["next_wrist_feats"][idx] - c["init_wrist_feats"][idx],
            "next_d_state":      next_d_state,
            "progress":          c["progress"][idx],
            "mc_returns":        c["mc_returns"][idx],
            "rewards":           c["rewards"][idx],
            "dones":             c["dones"][idx],
            "step_id":           c["step_id"][idx],
            "traj_len":          c["traj_len"][idx],
        }


def compute_state_stats(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """在差值空间计算均值和标准差（用于幯化 state 差值）"""
    mean = states.mean(dim=0)
    std  = states.std(dim=0).clamp(min=1e-3)
    return mean, std


# ──────────────────────────────────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────────────────────────────────

def compute_losses(
    model: ProgressHead,
    batch: dict,
    gamma: float,
    lambda_prog: float,
    lambda_mc: float,
    lambda_td: float,
    device: str,
    mc_last_n: int = 0,
) -> dict[str, torch.Tensor]:
    """
    三种损失：
      L_prog = MSE(P(s_t), t/(T-1))               直接拟合线性进度
      L_mc   = MSE(P(s_t), G_t)                    拟合 MC 折扣回报
      L_td   = MSE(P(s_t), r_t + γ*(1-d)*P(s_{t+1}))  Bellman 一致性
    """
    side  = batch["d_side_feat"].to(device)
    wrist = batch["d_wrist_feat"].to(device)
    state = batch["d_state"].to(device)

    pred = model(side, wrist, state)  # (B,)

    # ── 进度损失 ──
    prog_target = batch["progress"].to(device)
    loss_prog = F.mse_loss(pred, prog_target)

    # ── MC 损失（仅对轨迹最后 mc_last_n 步） ──
    mc_target = batch["mc_returns"].to(device)
    if mc_last_n > 0:
        step_id  = batch["step_id"].to(device)
        traj_len = batch["traj_len"].to(device)
        mc_mask  = (step_id >= traj_len - mc_last_n).float()  # 最后 N 步=1
        if mc_mask.sum() > 0:
            loss_mc = (mc_mask * (pred - mc_target) ** 2).sum() / mc_mask.sum()
        else:
            loss_mc = torch.tensor(0.0, device=device)
    else:
        loss_mc = F.mse_loss(pred, mc_target)

    # ── TD 损失 ──
    with torch.no_grad():
        next_pred = model(
            batch["next_d_side_feat"].to(device),
            batch["next_d_wrist_feat"].to(device),
            batch["next_d_state"].to(device),
        )
        td_target = (batch["rewards"].to(device)
                     + gamma * (1.0 - batch["dones"].to(device)) * next_pred)

    loss_td = F.mse_loss(pred, td_target)

    total = lambda_prog * loss_prog + lambda_mc * loss_mc + lambda_td * loss_td

    return {
        "loss": total,
        "loss_prog": loss_prog,
        "loss_mc": loss_mc,
        "loss_td": loss_td,
        "pred_mean": pred.mean(),
        "pred_min": pred.min(),
        "pred_max": pred.max(),
    }


# ──────────────────────────────────────────────────────────────────────
# Phase 2：训练
# ──────────────────────────────────────────────────────────────────────

def train(args) -> ProgressHead:
    set_seed(args.seed)
    device = args.device

    # ── Phase 1：提取并缓存特征 ──
    cache = extract_and_cache_demo_features(
        args.demo_dir,
        args.cache_path,
        device=device,
        batch_size=args.phase1_batch_size,
    )

    # ── 差值归一化统计：先计算状态差值再求 mean/std ──
    d_states_raw = cache["states"] - cache["init_states"]  # (N, 7)
    state_mean, state_std = compute_state_stats(d_states_raw)
    torch.save({"mean": state_mean, "std": state_std},
               os.path.join(args.output_dir, "state_stats.pt"))
    log(f"[train] d_state mean={state_mean.numpy().round(4)}")
    log(f"[train] d_state std= {state_std.numpy().round(4)}")

    # 将差值归一化写入 cache（DataLoader 起取时直接返回归一化后的差值）
    cache = {k: v.clone() if isinstance(v, torch.Tensor) else v
             for k, v in cache.items()}
    # 对所有 feat 字段无需额外归一化（ResNet 输出已在相同尺度上）
    # 对 state 差值归一化
    for key in ("states", "next_states", "init_states"):
        cache[key] = cache[key]  # 保留原始值，由 Dataset 实时减去 init 并除以 std
    # 在 Dataset 计算 d_state 后除以 std；将 std 导入 cache 以供 Dataset 使用
    cache["state_std"]  = state_std
    cache["state_mean"] = state_mean

    # ── 数据集 & DataLoader ──
    dataset = ProgressDataset(cache)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )
    log(f"[train] dataset size={len(dataset)}, "
        f"batch_size={args.batch_size}, "
        f"steps/epoch={len(loader)}")

    # ── 模型 ──
    model = ProgressHead(feat_dim=1024, state_dim=STATE_DIM,
                         hidden=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 训练循环 ──
    log(f"\n[train] 开始训练，共 {args.epochs} 个 epoch")
    best_loss = float("inf")
    history = {"loss": [], "loss_prog": [], "loss_mc": [], "loss_td": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_stats = {k: 0.0 for k in history}
        n_batches = 0

        for batch in loader:
            optimizer.zero_grad()
            losses = compute_losses(
                model, batch, args.gamma,
                args.lambda_prog, args.lambda_mc, args.lambda_td,
                device,
                mc_last_n=args.mc_last_n,
            )
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in history:
                epoch_stats[k] += losses[k].item()
            n_batches += 1

        scheduler.step()

        for k in history:
            epoch_stats[k] /= max(n_batches, 1)
            history[k].append(epoch_stats[k])

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  epoch {epoch:4d}/{args.epochs}  "
                f"loss={epoch_stats['loss']:.5f}  "
                f"prog={epoch_stats['loss_prog']:.5f}  "
                f"mc={epoch_stats['loss_mc']:.5f}  "
                f"td={epoch_stats['loss_td']:.5f}  "
                f"lr={lr_now:.2e}")

        if epoch_stats["loss"] < best_loss:
            best_loss = epoch_stats["loss"]
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "progress_model_best.pt"))

    # 保存最终模型
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "progress_model_final.pt"))
    log(f"[train] 训练完成，最佳 loss={best_loss:.5f}")

    # 绘制训练曲线
    _plot_training_curves(history, args.output_dir)

    # 绘制 demo 进度曲线（验证训练效果）
    _plot_demo_progress_curves(model, cache, state_mean, state_std, args)

    return model


def _plot_training_curves(history: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="total")
    axes[0].set_title("Total Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["loss_prog"], label="progress")
    axes[1].plot(history["loss_mc"],   label="MC")
    axes[1].plot(history["loss_td"],   label="TD")
    axes[1].set_title("Loss Components"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log(f"[plot] 训练曲线保存: {out}")


@torch.no_grad()
def _plot_demo_progress_curves(
    model: ProgressHead,
    cache: dict,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    args,
) -> None:
    """绘制每条 demo 的预测进度 vs 真实进度"""
    model.eval()
    n_traj = int(cache["traj_id"].max().item()) + 1
    ncols = min(5, n_traj)
    nrows = (n_traj + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for tid in range(n_traj):
        mask = cache["traj_id"] == tid
        idx  = torch.where(mask)[0]
        ax   = axes[tid]

        # 计算差值并归一化
        d_side  = cache["side_feats"][idx]  - cache["init_side_feats"][idx]
        d_wrist = cache["wrist_feats"][idx] - cache["init_wrist_feats"][idx]
        d_state = (cache["states"][idx] - cache["init_states"][idx] - state_mean) / state_std

        pred = model(
            d_side.to(args.device),
            d_wrist.to(args.device),
            d_state.to(args.device),
        ).cpu().numpy()
        true_prog = cache["progress"][idx].numpy()
        mc        = cache["mc_returns"][idx].numpy()

        x = np.arange(len(pred))
        ax.plot(x, true_prog, label="true progress", linestyle="--",
                color="#2ecc71")
        ax.plot(x, mc,        label="MC return",     linestyle=":",
                color="#e67e22")
        ax.plot(x, pred,      label="predicted P",   color="#2980b9", lw=2)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Demo {tid}  T={len(pred)}", fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)

    for i in range(n_traj, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Demo Progress Predictions (train set)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.output_dir, "demo_progress_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log(f"[plot] Demo 进度曲线保存: {out}")


# ──────────────────────────────────────────────────────────────────────
# Phase 3：评估 buffer 中的成功轨迹并可视化
# ──────────────────────────────────────────────────────────────────────

def get_buffer_files_sorted(buffer_dir: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".pkl", "")),
    )


def load_success_trajectories(
    buffer_dir: str,
    n: int = 50,
    min_rl_ratio: float = 0.3,
) -> list[dict]:
    """从 buffer 中加载前 n 条满足以下条件的轨迹：
      - reward >= 1.0（成功）
      - RL 步骤占比 >= min_rl_ratio（过滤正为纯人工的轨迹）
    """
    files = get_buffer_files_sorted(buffer_dir)
    results = []
    for fpath in files:
        if len(results) >= n:
            break
        with open(fpath, "rb") as fh:
            transitions = pickle.load(fh)
        rewards = [t["rewards"] for t in transitions]
        if max(rewards) < 1.0:
            continue

        # 计算 RL 占比
        labels_all = [t.get("labels", 1) for t in transitions]
        n_rl_steps = sum(1 for lb in labels_all if lb == 1)
        rl_ratio   = n_rl_steps / max(len(labels_all), 1)
        if rl_ratio < min_rl_ratio:
            continue

        states_list  = [np.array(t["observations"]["state"])[1]  for t in transitions]
        side_list    = [np.array(t["observations"][SIDE_KEY])[1]  for t in transitions]
        wrist_list   = [np.array(t["observations"][WRIST_KEY])[1] for t in transitions]
        labels_list  = labels_all
        rewards_list = rewards

        # 附上最后一帧的 next
        last_tr = transitions[-1]
        states_list.append(np.array(last_tr["next_observations"]["state"])[1])
        side_list.append(np.array(last_tr["next_observations"][SIDE_KEY])[1])
        wrist_list.append(np.array(last_tr["next_observations"][WRIST_KEY])[1])
        labels_list.append(labels_list[-1])
        rewards_list.append(last_tr["rewards"])

        traj_idx = int(os.path.basename(fpath).split("_")[-1].replace(".pkl", ""))
        results.append({
            "traj_idx":  traj_idx,
            "states":    np.array(states_list,  dtype=np.float32),
            "side_imgs": np.array(side_list,    dtype=np.uint8),
            "wrist_imgs":np.array(wrist_list,   dtype=np.uint8),
            "labels":    np.array(labels_list),
            "rewards":   np.array(rewards_list, dtype=np.float32),
            "rl_ratio":  rl_ratio,
        })

    log(f"[eval] 加载了 {len(results)} 条轨迹（reward≥1.0 且 RL占比≥0.3）")
    return results


@torch.no_grad()
def evaluate_and_visualize(
    model: ProgressHead,
    buffer_dir: str,
    output_dir: str,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    extractor: ResNetExtractor,
    args,
) -> None:
    """
    对 buffer 中高 RL 占比的成功轨迹做进度预测，
    每条轨迹生成一张可视化图。
    """
    vis_dir = os.path.join(output_dir, "eval_visuals")
    os.makedirs(vis_dir, exist_ok=True)

    device = args.device
    extractor = extractor.to(device).eval()
    model.eval()

    trajs = load_success_trajectories(
        buffer_dir, n=args.n_eval_trajs,
        min_rl_ratio=getattr(args, "min_rl_ratio", 0.3),
    )

    all_anomalies: dict[int, list[dict]] = {}

    for traj_data in trajs:
        traj_idx  = traj_data["traj_idx"]
        states    = traj_data["states"]     # (T+1, 7)
        side_imgs = traj_data["side_imgs"]  # (T+1, H, W, 3)
        wrist_imgs= traj_data["wrist_imgs"] # (T+1, h, w, 3)
        labels    = traj_data["labels"]     # (T+1,)
        T_total   = len(states)             # T+1 帧
        T = T_total - 1

        # ── 对所有帧提取特征（包括最后一帧） ──
        side_pils  = [img_array_to_pil(side_imgs[i])  for i in range(T_total)]
        wrist_pils = [img_array_to_pil(wrist_imgs[i]) for i in range(T_total)]

        def extract_pils(pils):
            feats = []
            for start in range(0, len(pils), args.phase1_batch_size):
                batch = pils[start:start + args.phase1_batch_size]
                t = torch.stack([_resnet_transform(p) for p in batch]).to(device)
                feats.append(extractor(t).cpu())
            return torch.cat(feats, dim=0)

        side_feats  = extract_pils(side_pils)   # (T+1, 512)
        wrist_feats = extract_pils(wrist_pils)  # (T+1, 512)

        # ── 差值归一化并预测 ──
        side_feats_t  = side_feats.to(device)   # (T+1, 512)
        wrist_feats_t = wrist_feats.to(device)  # (T+1, 512)
        # t=0 帧特征
        sf0 = side_feats_t[0:1]    # (1, 512)
        wf0 = wrist_feats_t[0:1]   # (1, 512)
        d_side  = side_feats_t  - sf0
        d_wrist = wrist_feats_t - wf0
        # state 差值归一化
        states_t   = torch.tensor(states, dtype=torch.float32)
        d_state    = (states_t - states_t[0:1] - state_mean) / state_std

        preds = model(
            d_side,
            d_wrist,
            d_state.to(device),
        ).cpu().numpy()  # (T+1,)

        # ── 统计 ──
        n_rl    = int((labels[:T] == 1).sum())
        n_human = int((labels[:T] == 2).sum())

        # ── 异常检测 ──
        anomalies = detect_anomalies(
            preds, labels,
            window_size=args.anomaly_window,
            delta_reg=args.delta_reg,
            delta_stag=args.delta_stag,
            detect_regression=bool(args.detect_regression),
            detect_stagnation=bool(args.detect_stagnation),
            recovery_k=args.recovery_k,
        )
        if anomalies:
            all_anomalies[traj_idx] = anomalies

        # ── 绘图 ──
        _plot_trajectory_summary(
            traj_idx, side_imgs, wrist_imgs, labels, preds,
            n_rl, n_human, T, vis_dir,
            anomalies=anomalies,
        )
        _save_trajectory_video(
            traj_idx, side_imgs, wrist_imgs, labels, preds,
            n_rl, n_human, T, vis_dir,
            fps=args.video_fps,
            anomalies=anomalies,
        )
        rl_ratio = traj_data.get("rl_ratio", 0.0)
        log(f"  traj_{traj_idx:03d}: T={T}  RL={n_rl}({rl_ratio:.0%})  Human={n_human}  "
            f"progress[-1]={preds[-1]:.3f}")
        if anomalies:
            seg_strs = [
                f"[{s['type']} t={s['start']}-{s['end']} "
                f"ΔP={s['delta_p']:+.3f} len={s['duration']}]"
                for s in anomalies
            ]
            log(f"    异常 ({len(anomalies)}段): {' '.join(seg_strs)}")

    # 保存异常检测结果汇总
    if all_anomalies:
        import json
        summary_path = os.path.join(vis_dir, "anomaly_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {str(k): v for k, v in all_anomalies.items()},
                f, indent=2, ensure_ascii=False,
            )
        n_total_segs = sum(len(v) for v in all_anomalies.values())
        log(f"[eval] 异常检测汇总: {len(all_anomalies)}/{len(trajs)} 条轨迹存在异常, "
            f"共 {n_total_segs} 个异常段")
        log(f"[eval] 异常结果已保存: {summary_path}")
    else:
        log(f"[eval] 所有 {len(trajs)} 条轨迹未检测到异常")

    log(f"[eval] 可视化图像保存在: {vis_dir}/")


def _plot_trajectory_summary(
    traj_idx: int,
    side_imgs: np.ndarray,
    wrist_imgs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    n_rl: int,
    n_human: int,
    T: int,
    vis_dir: str,
    anomalies: list | None = None,
) -> None:
    """
    为单条轨迹绘制摘要图（静态图，不生成视频）：

    布局（4 行）：
      Row 0: 侧面相机关键帧（T=0, T/4, T/2, 3T/4, T）
      Row 1: 腕部相机关键帧（同上）
      Row 2: 进度预测曲线（RL段 / Human段 区分颜色）
      Row 3: 控制类型标注条（RL=橙 / Human=绿）
    """
    # 选取关键帧索引
    kf_count  = 5
    kf_idxs   = [int(round(i / (kf_count - 1) * T)) for i in range(kf_count)]
    kf_idxs   = [min(k, len(side_imgs) - 1) for k in kf_idxs]

    fig = plt.figure(figsize=(4 * kf_count, 12), facecolor="white")
    gs  = GridSpec(4, kf_count, height_ratios=[3, 3, 4, 1.2],
                   hspace=0.35, wspace=0.08,
                   left=0.05, right=0.97, top=0.92, bottom=0.04)

    fig.suptitle(
        f"Traj {traj_idx}  |  T={T}  RL={n_rl}  Human={n_human}  "
        f"final_progress={preds[T]:.3f}",
        fontsize=13, fontweight="bold",
    )

    # ── Row 0/1: 关键帧 ──
    for col, kf in enumerate(kf_idxs):
        for row, (imgs, cam_name) in enumerate(
            [(side_imgs, "Side"), (wrist_imgs, "Wrist")]
        ):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(imgs[kf])
            ax.set_title(f"{cam_name} t={kf}", fontsize=8)
            ax.axis("off")
            # 边框颜色：RL=橙, Human=绿
            color = "#2ecc71" if labels[kf] == 2 else "#f39c12"
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(color)
                sp.set_linewidth(3)

    # ── Row 2: 进度曲线 ──
    ax_prog = fig.add_subplot(gs[2, :])
    t_axis  = np.arange(T + 1)

    # 背景色：Human 段浅绿，RL 段浅黄
    for i in range(T):
        c = "#e8f8f5" if labels[i] == 2 else "#fef9e7"
        ax_prog.axvspan(i - 0.5, i + 0.5, color=c, alpha=0.7, lw=0)

    # 异常段高亮（红色=回退，橙色=停滞）
    if anomalies:
        for seg in anomalies:
            c = "#e74c3c" if seg["type"] == "regression" else "#f39c12"
            ax_prog.axvspan(seg["start"] - 0.5, seg["end"] + 0.5,
                            color=c, alpha=0.20, zorder=2, lw=0)

    # 分段绘制进度曲线（RL 蓝色，Human 绿色）
    for i in range(T):
        color = "#27ae60" if labels[i] == 2 else "#2980b9"
        ax_prog.plot([i, i + 1], [preds[i], preds[i + 1]],
                     color=color, lw=2.5)

    # 真实线性进度参考线
    true_prog = np.linspace(0, 1, T + 1)
    ax_prog.plot(t_axis, true_prog, "--", color="#e74c3c", lw=1.2,
                 alpha=0.7, label="linear reference (t/T)")
    ax_prog.plot(t_axis, preds, ".", color="#34495e", markersize=4,
                 alpha=0.5)

    ax_prog.set_xlim(-0.5, T + 0.5)
    ax_prog.set_ylim(-0.05, 1.10)
    ax_prog.set_xlabel("Timestep", fontsize=10)
    ax_prog.set_ylabel("Predicted Progress P(o,s)", fontsize=10)
    ax_prog.set_title("Progress Prediction", fontsize=11)
    ax_prog.grid(alpha=0.25)
    legend_elems = [
        Patch(facecolor="#2980b9", label="RL control"),
        Patch(facecolor="#27ae60", label="Human control"),
    ]
    if anomalies:
        if any(s["type"] == "regression" for s in anomalies):
            legend_elems.append(Patch(facecolor="#e74c3c", alpha=0.3,
                                     label="Regression"))
        if any(s["type"] == "stagnation" for s in anomalies):
            legend_elems.append(Patch(facecolor="#f39c12", alpha=0.3,
                                     label="Stagnation"))
    ax_prog.legend(handles=legend_elems + [
        plt.Line2D([0], [0], color="#e74c3c", ls="--", label="linear ref")
    ], loc="upper left", fontsize=9)

    # ── Row 3: 控制类型标注条 ──
    ax_bar = fig.add_subplot(gs[3, :])
    ax_bar.set_xlim(-0.5, T + 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Timestep", fontsize=10)
    ax_bar.set_title("Control Type", fontsize=10)

    for i in range(T):
        c = "#2ecc71" if labels[i] == 2 else "#f39c12"
        ax_bar.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1,
                                       facecolor=c, lw=0))
    ax_bar.legend(handles=[
        Patch(facecolor="#f39c12", label="RL"),
        Patch(facecolor="#2ecc71", label="Human"),
    ], loc="upper right", fontsize=9, ncol=2)

    out = os.path.join(vis_dir, f"traj_{traj_idx:03d}.png")
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)


def _save_trajectory_video(
    traj_idx: int,
    side_imgs: np.ndarray,
    wrist_imgs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    n_rl: int,
    n_human: int,
    T: int,
    vis_dir: str,
    fps: int = 5,
    anomalies: list | None = None,
) -> None:
    """
    逐帧渲染评估轨迹视频，保存为 MP4。

    布局（每帧）：
      左：侧面相机当前帧（边框颜色 = 当前控制类型）
      中：腕部相机当前帧
      右上：进度预测曲线
      右下：控制类型标注条
    """
    COLOR_RL    = "#2980b9"
    COLOR_HUMAN = "#27ae60"
    COLOR_CUR   = "#e74c3c"

    t_axis    = np.arange(T + 1)
    true_prog = np.linspace(0, 1, T + 1)
    
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="pm_video_")

    for t in range(T + 1):
        label_t    = int(labels[t])
        ctrl_str   = "Human" if label_t == 2 else "RL"
        ctrl_color = COLOR_HUMAN if label_t == 2 else COLOR_RL

        fig = plt.figure(figsize=(18, 6), dpi=80, facecolor="white")
        gs  = GridSpec(
            2, 3,
            height_ratios=[5, 1.2],
            width_ratios=[3, 3, 5],
            left=0.04, right=0.98, top=0.88, bottom=0.05,
            hspace=0.15, wspace=0.08,
        )
        fig.suptitle(
            f"Traj {traj_idx}  t = {t:3d} / {T}   [{ctrl_str}]   "
            f"P(t) = {preds[t]:.4f}     RL={n_rl}  Human={n_human}",
            fontsize=12, color=ctrl_color, fontweight="bold",
        )

        # ── 侧面相机 ──
        ax_side = fig.add_subplot(gs[0, 0])
        ax_side.imshow(side_imgs[t])
        ax_side.set_title("Side Camera", fontsize=9)
        ax_side.axis("off")
        for sp in ax_side.spines.values():
            sp.set_visible(True)
            sp.set_color(ctrl_color)
            sp.set_linewidth(4)

        # ── 腕部相机 ──
        ax_wrist = fig.add_subplot(gs[0, 1])
        ax_wrist.imshow(wrist_imgs[t])
        ax_wrist.set_title("Wrist Camera", fontsize=9)
        ax_wrist.axis("off")
        for sp in ax_wrist.spines.values():
            sp.set_visible(True)
            sp.set_color(ctrl_color)
            sp.set_linewidth(4)

        # ── 进度曲线 ──
        ax_prog = fig.add_subplot(gs[0, 2])
        # 背景色块（RL/Human 分段）
        for i in range(T):
            c = "#e8f8f5" if labels[i] == 2 else "#fef9e7"
            ax_prog.axvspan(i - 0.5, i + 0.5, color=c, alpha=0.7, lw=0)
        # 异常段高亮（红色=回退，橙色=停滞）
        if anomalies:
            for seg in anomalies:
                c = "#e74c3c" if seg["type"] == "regression" else "#f39c12"
                ax_prog.axvspan(seg["start"] - 0.5, seg["end"] + 0.5,
                                color=c, alpha=0.20, zorder=2, lw=0)
        # 完整预测曲线（淡灰背景，供参考）
        ax_prog.plot(t_axis, preds, color="#cccccc", lw=1.5, zorder=1)
        # 线性进度参考线
        ax_prog.plot(t_axis, true_prog, "--", color=COLOR_CUR,
                     lw=1.0, alpha=0.35, zorder=1)
        # 已走历史点（按 RL/Human 染色）
        for i in range(t + 1):
            c = COLOR_HUMAN if labels[i] == 2 else COLOR_RL
            ax_prog.plot(i, preds[i], "o", color=c,
                         markersize=5, zorder=3, alpha=0.85)
        # 当前帧：大圆 + P 值标注
        ax_prog.plot(t, preds[t], "o",
                     color=ctrl_color, markersize=12,
                     markeredgecolor="white", markeredgewidth=2, zorder=6)
        offset_pts = 10 if preds[t] < 0.88 else -18
        ax_prog.annotate(
            f"P={preds[t]:.3f}",
            (t, preds[t]),
            xytext=(0, offset_pts),
            textcoords="offset points",
            fontsize=9, color=ctrl_color, fontweight="bold",
            ha="center", zorder=7,
        )
        # 当前时刻竖线
        ax_prog.axvline(t, color=COLOR_CUR, lw=1.5, ls="--",
                        alpha=0.7, zorder=5)
        ax_prog.set_xlim(-0.5, T + 0.5)
        ax_prog.set_ylim(-0.08, 1.15)
        ax_prog.set_xlabel("Timestep", fontsize=9)
        ax_prog.set_ylabel("P(o, s)", fontsize=9)
        ax_prog.set_title("Progress Prediction", fontsize=10)
        ax_prog.grid(alpha=0.2)
        _legend_handles = [
            Patch(facecolor=COLOR_RL,    label="RL"),
            Patch(facecolor=COLOR_HUMAN, label="Human"),
        ]
        if anomalies:
            if any(s["type"] == "regression" for s in anomalies):
                _legend_handles.append(Patch(facecolor="#e74c3c", alpha=0.3,
                                             label="Regress"))
            if any(s["type"] == "stagnation" for s in anomalies):
                _legend_handles.append(Patch(facecolor="#f39c12", alpha=0.3,
                                             label="Stagnate"))
        ax_prog.legend(handles=_legend_handles,
                       loc="upper left", fontsize=8, ncol=2)

        # ── 控制类型标注条（左+中列底部合并） ──
        ax_bar = fig.add_subplot(gs[1, :2])
        ax_bar.set_xlim(-0.5, T + 0.5)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Timestep", fontsize=8)
        ax_bar.set_title("Control Type", fontsize=8)
        for i in range(T):
            c = "#2ecc71" if labels[i] == 2 else "#f39c12"
            ax_bar.add_patch(
                Rectangle((i - 0.5, 0), 1, 1, facecolor=c, lw=0)
            )
        ax_bar.axvline(t, color=COLOR_CUR, lw=2, ls="--", zorder=5)
        ax_bar.legend(handles=[
            Patch(facecolor="#f39c12", label="RL"),
            Patch(facecolor="#2ecc71", label="Human"),
        ], loc="upper right", fontsize=8, ncol=2)

        # ── 当前 P 值文本（右列底部） ──
        ax_txt = fig.add_subplot(gs[1, 2])
        ax_txt.axis("off")
        # 检查当前帧是否处于异常段
        _anomaly_tag = ""
        _txt_color = ctrl_color
        _txt_bg = "#f0f0f0"
        if anomalies:
            for seg in anomalies:
                if seg["start"] <= t <= seg["end"]:
                    _atype = seg["type"]
                    _anomaly_tag = f"\n⚠ {_atype.upper()}"
                    _txt_color = "#e74c3c" if _atype == "regression" else "#f39c12"
                    _txt_bg = "#fde8e8" if _atype == "regression" else "#fef9e7"
                    break
        ax_txt.text(
            0.5, 0.5,
            f"t = {t}  [{ctrl_str}]  P = {preds[t]:.4f}{_anomaly_tag}",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=_txt_color,
            transform=ax_txt.transAxes,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=_txt_bg,
                      edgecolor=_txt_color, lw=2),
        )

        frame_path = os.path.join(tmp_dir, f"frame_{t:04d}.png")
        fig.savefig(frame_path, dpi=100, facecolor="white")
        plt.close(fig)

    out_mp4 = os.path.join(vis_dir, f"traj_{traj_idx:03d}.mp4")
    _ffmpeg_frames_to_video(tmp_dir, out_mp4, fps)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _ffmpeg_frames_to_video(frame_dir: str, output_path: str, fps: int):
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log(f"  [video] 已保存 MP4: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log(f"  [video] ffmpeg 失败 ({e})，帧目录: {frame_dir}")


# ──────────────────────────────────────────────────────────────────────
# 异常检测：基于进度值的滑动窗口检测
# ──────────────────────────────────────────────────────────────────────

def detect_anomalies(
    preds: np.ndarray,
    labels: np.ndarray,
    window_size: int = 5,
    delta_reg: float = 0.02,
    delta_stag: float = 0.01,
    detect_regression: bool = True,
    detect_stagnation: bool = True,
    recovery_k: int = 3,
) -> list[dict]:
    """
    基于进度预测值的 **锚点扩展 + 恢复确认** 异常检测。

    算法：
      1. 在每个 RL 帧 t 上，检查初始窗口 [t, t+W] 的 ΔP：
         - 回退: ΔP < -delta_reg
         - 停滞: |ΔP| < delta_stag
      2. 若触发异常，以 P_anchor = P[t] 为锚点，向后逐帧扩展 e：
         只要 P[e] < P_anchor，e 继续后移
      3. 恢复确认：需要连续 recovery_k 帧都 >= P_anchor 才算真正恢复。
         异常段终止于最后一个低于锚点的帧。
      4. 下一轮检测从异常段结束后开始。

    Human 段 (label==2) 不参与检测。

    Args:
        preds:              (T+1,) 进度预测值
        labels:             (T+1,) 控制类型标签 (1=RL, 2=Human)
        window_size:        初始触发窗口大小（步数）
        delta_reg:          回退检测阈值（ΔP < -delta_reg 判为回退）
        delta_stag:         停滞检测阈值（|ΔP| < delta_stag 判为停滞）
        detect_regression:  是否启用回退检测
        detect_stagnation:  是否启用停滞检测
        recovery_k:         恢复确认窗口（连续 K 帧 >= P_anchor 才算恢复）

    Returns:
        异常段列表: [{start, end, type, delta_p, duration}, ...]
    """
    if not detect_regression and not detect_stagnation:
        return []

    N = len(preds)
    if window_size < 1 or N <= window_size:
        return []

    segments: list[dict] = []
    t = 0

    while t < N - window_size:
        # 跳过 Human 帧
        if labels[t] == 2:
            t += 1
            continue

        # 检查初始窗口 [t, t+W] 内是否全部为 RL
        window_labels = labels[t : t + window_size + 1]
        if np.any(window_labels == 2):
            t += 1
            continue

        delta_p = preds[t + window_size] - preds[t]

        # 判断异常类型
        anom_type = None
        if detect_regression and delta_p < -delta_reg:
            anom_type = "regression"
        elif detect_stagnation and abs(delta_p) < delta_stag:
            anom_type = "stagnation"

        if anom_type is None:
            t += 1
            continue

        # ── 触发异常：锚点扩展 ──
        anchor_t = t
        p_anchor = preds[anchor_t]
        last_bad = t + window_size  # 至少到初始窗口末端

        e = t + window_size + 1
        consec_ok = 0  # 连续 >= P_anchor 的帧计数

        while e < N:
            # 遇到 Human 帧：强制终止异常段
            if labels[e] == 2:
                break
            if preds[e] >= p_anchor:
                consec_ok += 1
                if consec_ok >= recovery_k:
                    break  # 恢复确认通过
            else:
                consec_ok = 0
                last_bad = e  # 更新最后一个异常帧
            e += 1

        segments.append({
            "start":    anchor_t,
            "end":      last_bad,
            "type":     anom_type,
            "delta_p":  float(preds[last_bad] - preds[anchor_t]),
            "duration": last_bad - anchor_t + 1,
        })

        # 下一轮从异常段结束后开始
        t = last_bad + 1

    return segments


# ──────────────────────────────────────────────────────────────────────
# 命令行参数
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练进度估计模型 P(o, s) → [0, 1]"
    )
    # 路径
    p.add_argument("--demo_dir",    default="segmentation/liuzeyi_data/insert_block/demo_data/20260228")
    p.add_argument("--buffer_dir",  default="segmentation/liuzeyi_data/insert_block/buffer/buffer")
    p.add_argument("--output_dir",  default="VF_training/runs/progress_model")
    p.add_argument("--cache_path",  default="VF_training/runs/progress_model/demo_features.pt",
                   help="Phase1 特征缓存路径；若文件已存在则直接加载")

    # 训练超参
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--gamma",         type=float, default=0.95)
    p.add_argument("--log_interval",  type=int,   default=20)
    p.add_argument("--seed",          type=int,   default=42)

    # 损失权重
    p.add_argument("--lambda_prog", type=float, default=1.0,  help="进度损失权重")
    p.add_argument("--lambda_mc",   type=float, default=0.1,  help="MC 损失权重")
    p.add_argument("--lambda_td",   type=float, default=0.1,  help="TD 损失权重")
    p.add_argument("--mc_last_n",   type=int,   default=5,
                   help="MC 损失仅作用于每条轨迹最后 N 步（0=全部步都参与）")

    # Phase 1
    p.add_argument("--phase1_batch_size", type=int, default=32,
                   help="ResNet-18 特征提取时的 batch size")

    # 评估
    p.add_argument("--n_eval_trajs", type=int, default=50,
                   help="评估的成功轨迹数量")
    p.add_argument("--min_rl_ratio", type=float, default=0.1,
                   help="最小 RL 占比阈値")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--video_fps",    type=int,   default=5,
                   help="评估视频帧率（MP4，默认5fps）")

    # 异常检测
    p.add_argument("--anomaly_window",     type=int,   default=4,
                   help="异常检测滑动窗口大小（步数）")
    p.add_argument("--delta_reg",          type=float, default=0.045,
                   help="回退检测阈值（ΔP < -delta_reg 判为回退）")
    p.add_argument("--delta_stag",         type=float, default=0.001,
                   help="停滞检测阈值（|ΔP| < delta_stag 判为停滞）")
    p.add_argument("--detect_regression",  type=int,   default=1,
                   help="是否启用回退检测（0=关, 1=开）")
    p.add_argument("--detect_stagnation",  type=int,   default=0,
                   help="是否启用停滞检测（0=关, 1=开）")
    p.add_argument("--recovery_k",         type=int,   default=3,
                   help="恢复确认窗口：连续 K 帧 >= P_anchor 才算恢复")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────

def _make_experiment_dir(base_dir: str) -> str:
    """在 base_dir 下创建 exp_NNN_YYYYMMDD_HHMM 格式的实验目录。"""
    from datetime import datetime
    os.makedirs(base_dir, exist_ok=True)
    # 扫描已有编号
    existing = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp_")]
    max_id = 0
    for d in existing:
        try:
            max_id = max(max_id, int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"exp_{max_id + 1:03d}_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def main() -> None:
    args = parse_args()

    # 在 output_dir 下自动建独立实验文件夹
    args.output_dir = _make_experiment_dir(args.output_dir)
    # cache 放到实验目录内（但优先复用已有缓存）
    if args.cache_path == "VF_training/runs/progress_model/demo_features.pt":
        args.cache_path = os.path.join(args.output_dir, "demo_features.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    log(f"[main] 实验目录: {args.output_dir}")

    log("=" * 60)
    log("Step 1 & 2: 特征提取 + 训练进度模型")
    log("=" * 60)
    model = train(args)

    log("\n" + "=" * 60)
    log("Step 3: 评估 buffer 成功轨迹并可视化")
    log("=" * 60)

    # 加载 state 归一化统计
    stats = torch.load(os.path.join(args.output_dir, "state_stats.pt"),
                       map_location="cpu")
    state_mean, state_std = stats["mean"], stats["std"]

    # 加载最佳模型权重
    best_ckpt = os.path.join(args.output_dir, "progress_model_best.pt")
    model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    log(f"[eval] 加载最佳模型: {best_ckpt}")

    # 重建特征提取器用于 buffer 推理
    extractor = ResNetExtractor()

    evaluate_and_visualize(
        model=model,
        buffer_dir=args.buffer_dir,
        output_dir=args.output_dir,
        state_mean=state_mean,
        state_std=state_std,
        extractor=extractor,
        args=args,
    )

    log("\n" + "=" * 60)
    log(f"完成！结果保存在: {args.output_dir}/")
    log(f"  训练曲线:       {args.output_dir}/training_curves.png")
    log(f"  Demo 验证曲线:  {args.output_dir}/demo_progress_curves.png")
    log(f"  Buffer 评估:    {args.output_dir}/eval_visuals/")
    log("=" * 60)


if __name__ == "__main__":
    main()
