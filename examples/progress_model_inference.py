"""
progress_model_inference.py
===========================
Progress Model 推理模块（供 train_rlpd_hil.py 的 actor 端调用）。

提供：
  - ResNetExtractor       : 冻结的 ResNet-18 特征提取器（512 维输出）
  - ProgressHead          : MLP 进度头（输出 ∈ [0, 1]）
  - ProgressModelRunner   : 封装加载 + 单次 episode 推理的类
  - detect_anomalies      : 锚点扩展 + 恢复确认 的滑动窗口异常检测

模型输入（均为相对轨迹起点 t=0 的差值）：
  d_side  = side_feat_t  - side_feat_0   (512,)  ResNet-18 特征差
  d_wrist = wrist_feat_t - wrist_feat_0  (512,)
  d_state = (state_t - state_0 - mean) / std     (STATE_DIM,) 归一化状态差

模型产出文件（由 train_progress_model.py 训练后生成）：
  progress_model_best.pt  —— ProgressHead 权重
  state_stats.pt          —— {"mean": (7,), "std": (7,)} 归一化统计
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ──────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────

STATE_DIM = 7

_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────
# 图像工具
# ──────────────────────────────────────────────────────────────────────

def obs_img_to_pil(arr: np.ndarray) -> Image.Image:
    """
    将来自 replay buffer 观测的图像数组转换为 PIL Image。

    处理以下格式：
      (obs_horizon, H, W, 3) — ChunkingWrapper 堆叠帧，取最后帧（最新）
      (H, W, 3)              — 单帧
      其他维度会尽量 squeeze 后处理

    注意：始终取最后一帧（index = -1），对 obs_horizon=1 和 obs_horizon=2
    均能正确得到当前帧。
    """
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[-1]   # (obs_horizon, H, W, 3) → 取最新帧 (H, W, 3)
    elif arr.ndim == 2:
        # 灰度 (H, W) → 重复三通道
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr.astype(np.uint8))


def obs_state_to_vec(state_obs: np.ndarray) -> np.ndarray:
    """
    将来自 replay buffer 的 state 观测转换为 (STATE_DIM,) float32 向量。

    处理以下格式：
      (obs_horizon, STATE_DIM) — 取最后帧（最新）
      (STATE_DIM,)             — 直接使用
    """
    arr = np.asarray(state_obs, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[-1]       # 取最新帧
    return arr.flatten()[:STATE_DIM]


# ──────────────────────────────────────────────────────────────────────
# 模型结构（与 train_progress_model.py 保持完全一致）
# ──────────────────────────────────────────────────────────────────────

class ResNetExtractor(nn.Module):
    """截断到全局平均池化层的 ResNet-18，输出 512 维特征向量。"""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 去掉最后的 fc 层，保留到 avgpool：输出 (B, 512, 1, 1)
        self.feature = nn.Sequential(*list(backbone.children())[:-1])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, 512)"""
        return self.feature(x).flatten(1)


class ProgressHead(nn.Module):
    """
    MLP 进度头。

    输入（均为差值）：
      d_side  : (B, 512)  side_feat_t  - side_feat_0
      d_wrist : (B, 512)  wrist_feat_t - wrist_feat_0
      d_state : (B, 7)    state_t      - state_0（归一化后）

    输出：(B,) 进度值 ∈ [0, 1]

    结构：
      state_encoder : STATE_DIM → state_enc_dim（64）
      fusion_net    : 1024 + 64 → hidden → hidden//2 → 1 → Sigmoid
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        state_dim: int = STATE_DIM,
        hidden: int = 256,
        state_enc_dim: int = 64,
    ):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_enc_dim),
            nn.LayerNorm(state_enc_dim),
            nn.ReLU(),
        )
        in_dim = feat_dim + state_enc_dim   # 1024 + 64 = 1088
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
        d_side: torch.Tensor,   # (B, 512)
        d_wrist: torch.Tensor,  # (B, 512)
        d_state: torch.Tensor,  # (B, STATE_DIM)
    ) -> torch.Tensor:
        """→ (B,) 进度值"""
        s_enc = self.state_encoder(d_state)                        # (B, 64)
        x = torch.cat([d_side, d_wrist, s_enc], dim=-1)            # (B, 1088)
        return self.fusion_net(x).squeeze(-1)                       # (B,)


# ──────────────────────────────────────────────────────────────────────
# 推理运行器
# ──────────────────────────────────────────────────────────────────────

class ProgressModelRunner:
    """
    加载已训练的 Progress Model 并对 episode_buffer 进行在线推理。

    典型用法（actor 启动时初始化一次，每个 episode 结束时调用 infer_episode）：

        runner = ProgressModelRunner(
            model_path = "VF_training/runs/progress_model/exp_001/progress_model_best.pt",
            stats_path = "VF_training/runs/progress_model/exp_001/state_stats.pt",
            side_key   = "side_policy_256",
            wrist_key  = "wrist_1",
            hidden_dim = 128,
            device     = "cpu",
        )

        # 在 episode 结束后：
        preds, labels = runner.infer_episode(episode_buffer)
    """

    def __init__(
        self,
        model_path: str,
        stats_path: str,
        side_key: str = "side_policy_256",
        wrist_key: str = "wrist_1",
        hidden_dim: int = 128,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.side_key   = side_key
        self.wrist_key  = wrist_key
        self.device     = device
        self.batch_size = batch_size

        # 加载 ResNet-18 特征提取器（冻结）
        self.extractor = ResNetExtractor().to(device).eval()

        # 加载 MLP head（hidden_dim 需与训练时一致）
        self.model = ProgressHead(
            feat_dim=1024, state_dim=STATE_DIM, hidden=hidden_dim
        ).to(device).eval()
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        # 加载状态差值的归一化统计
        stats = torch.load(stats_path, map_location="cpu")
        self.state_mean: torch.Tensor = stats["mean"]   # (STATE_DIM,)
        self.state_std:  torch.Tensor = stats["std"]    # (STATE_DIM,)

        print(
            f"[ProgressModelRunner] 加载完成  "
            f"model={model_path}  device={device}",
            flush=True,
        )

    def _extract_pils(self, pils: List[Image.Image]) -> torch.Tensor:
        """批量提取 ResNet-18 特征，返回 (N, 512) CPU 张量。"""
        feats = []
        for start in range(0, len(pils), self.batch_size):
            batch = pils[start : start + self.batch_size]
            tensors = torch.stack([_resnet_transform(p) for p in batch]).to(self.device)
            with torch.no_grad():
                f = self.extractor(tensors).cpu()
            feats.append(f)
        return torch.cat(feats, dim=0)  # (N, 512)

    @torch.no_grad()
    def infer_episode(
        self,
        episode_buffer: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对一条完整 episode 的所有帧推理进度值。

        Args:
            episode_buffer: T 条 transition dict，每条包含：
                              observations       - 当前帧观测（含图像和 state）
                              next_observations  - 下一帧观测
                              _was_intervened    - bool，该步是否为人类干预步

        Returns:
            preds:  np.ndarray, shape (T+1,), float32
                    每帧的预测进度值 ∈ [0, 1]。
                    索引 0..T-1 对应 observations，索引 T 对应最终 next_observations。
            labels: np.ndarray, shape (T+1,), int32
                    控制类型标签：1 = RL 步，2 = Human 干预步。
                    最终帧（index T）继承前帧标签。
        """
        T = len(episode_buffer)
        if T == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        # ── 收集 T+1 帧观测（T 个当前帧 + 1 个最终 next 帧）──
        obs_list = [ep["observations"] for ep in episode_buffer]
        obs_list.append(episode_buffer[-1]["next_observations"])

        # 提取图像 PIL 列表
        side_pils  = [obs_img_to_pil(np.asarray(obs[self.side_key]))  for obs in obs_list]
        wrist_pils = [obs_img_to_pil(np.asarray(obs[self.wrist_key])) for obs in obs_list]

        # 提取状态向量 (T+1, STATE_DIM)
        states = np.stack(
            [obs_state_to_vec(obs["state"]) for obs in obs_list], axis=0
        )  # (T+1, STATE_DIM)

        # ── 批量提取 ResNet-18 特征 ──
        side_feats  = self._extract_pils(side_pils)    # (T+1, 512)
        wrist_feats = self._extract_pils(wrist_pils)   # (T+1, 512)

        # ── 计算相对 t=0 的差值，归一化状态差值 ──
        sf0 = side_feats[0:1]    # (1, 512)  t=0 帧侧面特征
        wf0 = wrist_feats[0:1]   # (1, 512)  t=0 帧腕部特征

        d_side  = (side_feats  - sf0).to(self.device)   # (T+1, 512)
        d_wrist = (wrist_feats - wf0).to(self.device)   # (T+1, 512)

        states_t = torch.tensor(states, dtype=torch.float32)
        # 归一化：(s_t - s_0 - mean) / std
        d_state  = (states_t - states_t[0:1] - self.state_mean) / self.state_std
        d_state  = d_state.to(self.device)              # (T+1, STATE_DIM)

        # ── 推理 ──
        preds = self.model(d_side, d_wrist, d_state).cpu().numpy()   # (T+1,)

        # ── 构造控制类型标签（1=RL, 2=Human）──
        labels = np.ones(T + 1, dtype=np.int32)
        for i, ep in enumerate(episode_buffer):
            if ep.get("_was_intervened", False):
                labels[i] = 2
        labels[T] = labels[T - 1]   # 最终帧继承前帧标签

        return preds, labels


# ──────────────────────────────────────────────────────────────────────
# 异常检测（与 train_progress_model.py 保持完全一致）
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
) -> List[Dict]:
    """
    基于进度预测值的 **锚点扩展 + 恢复确认** 异常检测。

    算法：
      1. 在每个 RL 帧 t 上，检查初始窗口 [t, t+W] 的 ΔP：
           回退：ΔP < -delta_reg
           停滞：|ΔP| < delta_stag
      2. 若触发异常，以 P_anchor = P[t] 为锚点，向后逐帧扩展 e：
           只要 P[e] < P_anchor，e 继续后移
      3. 恢复确认：需要连续 recovery_k 帧都 >= P_anchor 才算真正恢复。
           异常段终止于最后一个低于锚点的帧。
      4. 下一轮检测从异常段结束后开始。

    Human 段 (label==2) 不参与检测，遇到 Human 帧强制终止异常段扩展。

    Args:
        preds:             (T+1,) float32  进度预测值
        labels:            (T+1,) int      控制类型（1=RL, 2=Human）
        window_size:       初始触发窗口大小（步数）
        delta_reg:         回退阈值（ΔP < -delta_reg 则触发）
        delta_stag:        停滞阈值（|ΔP| < delta_stag 则触发）
        detect_regression: 是否启用回退检测
        detect_stagnation: 是否启用停滞检测
        recovery_k:        恢复确认窗口（连续 K 帧 >= P_anchor 才算恢复）

    Returns:
        异常段列表，每条为 dict：
          {
            "start":    int    — 异常起始帧索引（episode_buffer 中）
            "end":      int    — 异常终止帧索引（最后一个低于锚点的帧）
            "type":     str    — "regression" 或 "stagnation"
            "delta_p":  float  — P[end] - P[start]
            "duration": int    — end - start + 1
          }
    """
    if not detect_regression and not detect_stagnation:
        return []

    N = len(preds)
    if window_size < 1 or N <= window_size:
        return []

    segments: List[Dict] = []
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

        # ── 触发异常：以 P[t] 为锚点向后扩展 ──────────────────────────────
        anchor_t = t
        p_anchor = preds[anchor_t]
        last_bad = t + window_size   # 至少覆盖初始窗口末端

        e = t + window_size + 1
        consec_ok = 0   # 连续 >= P_anchor 的帧计数

        while e < N:
            # Human 帧：强制终止
            if labels[e] == 2:
                break
            if preds[e] >= p_anchor:
                consec_ok += 1
                if consec_ok >= recovery_k:
                    break   # 恢复确认通过
            else:
                consec_ok = 0
                last_bad = e   # 更新最后一个异常帧
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
