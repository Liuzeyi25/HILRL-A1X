"""
模拟 online action 从干预 → trajectory → reorganize_chunks → data_store → replay_buffer 的完整流程
"""
import numpy as np

# ── 复制自 train_conrft_octo.py ──────────────────────────────────────────────
def reorganize_transitions_to_chunks(transitions, chunk_size):
    if len(transitions) == 0:
        return []
    chunked_transitions = []
    first_action = transitions[0]['actions']
    if first_action.ndim == 1:
        action_dim = first_action.shape[0]
    else:
        action_dim = first_action.shape[1]

    for i in range(len(transitions)):
        trans = transitions[i].copy()
        action_chunk = []
        for j in range(chunk_size):
            if i + j < len(transitions):
                action = transitions[i + j]['actions']
                if action.ndim == 1:
                    action_chunk.append(action)
            else:
                action_chunk.append(np.zeros(action_dim, dtype=first_action.dtype))
        trans['actions'] = np.array(action_chunk)
        chunked_transitions.append(trans)
    return chunked_transitions


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── 参数设置 ─────────────────────────────────────────────────────────────────
np.random.seed(42)
ACTION_DIM   = 7    # 末端执行器 6DoF + gripper
CHUNK_SIZE   = 4    # action_chunk_size
EPISODE_LEN  = 6    # episode 长度（模拟6步）
BATCH_SIZE   = 4    # learner 取出的 batch 大小

# ── ① 模拟策略输出的 chunk action（正常步骤）─────────────────────────────────
sep("① 策略输出的原始 chunk actions（未发生干预）")
policy_actions = np.random.uniform(-1, 1, (EPISODE_LEN, CHUNK_SIZE, ACTION_DIM)).astype(np.float32)
for i, a in enumerate(policy_actions):
    print(f"  step {i}: shape={a.shape}  a[0]={np.round(a[0], 4)}")

# ── ② 模拟 info["intervene_action_eef"] 覆盖（第 2、3 步发生干预）────────────
sep("② info['intervene_action_eef'] 覆盖（单步，第 2~3 步干预）")
intervene_steps = {2, 3}
intvn_actions_raw = {
    2: np.array([0.1, -0.2,  0.3, 0.05, -0.1,  0.2,  1.0], dtype=np.float32),
    3: np.array([0.15, -0.25, 0.35, 0.08, -0.12, 0.22, 1.0], dtype=np.float32),
}
trajectory = []
for i in range(EPISODE_LEN):
    if i in intervene_steps:
        actions = intvn_actions_raw[i]          # 单步干预
        intervened = True
        print(f"  step {i}: [INTERVENED] shape={actions.shape}  values={np.round(actions, 4)}")
    else:
        actions = policy_actions[i]             # 策略 chunk
        intervened = False

    transition = dict(
        observations=np.zeros((84, 84, 3)),     # 占位图像
        actions=actions,
        next_observations=np.zeros((84, 84, 3)),
        rewards=float(i == EPISODE_LEN - 1),    # 最后一步给 reward
        masks=1.0,
        dones=float(i == EPISODE_LEN - 1),
        intervened=intervened,
    )
    trajectory.append(transition)

# ── ③ reorganize_transitions_to_chunks ───────────────────────────────────────
sep("③ reorganize_transitions_to_chunks 前后对比")
print("  [before]")
for i, t in enumerate(trajectory):
    a = t['actions']
    print(f"  step {i}: shape={a.shape}  ndim={a.ndim}  intervened={t['intervened']}")

has_single_step = any(t['actions'].ndim == 1 for t in trajectory)
print(f"\n  has_single_step = {has_single_step}  → {'需要重组' if has_single_step else '无需重组'}")

if has_single_step:
    trajectory = reorganize_transitions_to_chunks(trajectory, CHUNK_SIZE)

print("\n  [after reorganize]")
for i, t in enumerate(trajectory):
    a = t['actions']
    print(f"  step {i}: shape={a.shape}  a[0]={np.round(a[0], 4)}  intervened={t['intervened']}")

# ── ④ 模拟 data_store.insert → replay_buffer ─────────────────────────────────
sep("④ 写入 data_store（pre-insert 打印）")
replay_buffer = []
for t in trajectory:
    a = t['actions']
    print(
        f"  [pre-insert] shape={a.shape}  "
        f"values={np.round(a, 4)}  "
        f"intervened={t['intervened']}"
    )
    replay_buffer.append(t)

# ── ⑤ 模拟 replay_buffer.get_iterator → next(replay_iterator) ───────────────
sep("⑤ Learner next(replay_iterator) 取出 batch")
indices = np.random.choice(len(replay_buffer), size=BATCH_SIZE, replace=True)
batch_actions = np.stack([replay_buffer[i]['actions'] for i in indices], axis=0)
print(f"  batch['actions'] shape = {batch_actions.shape}")
print(f"  min={batch_actions.min():.4f}  max={batch_actions.max():.4f}  mean={batch_actions.mean():.4f}")
print(f"  sample[0] = {np.round(batch_actions[0], 4)}")

# ── ⑥ 汇总：哪些步骤 action 发生了变化 ──────────────────────────────────────
sep("⑥ 汇总：action 变化点")
print("""
  步骤 | 变化原因                          | 形状变化
  ─────┼───────────────────────────────────┼────────────────────────────
   ①  | 策略输出 chunk action              | (chunk, dim)  无变化
   ②  | intervene_action_eef 覆盖          | (chunk,dim) → (dim,) 降维！
   ③  | reorganize_transitions_to_chunks  | (dim,) → (chunk,dim) 重组
       |   · 干预步的 action[0] = 干预动作  |
       |   · action[1..] = 后续步的动作    |
       |   · 末尾不足补 0                  |
   ④  | data_store.insert                 | 无变化，原样存入
   ⑤  | next(replay_iterator) → batch     | stack → (batch,chunk,dim)
""")
