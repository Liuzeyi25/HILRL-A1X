#!/usr/bin/env python3
"""
HG-DAgger 训练脚本（Human-Guided DAgger）

两阶段训练：
  Phase 1 (离线 BC 预热)：在 demo 数据上做行为克隆，训练 pretrain_steps 步
  Phase 2 (在线 DAgger) ：部署策略，人类随时接管，仅用人类干预数据做 BC 更新

架构与 HIL-SERL (train_rlpd.py) 完全一致：
  - 相同的 SAC 网络结构（actor 编码器 + policy head + 可选 grasp_critic）
  - 仅更新 actor 参数（无 critic、无 Q 函数、无 temperature 更新）
  - BC 损失：pre-tanh 空间 MSE，等价于行为克隆但比 log_prob 更稳定
  - 双进程：Learner（训练）+ Actor（采样），使用 agentlace 通信

用法：
  # 第一步：采集 demo 数据（与 HIL-SERL 共用脚本）
  python examples/record_demos_octo_manual_new.py --exp_name a1x_pick_banana ...

  # 第二步：合并 demo 数据
  python examples/merge_trajectories.py experiments/a1x_pick_banana/demo_data/<日期> \
      experiments/a1x_pick_banana/demo_data/<日期>/traj_merged.pkl

  # 第三步：启动 Learner
  python examples/train_hgdagger.py --exp_name a1x_pick_banana --learner \
      --demo_path ./experiments/a1x_pick_banana/demo_data/<日期>/traj_merged.pkl \
      --checkpoint_path ./hgdagger/run1 \
      --pretrain_steps 5000 --mix_demo_ratio 0.5

  # 第四步（等 Learner 打印 "sent initial network to actor" 后启动）：
  python examples/train_hgdagger.py --exp_name a1x_pick_banana --actor \
      --checkpoint_path ./hgdagger/run1
"""

import glob
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax import config as flax_config
from flax.training import checkpoints

flax_config.update("flax_use_orbax_checkpointing", False)

import os
import pickle as pkl
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

# ---- 与 train_rlpd.py 相同的基础 flags ----
flags.DEFINE_string("exp_name", None, "实验名，对应 CONFIG_MAPPING 中的 key。")
flags.DEFINE_integer("seed", 42, "随机种子。")
flags.DEFINE_boolean("learner", False, "是否以 Learner 模式运行。")
flags.DEFINE_boolean("actor", False, "是否以 Actor 模式运行。")
flags.DEFINE_string("ip", "localhost", "Learner 的 IP 地址（Actor 侧使用）。")
flags.DEFINE_multi_string("demo_path", None, "离线 demo 数据路径（可多次指定）。")
flags.DEFINE_string("checkpoint_path", None, "checkpoint 保存/加载路径。")
flags.DEFINE_integer("eval_checkpoint_step", 0, "评估时加载的 checkpoint 步数（0=不评估）。")
flags.DEFINE_integer("eval_n_trajs", 0, "评估轨迹数量。")
flags.DEFINE_boolean("save_video", False, "是否保存视频。")
flags.DEFINE_boolean("debug", False, "调试模式（关闭 WandB 日志）。")
flags.DEFINE_string("run_tag", "", "WandB run 名称后缀（可选）。")

# ---- HG-DAgger 专用 flags ----
flags.DEFINE_integer(
    "pretrain_steps",
    5000,
    "Phase 1 离线 BC 预热步数。"
    "如果 checkpoint_path 已有 step >= pretrain_steps 的存档，自动跳过。"
    "设为 0 直接跳过 Phase 1 进入在线 DAgger。",
)
flags.DEFINE_float(
    "mix_demo_ratio",
    0.5,
    "在线训练阶段每个 batch 中离线 demo 数据的比例。\n"
    "  0.0 = 纯在线干预数据（标准 DAgger）\n"
    "  0.5 = 50/50 混合（推荐，防止遗忘初始行为）\n"
    "取值范围 [0.0, 1.0)。",
)

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


# ===========================================================================
# BC 更新步（核心）
# ===========================================================================

@partial(jax.jit, static_argnames=("pmap_axis", "has_grasp_critic"))
def bc_update_step(agent, batch, *, pmap_axis=None, has_grasp_critic=False):
    """
    纯行为克隆更新：更新 actor（EEF）和可选的 grasp_critic（gripper），
    不更新 critic / temperature。

    损失函数：
      EEF（连续，前 D_policy 维）：pre-tanh 空间 MSE
          L_eef = E[ || μ_θ(s) - atanh(clip(a_demo[:D])) ||² ]

      Gripper（离散，最后 1 维，仅 has_grasp_critic=True 时）：交叉熵
          L_gripper = CrossEntropy(Q_grasp(s), demo_gripper_class)
          demo_gripper_class = round(a_demo[-1]) + 1  →  {0, 1, 2}（与训练时编码一致）

    自动适配：
      - SACAgent（fixed-gripper）：has_grasp_critic=False，只做 EEF BC
      - SACAgentHybridSingleArm（learned-gripper）：has_grasp_critic=True，
        同时做 EEF BC + gripper 交叉熵
    """

    def _bc_actor(params, rng):
        rng, policy_rng = jax.random.split(rng)

        # 前向传播：获取 EEF 动作分布（梯度通过 params 传播）
        dist = agent.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params, train=True
        )

        # mu_theta: pre-tanh 空间策略均值，shape=(B, D_policy)，一般 D_policy=6（EEF）
        mu_theta = dist.distribution.loc
        D_policy = mu_theta.shape[-1]

        # EEF 部分：只取前 D_policy 维（自动忽略 gripper 维）
        demo_eef = batch["actions"][..., :D_policy]  # (B, D_policy)

        # 映射到 pre-tanh 空间（clip 保证 atanh 有界，±1 处 atanh→±∞）
        clip_eps = 1e-2
        a_clipped = jnp.clip(demo_eef, -1.0 + clip_eps, 1.0 - clip_eps)
        u_demo = jnp.arctanh(a_clipped)  # (B, D_policy)，∈ [-2.65, 2.65]

        eef_bc_loss = jnp.mean((mu_theta - u_demo) ** 2)

        # 监控指标（stop_gradient）
        log_prob = jax.lax.stop_gradient(dist.log_prob(demo_eef))
        action_l2 = jnp.mean(
            jnp.linalg.norm(jnp.tanh(mu_theta) - demo_eef, axis=-1)
        )

        return eef_bc_loss, {
            "eef_bc_loss": eef_bc_loss,
            "bc_log_prob": jnp.mean(log_prob),
            "eef_action_l2": action_l2,
        }

    def _bc_grasp_critic(params, rng):
        """
        Gripper BC：用交叉熵模仿 demo 的离散 gripper 动作。

        编码约定（与 grasp_critic_loss_fn 完全一致）：
          env 动作空间中 gripper ∈ {-1, 0, 1}（浮点），
          存入 buffer 时为浮点，训练时 round + 1 映射到类别 {0, 1, 2}。
        """
        # 从 batch 动作最后一维提取 gripper
        demo_gripper_float = batch["actions"][..., -1]  # (B,)
        # 映射到离散类别：{-1→0, 0→1, 1→2}
        demo_gripper_class = jnp.round(demo_gripper_float).astype(jnp.int32) + 1  # (B,)

        # 前向传播 grasp_critic（梯度通过 params 传播）
        rng, gc_rng = jax.random.split(rng)
        grasp_q_values = agent.forward_grasp_critic(
            batch["observations"], rng=gc_rng, grad_params=params, train=True
        )  # (B, 3)

        # 交叉熵：log_softmax + one_hot 选取
        log_probs = jax.nn.log_softmax(grasp_q_values, axis=-1)  # (B, 3)
        # 用 one-hot 索引提取对应类别的 log_prob
        one_hot = jax.nn.one_hot(demo_gripper_class, num_classes=3)  # (B, 3)
        gripper_bc_loss = -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))

        # 监控：gripper 预测准确率（argmax == demo_class）
        predicted_class = jnp.argmax(grasp_q_values, axis=-1)  # (B,)
        gripper_accuracy = jnp.mean(predicted_class == demo_gripper_class)

        return gripper_bc_loss, {
            "gripper_bc_loss": gripper_bc_loss,
            "gripper_accuracy": gripper_accuracy,
        }

    # 构建各网络的 loss_fn：actor 和（可选）grasp_critic 做 BC，其余置 0
    _zero = lambda p, r: (0.0, {})
    std_loss_fns = agent.loss_fns(batch)

    bc_loss_fns = {}
    for k in std_loss_fns:
        if k == "actor":
            bc_loss_fns[k] = _bc_actor
        elif k == "grasp_critic" and has_grasp_critic:
            bc_loss_fns[k] = _bc_grasp_critic
        else:
            bc_loss_fns[k] = _zero

    new_state, info = agent.state.apply_loss_fns(
        bc_loss_fns, pmap_axis=pmap_axis, has_aux=True
    )
    new_state = new_state.replace(rng=jax.random.split(agent.state.rng)[0])
    return agent.replace(state=new_state), info


# ===========================================================================
# Actor 循环
# ===========================================================================

def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    Actor 循环，与 train_rlpd.py 完全一致。

    数据流：
      - 所有 transition → data_store（发送给 Learner 的 replay_buffer，
        HG-DAgger 中 Learner 不用此数据训练，仅保存到磁盘用于分析）
      - 人类干预的 transition → intvn_data_store（Learner 用此做 BC 训练）

    干预检测逻辑：
      - info["intervene_action_eef"] 存在 → 当前帧为人类干预
      - 干预动作覆盖策略动作并写入 intvn_data_store
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key,
                )
                actions = np.asarray(jax.device_get(actions))
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)
                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return

    # 从磁盘 buffer 恢复 start_step（续训用）
    _buf_dir = os.path.join(FLAGS.checkpoint_path, "buffer") if FLAGS.checkpoint_path else None
    _buf_files = glob.glob(os.path.join(_buf_dir, "*.pkl")) if _buf_dir and os.path.exists(_buf_dir) else []
    start_step = (
        int(os.path.basename(natsorted(_buf_files)[-1])[12:-4]) + 1
        if _buf_files
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=10000,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []        # 所有 transition（存磁盘，不用于训练）
    intvn_transitions = []  # 人类干预 transition（存磁盘 + 送 Learner 做 BC）

    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    episode_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            if "intervene_action_eef" in info:
                # 人类干预：用人类动作覆盖策略动作
                actions = info.pop("intervene_action_eef")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
                # 策略动作：将 action_scale==0 的维度置 0，保持与环境执行一致
                _scale = np.asarray(config.action_scale) if hasattr(config, "action_scale") else None
                if _scale is not None and _scale.ndim > 0:
                    _zero_mask = _scale == 0
                    if _zero_mask.any():
                        actions = actions.copy()
                        actions[: len(_zero_mask)][_zero_mask[: len(actions)]] = 0.0

            running_return += reward
            episode_steps += 1

            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                label=2 if already_intervened else 1,
            )
            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]

            # 所有 transition 都发给 data_store（Learner 侧 replay_buffer）
            data_store.insert(transition)
            transitions.append(transition)

            # 人类干预 transition 额外发给 intvn_data_store（Learner 侧 intvn_buffer）
            if already_intervened:
                intvn_data_store.insert(transition)
                intvn_transitions.append(transition)

            obs = next_obs

            if done or truncated:
                trajectory_length = episode_steps
                human_intervention_ratio = intervention_steps / max(trajectory_length, 1)
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["intervention_rate"] = human_intervention_ratio
                info["episode"]["human_intervention_step_ratio"] = human_intervention_ratio
                info["episode"]["trajectory_length"] = trajectory_length
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(
                    f"return={running_return:.2f} intvn={intervention_steps}/{trajectory_length}"
                )
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                episode_steps = 0
                already_intervened = False
                client.update()

                # 将 transition 批量写入磁盘（episode 结束时）
                if config.buffer_period > 0 and FLAGS.checkpoint_path:
                    buf_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                    intvn_buf_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                    os.makedirs(buf_path, exist_ok=True)
                    os.makedirs(intvn_buf_path, exist_ok=True)
                    if transitions:
                        with open(os.path.join(buf_path, f"transitions_{step}.pkl"), "wb") as f:
                            pkl.dump(transitions, f)
                        transitions = []
                    if intvn_transitions:
                        with open(os.path.join(intvn_buf_path, f"transitions_{step}.pkl"), "wb") as f:
                            pkl.dump(intvn_transitions, f)
                        intvn_transitions = []

                obs, _ = env.reset()

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


# ===========================================================================
# Learner 循环
# ===========================================================================

def learner(rng, agent, intvn_buffer, demo_buffer, env_obs_space, env_act_space,
            include_grasp_penalty, has_grasp_critic=False, wandb_logger=None):
    """
    Learner 循环（HG-DAgger）。

    Phase 1 - 离线 BC 预热（pretrain_steps > 0 且尚未完成时）：
      在 demo_buffer 上做 BC，训练 pretrain_steps 步后保存 checkpoint。
      如果 checkpoint_path 中已有 step >= pretrain_steps 的存档，自动跳过。

    Phase 2 - 在线 DAgger：
      - 启动 agentlace 服务，接收 Actor 推送的干预数据到 intvn_buffer
      - 等待 intvn_buffer 积累 training_starts 条后开始训练
      - 持续对 intvn_buffer（+ 可选 demo_buffer 混合）做 BC 更新
      - 每 steps_per_update 步同步网络到 Actor

    参数：
      intvn_buffer: 在线干预 buffer，从空开始，随 Actor 运行持续增长（DAgger 聚合）
      demo_buffer:  离线 demo buffer，只读，用于 Phase 1 预训练 + Phase 2 可选混合
      env_obs_space, env_act_space: 用于创建 replay_buffer（agentlace 占位）
    """
    # 确定 start_step
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest is not None:
            start_step = int(os.path.basename(latest)[11:]) + 1

    step = start_step  # stats_callback 闭包捕获此变量

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    # =========================================================
    # Phase 1：离线 BC 预热
    # =========================================================
    pretrain_steps = FLAGS.pretrain_steps

    if pretrain_steps > 0 and start_step < pretrain_steps:
        print_green(
            f"[Phase 1] 离线 BC 预热：{start_step} → {pretrain_steps} 步，"
            f"demo_buffer 大小={len(demo_buffer)}"
        )

        pretrain_iterator = demo_buffer.get_iterator(
            sample_args={"batch_size": config.batch_size, "pack_obs": True},
            device=sharding.replicate(),
        )

        for pt_step in tqdm.tqdm(
            range(start_step, pretrain_steps), dynamic_ncols=True, desc="[Phase 1] pretrain BC"
        ):
            batch = next(pretrain_iterator)
            agent, info = bc_update_step(agent, batch, has_grasp_critic=has_grasp_critic)

            if pt_step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(
                    {"pretrain/" + k: v for k, v in info.items()}, step=pt_step
                )

            if (
                pt_step > 0
                and config.checkpoint_period
                and pt_step % config.checkpoint_period == 0
                and FLAGS.checkpoint_path
            ):
                checkpoints.save_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path),
                    agent.state,
                    step=pt_step,
                    keep=100,
                )

        # 保存预训练完成 checkpoint
        if FLAGS.checkpoint_path:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=pretrain_steps,
                keep=100,
            )
        print_green(f"[Phase 1] 完成，checkpoint 已保存 (step={pretrain_steps})")
        step = pretrain_steps

    elif pretrain_steps > 0 and start_step >= pretrain_steps:
        print_green(
            f"[Phase 1] 已完成（start_step={start_step} >= pretrain_steps={pretrain_steps}），跳过"
        )
    else:
        print_green("[Phase 1] pretrain_steps=0，跳过")

    # =========================================================
    # Phase 2：在线 DAgger
    # =========================================================
    print_green("[Phase 2] 在线 DAgger 启动，等待 Actor 推送干预数据...")

    # agentlace 服务
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)

    # replay_buffer：接收 Actor 推送的所有 transition，Learner 不用于训练
    # （保留此注册是为了与 Actor 的 agentlace client 兼容）
    replay_buffer = MemoryEfficientReplayBufferDataStore(
        env_obs_space,
        env_act_space,
        capacity=min(config.replay_buffer_capacity, 50000),  # 不用于训练，减小容量
        image_keys=config.image_keys,
        include_grasp_penalty=include_grasp_penalty,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", intvn_buffer)
    server.start(threaded=True)

    # 等待 intvn_buffer 积累足够数据才开始训练
    # 注意：DAgger 的干预数据比 RL 的 replay buffer 填充慢（只有人类接管时才有）
    # 如果 training_starts 太大，等待时间会很长；建议在 config 中适当调小
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(intvn_buffer),
        desc="[Phase 2] Filling intervention buffer",
        position=0,
        leave=True,
    )
    while len(intvn_buffer) < config.training_starts:
        pbar.update(len(intvn_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(intvn_buffer) - pbar.n)
    pbar.close()

    # 发送初始网络给 Actor（Phase 1 训练后的网络）
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # --- 构建迭代器 ---
    mix_demo_ratio = FLAGS.mix_demo_ratio
    assert 0.0 <= mix_demo_ratio < 1.0, "--mix_demo_ratio 必须在 [0.0, 1.0) 之间"

    n_intvn = (
        config.batch_size
        if mix_demo_ratio == 0.0
        else int(config.batch_size * (1.0 - mix_demo_ratio))
    )
    n_demo = config.batch_size - n_intvn  # 0 when mix_demo_ratio == 0.0

    intvn_iterator = intvn_buffer.get_iterator(
        sample_args={"batch_size": n_intvn, "pack_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = (
        demo_buffer.get_iterator(
            sample_args={"batch_size": n_demo, "pack_obs": True},
            device=sharding.replicate(),
        )
        if mix_demo_ratio > 0.0
        else None
    )

    if mix_demo_ratio > 0.0:
        print_green(
            f"Batch 构成：干预数据 {n_intvn} 条 + demo {n_demo} 条 = {config.batch_size} 条"
        )
    else:
        print_green(f"Batch 构成：纯干预数据 {n_intvn} 条（不混合 demo）")

    timer = Timer()

    for step in tqdm.tqdm(
        range(step, config.max_steps), dynamic_ncols=True, desc="[Phase 2] learner"
    ):
        # 采样 batch
        with timer.context("sample"):
            intvn_batch = next(intvn_iterator)
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(intvn_batch, demo_batch, axis=0)
            else:
                batch = intvn_batch

        # BC 更新
        with timer.context("bc_update"):
            agent, update_info = bc_update_step(agent, batch, has_grasp_critic=has_grasp_critic)

        # 定期同步网络到 Actor
        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # 日志
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"online/" + k: v for k, v in update_info.items()}, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            wandb_logger.log(
                {
                    "intvn_buffer_size": len(intvn_buffer),
                    "replay_buffer_size": len(replay_buffer),
                },
                step=step,
            )

        # 保存 checkpoint
        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
            and FLAGS.checkpoint_path
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=step,
                keep=100,
            )


# ===========================================================================
# Main
# ===========================================================================

def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=False,
        stack_obs_num=2,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    # 创建 agent（与 train_rlpd.py 完全一致）
    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    # 加载已有 checkpoint（如果存在）
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest is not None:
            input(f"Checkpoint 已存在（{os.path.basename(latest)}）。按 Enter 继续训练。")
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state
            )
            agent = agent.replace(state=ckpt)
            print_green(f"加载 checkpoint at step {os.path.basename(latest)[11:]}.")

    # --- Learner 模式 ---
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())

        # demo_buffer：离线 demo 数据（只读）
        assert FLAGS.demo_path is not None, "--demo_path 必须提供（用于 Phase 1 BC 预热）"
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for t in transitions:
                    if "infos" in t and "grasp_penalty" in t["infos"]:
                        t["grasp_penalty"] = t["infos"]["grasp_penalty"]
                    demo_buffer.insert(t)
        print_green(f"Demo buffer 大小: {len(demo_buffer)}")

        # intvn_buffer：在线干预数据（从 Actor 积累）
        intvn_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        # 续训：恢复历史干预数据
        _intvn_dir = (
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if FLAGS.checkpoint_path
            else None
        )
        if _intvn_dir and os.path.exists(_intvn_dir):
            for f_path in glob.glob(os.path.join(_intvn_dir, "*.pkl")):
                with open(f_path, "rb") as f:
                    for t in pkl.load(f):
                        intvn_buffer.insert(t)
            print_green(f"恢复历史干预数据，intvn_buffer 大小: {len(intvn_buffer)}")

        _wandb_desc = (
            f"{FLAGS.exp_name}__hgdagger__{FLAGS.run_tag}"
            if FLAGS.run_tag
            else f"{FLAGS.exp_name}__hgdagger"
        )
        wandb_logger = make_wandb_logger(
            project="hg-dagger",
            description=_wandb_desc,
            debug=FLAGS.debug,
        )

        print_green("启动 Learner 循环")
        learner(
            sampling_rng,
            agent,
            intvn_buffer,
            demo_buffer,
            env_obs_space=env.observation_space,
            env_act_space=env.action_space,
            include_grasp_penalty=include_grasp_penalty,
            has_grasp_critic=include_grasp_penalty,  # single/dual-arm-learned-gripper 时为 True
            wandb_logger=wandb_logger,
        )

    # --- Actor 模式 ---
    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(20000)
        intvn_data_store = QueuedDataStore(20000)

        print_green("启动 Actor 循环")
        actor(agent, data_store, intvn_data_store, env, sampling_rng)

    else:
        raise NotImplementedError("必须指定 --learner 或 --actor")


if __name__ == "__main__":
    app.run(main)
