#!/usr/bin/env python3
"""
train_rlpd_hil_simple.py
========================
原始 HIL-SERL 训练脚本（无 Q 值修正、无对比损失）

回归 train_rlpd.py 基础架构，保留人类干预（HIL）交互循环。
对比 train_rlpd_hil.py：
  - 去除 Module 2：偏好感知 Q 值修正（alpha_weight / corrected_critic_loss）
  - 去除 Module 3：ORPO 对比损失（PreferenceBufferDataStore / contrastive_policy_loss）
  - 去除 Module 1：Progress Model 次优片段识别（detect_suboptimal_segments）
  - 保留：eval 模式、checkpoint 续训、sampling_strategy、cov_actor_loss
"""

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from flax import config as flax_config
flax_config.update('flax_use_orbax_checkpointing', False)
import os
import json as _json
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
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
from sampling_strategies import make_sampling_strategy
from cov_actor_loss import make_cov_policy_loss_fn, compute_label_cov_stats_from_batch

FLAGS = flags.FLAGS

# ── 基础 flags ────────────────────────────────────────────────────────────
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("eval", False, "Whether to run in eval mode.")
flags.DEFINE_integer("eval_steps", 0, "Step of checkpoint to load for eval mode.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes to evaluate in eval mode.")
flags.DEFINE_boolean("debug", False, "Debug mode.")
flags.DEFINE_string("run_tag", "", "Optional tag appended to wandb run name.")

# ── 采样策略 ───────────────────────────────────────────────────────────────
flags.DEFINE_string(
    "sampling_strategy", "none",
    "Sampling strategy: none, workspace_filtering, random_drop, per.",
)
flags.DEFINE_string(
    "sampling_strategy_kwargs", "",
    'JSON-encoded kwargs for the sampling strategy, e.g. \'{"drop_ratio": 0.15}\'.',
)

# ── Cov Actor Loss ────────────────────────────────────────────────────────
flags.DEFINE_boolean(
    "use_cov_actor_loss", False,
    "Use covariance-based entropy-bounded actor loss instead of standard SAC actor loss.",
)
flags.DEFINE_integer("cov_K", 4, "MC samples per state for cov actor loss.")
flags.DEFINE_float("cov_q_low", 0.05, "Lower percentile for |c| bound.")
flags.DEFINE_float("cov_q_high", 0.90, "Upper percentile for |c| bound.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_banner(title: str, lines: list, color: str = "yellow") -> None:
    _codes = {"yellow": "\033[93m", "green": "\033[92m",
               "cyan": "\033[96m", "red": "\033[91m"}
    bold  = "\033[1m"
    reset = "\033[00m"
    c     = _codes.get(color, _codes["yellow"])
    width = max(len(title), max((len(l) for l in lines), default=0)) + 4
    border = "═" * width
    print(f"\n{c}{bold}╔{border}╗")
    print(f"║  {title:^{width - 2}}  ║")
    print(f"╠{border}╣")
    for line in lines:
        print(f"║  {line:<{width - 2}}  ║")
    print(f"╚{border}╝{reset}\n")


# =============================================================================
# Actor 循环
# =============================================================================

def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    Actor 循环（--actor 模式）。
    标准 HIL-SERL：人类干预时覆盖动作并同时写入 intvn_data_store（demo buffer）。
    """
    # ── 验证模式 ──────────────────────────────────────────────────────────
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

    # ── 断点续训起始步 ────────────────────────────────────────────────────
    start_step = (
        int(os.path.basename(
            natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1]
        )[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
           and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    datastore_dict = {
        "actor_env":       data_store,
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

    transitions      = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return   = 0.0
    already_intervened  = False
    intervention_count  = 0
    intervention_steps  = 0

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

            # 干预覆盖
            if "intervene_action_eef" in info:
                actions = info.pop("intervene_action_eef")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
                # 将 action_scale==0 的维度置 0
                _scale = np.asarray(config.action_scale) if hasattr(config, "action_scale") else None
                if _scale is not None and _scale.ndim > 0:
                    _zero_mask = (_scale == 0)
                    if _zero_mask.any():
                        actions = actions.copy()
                        actions[:len(_zero_mask)][_zero_mask[:len(actions)]] = 0.0

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                # 数据来源标签: 1=在线策略, 2=在线干预
                labels=2 if already_intervened else 1,
            )
            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]

            data_store.insert(transition)
            transitions.append(transition)

            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(transition)

            obs = next_obs

            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["intervention_rate"] = (
                    intervention_steps / max(len(transitions), 1)
                )
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return:.2f}")
                running_return      = 0.0
                intervention_count  = 0
                intervention_steps  = 0
                already_intervened  = False
                client.update()

                # 周期性落盘
                if config.buffer_period > 0 and FLAGS.checkpoint_path:
                    buffer_path      = os.path.join(FLAGS.checkpoint_path, "buffer")
                    demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                    os.makedirs(buffer_path,      exist_ok=True)
                    os.makedirs(demo_buffer_path, exist_ok=True)
                    if transitions:
                        with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                            pkl.dump(transitions, f)
                        transitions = []
                    if demo_transitions:
                        with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                            pkl.dump(demo_transitions, f)
                        demo_transitions = []

                obs, _ = env.reset()

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


# =============================================================================
# Eval 循环
# =============================================================================

def eval_policy(agent, env, sampling_rng):
    """按 's' 键成功、'f' 键失败/reset 的确定性评估循环。"""
    assert FLAGS.checkpoint_path, "Must provide --checkpoint_path for eval mode."
    assert FLAGS.eval_steps > 0, "Must provide --eval_steps for eval mode."

    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path),
        agent.state,
        step=FLAGS.eval_steps,
    )
    agent = agent.replace(state=ckpt)
    print_green(f"Loaded checkpoint at step {FLAGS.eval_steps} for eval.")
    print_green("按 's' 标记成功, 按 'f' 提前 reset（失败）")

    success_counter = 0
    episode_returns = []
    episode_lengths = []
    time_list = []

    for episode in range(FLAGS.eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        start_time = time.time()
        succeeded = False

        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=True,
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions))
            obs, reward, done, truncated, info = env.step(actions)
            ep_return += reward
            ep_length += 1
            if done or truncated:
                succeeded = bool(info.get("succeed", reward > 0))
            done = done or truncated

        elapsed = time.time() - start_time
        time_list.append(elapsed)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        if succeeded:
            success_counter += 1

        status = "✅ 成功" if succeeded else "❌ 失败/reset"
        print_green(
            f"Episode {episode + 1}/{FLAGS.eval_episodes} {status} | "
            f"return={ep_return:.2f} | length={ep_length} | time={elapsed:.1f}s"
        )

    print_green(
        f"===== Eval done | success={success_counter}/{FLAGS.eval_episodes} "
        f"({100*success_counter/FLAGS.eval_episodes:.1f}%) | "
        f"avg_return={np.mean(episode_returns):.2f} | "
        f"avg_length={np.mean(episode_lengths):.1f} | "
        f"avg_time={np.mean(time_list):.1f}s ====="
    )


# =============================================================================
# Learner 循环
# =============================================================================

def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None,
            sampling_strategy=None):
    """
    Learner 循环（--learner 模式）。
    标准 RLPD 50/50：online replay buffer + demo buffer，高 UTD 比率训练。
    """
    start_step = (
        int(os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env",       replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # 等待 replay buffer 积累足够数据
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 RLPD 采样迭代器
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True},
        device=sharding.replicate(),
    )

    timer = Timer()
    sampling_rng = jax.random.PRNGKey(42)

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # 高 UTD：前 cta_ratio-1 次只更新 Critic，最后 1 次全局更新
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch      = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch      = concat_batches(batch, demo_batch, axis=0)

            if sampling_strategy is not None:
                sampling_rng, _key = jax.random.split(sampling_rng)
                batch = sampling_strategy.apply(batch, _key, agent=agent)

            with timer.context("train_critics"):
                agent, _ = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch      = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch      = concat_batches(batch, demo_batch, axis=0)

            if sampling_strategy is not None:
                sampling_rng, _key = jax.random.split(sampling_rng)
                batch = sampling_strategy.apply(batch, _key, agent=agent)

            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        # 发布网络参数
        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # 日志
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            sampling_rng, _cov_diag_rng = jax.random.split(sampling_rng)
            _label_cov = compute_label_cov_stats_from_batch(
                forward_policy_fn=agent.forward_policy,
                forward_critic_fn=agent.forward_critic,
                temperature=agent.forward_temperature(),
                batch=batch,
                rng=_cov_diag_rng,
                K=FLAGS.cov_K,
            )
            wandb_logger.log({"cov_diag": _label_cov}, step=step)

        # Checkpoint
        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


# =============================================================================
# Main
# =============================================================================

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

    # ── 初始化 Agent ─────────────────────────────────────────────────────
    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
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

    # ── 注入 Cov Actor Loss 配置 ──────────────────────────────────────────
    if FLAGS.use_cov_actor_loss:
        agent.config["use_cov_actor_loss"] = True
        agent.config["cov_K"]              = FLAGS.cov_K
        agent.config["cov_q_low"]          = FLAGS.cov_q_low
        agent.config["cov_q_high"]         = FLAGS.cov_q_high
        agent.config["_cov_fn_factory"]    = make_cov_policy_loss_fn
        print_banner(
            "[ACTIVE] COV ACTOR LOSS",
            [
                f"MC samples  K        = {FLAGS.cov_K}",
                f"Lower  percentile    = {FLAGS.cov_q_low}",
                f"Upper  percentile    = {FLAGS.cov_q_high}",
            ],
            color="cyan",
        )

    # 分发到所有设备
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    # ── 加载已有 checkpoint ───────────────────────────────────────────────
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path), agent.state
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_label=True,
        )
        _wandb_desc = (
            f"{FLAGS.exp_name}__{FLAGS.run_tag}" if FLAGS.run_tag else FLAGS.exp_name
        )
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=_wandb_desc,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    # ── Learner 进程 ──────────────────────────────────────────────────────
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_label=True,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if "infos" in transition and "grasp_penalty" in transition["infos"]:
                        transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                    # 离线 demo 标签: 0=离线演示数据
                    transition["labels"] = 0
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # 恢复 checkpoint 时加载已有 buffer
        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    saved = pkl.load(f)
                    for t in saved:
                        t.setdefault("labels", 1)
                        replay_buffer.insert(t)
            print_green(f"Loaded previous buffer. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    saved = pkl.load(f)
                    for t in saved:
                        t.setdefault("labels", 2)
                        demo_buffer.insert(t)
            print_green(f"Loaded previous demo buffer. Demo buffer size: {len(demo_buffer)}")

        # 构建采样策略
        _ss_kwargs = _json.loads(FLAGS.sampling_strategy_kwargs) if FLAGS.sampling_strategy_kwargs else {}
        _sampling_strategy = make_sampling_strategy(FLAGS.sampling_strategy, **_ss_kwargs)
        if FLAGS.sampling_strategy != "none":
            print_banner(
                "[ACTIVE] SAMPLING STRATEGY",
                [
                    f"strategy  = {FLAGS.sampling_strategy}",
                    f"kwargs    = {_ss_kwargs if _ss_kwargs else '(default)'}",
                ],
                color="yellow",
            )

        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
            sampling_strategy=_sampling_strategy,
        )

    # ── Actor 进程 ────────────────────────────────────────────────────────
    elif FLAGS.actor:
        sampling_rng  = jax.device_put(sampling_rng, sharding.replicate())
        data_store    = QueuedDataStore(20000)
        intvn_data_store = QueuedDataStore(20000)

        print_green("starting actor loop")
        actor(agent, data_store, intvn_data_store, env, sampling_rng)

    elif FLAGS.eval:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        print_green("starting eval loop")
        eval_policy(agent, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either --learner, --actor, or --eval")


if __name__ == "__main__":
    app.run(main)
