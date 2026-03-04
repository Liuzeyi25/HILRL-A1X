o#!/usr/bin/env python3

import gc
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
from flax.core import frozen_dict
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.conrft_single_octo_cp import ConrftCPOctoAgentSingleArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from data_util import add_mc_returns_to_trajectory, add_next_embeddings_to_trajectory

from serl_launcher.utils.launcher import (
    make_conrft_octo_cp_pixel_agent_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
from sampling_strategies import make_sampling_strategy
from cov_actor_loss_diffusion import make_cov_policy_loss_fn_diffusion

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0,
                     "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")

flags.DEFINE_float("gamma", 0.95, "return discount")
flags.DEFINE_float("reward_neg", -1.0, "reward_neg for spase reward envs")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_float("q_weight", 0.1, "q_weight ")
flags.DEFINE_float("bc_weight", 1.0, "bc_weight")

flags.DEFINE_integer("pretrain_steps", 2000, "Number of pretrain steps.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_boolean(
    "save_debug_pkl", False, "Save actor obs and learner batch to debug/ every 10 steps."
)

# ---- 采样策略 (none / workspace_filtering / random_drop / per) ----
flags.DEFINE_string(
    "sampling_strategy", "none",
    "Sampling strategy for learner batches: none, workspace_filtering, random_drop, per.",
)
flags.DEFINE_string(
    "sampling_strategy_kwargs", "",
    'JSON-encoded kwargs for the sampling strategy, e.g. '
    '\'{ "x_range": [0.2, 0.8], "drop_ratio": 0.15}\'.',
)

# ---- Diffusion-adapted Cov Actor Loss (entropy-bounded masking) ----
flags.DEFINE_boolean(
    "use_cov_actor_loss", False,
    "Enable diffusion-adapted Cov Actor Loss (entropy-bounded masking using "
    "denoising reconstruction loss as proxy log-prob).",
)
flags.DEFINE_integer(
    "cov_K", 4,
    "Number of MC action samples per state for cov estimation.",
)
flags.DEFINE_float(
    "cov_q_low", 0.05,
    "Lower quantile threshold for |c(s)| mask.",
)
flags.DEFINE_float(
    "cov_q_high", 0.90,
    "Upper quantile threshold for |c(s)| mask.",
)


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_banner(title: str, lines: list, color: str = "yellow") -> None:
    """打印带边框的醒目横幅，用于确认关键配置已生效。

    color: yellow | green | cyan | red
    """
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


def reorganize_transitions_to_chunks(transitions, chunk_size):
    """
    📦 方案B：离线重组transitions为action chunks
    
    输入: [(s0,a0), (s1,a1), (s2,a2), ...]
    输出: [(s0,[a0,a1,a2,a3]), (s1,[a1,a2,a3,a4]), ...]
    
    Args:
        transitions: List of single-step transitions (actions可能是单步或chunk)
        chunk_size: Action chunk size (e.g., 4)
    
    Returns:
        List of chunked transitions
    """
    if len(transitions) == 0:
        return []
    
    chunked_transitions = []
    
    # 获取action_dim（从第一个单步动作中获取）用于对齐dtype
    first_action = transitions[0]['actions']
    if first_action.ndim == 1:
        action_dim = first_action.shape[0]
    else:  # ndim == 2, 已经是chunk
        action_dim = first_action.shape[1]
    
    for i in range(len(transitions)):
        trans = transitions[i].copy()
        
        # 收集当前及未来的actions: [a_i, a_{i+1}, ..., a_{i+chunk_size-1}]
        action_chunk = []
        for j in range(chunk_size):
            if i + j < len(transitions):
                action = transitions[i + j]['actions']
                # 如果是单步动作，直接添加；如果是chunk，取第一步
                if action.ndim == 1:
                    action_chunk.append(action)
                # else:  # 已经是chunk，只取第一步（因为只有第一步被执行了）
                #     action_chunk.append(action[0])
            else:
                # Episode结束，用单步的0填充
                action_chunk.append(np.zeros(action_dim, dtype=first_action.dtype))
        
        # 更新action为chunk
        trans['actions'] = np.array(action_chunk)  # Shape: [chunk_size, action_dim]
        
        chunked_transitions.append(trans)
    
    return chunked_transitions


##############################################################################


def actor(tasks, agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    # 🎯 评估模式：不连接 learner server
    is_eval_mode = FLAGS.eval_checkpoint_step > 0
    
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(
            FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
        else 0
    )

    client = None
    # 🔍 网络通信监控（使用 deque 限制最大长度，避免无限增长）
    from collections import deque as _deque
    network_stats = {
        "send_times": _deque(maxlen=200),  # 统计发送耗时，只保留最近200条
        "send_failures": 0,  # 发送失败次数
        "last_param_update": time.time(),  # 最后一次收到参数的时间
        "param_update_intervals": _deque(maxlen=200),  # 参数更新间隔，只保留最近200条
    }
    
    if not is_eval_mode:
        # 训练模式：连接 learner server
        datastore_dict = {
            "actor_env": data_store,
            "actor_env_intvn": intvn_data_store,
        }

        print_green(f"🔌 正在连接到 Learner: {FLAGS.ip}...")
        connection_start = time.time()
        
        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(),
            data_stores=datastore_dict,
            wait_for_server=True,
            timeout_ms=10000,  # 增加到 10 秒，避免网络延迟导致的超时
        )
        
        connection_time = time.time() - connection_start
        print_green(f"✅ 连接成功！耗时: {connection_time:.2f}s")

        # Function to update the agent with new params
        param_update_count = [0]  # 使用列表以便在闭包中修改
        
        def update_params(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))
            param_update_count[0] += 1
            
            # 🔍 记录参数更新时间
            now = time.time()
            interval = now - network_stats["last_param_update"]
            network_stats["param_update_intervals"].append(interval)
            network_stats["last_param_update"] = now
            
            # 首次接收时打印醒目提示
            if param_update_count[0] == 1:
                print_green("=" * 60)
                print_green("🎉 [Actor] 首次接收到网络参数！")
                print_green("=" * 60)
            
            # 每10次更新打印一次，避免刷屏
            if param_update_count[0] % 10 == 1:
                avg_interval = np.mean(network_stats["param_update_intervals"][-10:]) if len(network_stats["param_update_intervals"]) > 0 else 0
                print_green(
                    f"📥 [Actor] Received network params "
                    f"(update #{param_update_count[0]}, avg interval: {avg_interval:.2f}s)")

        client.recv_network_callback(update_params)
        print_green("✅ [Actor] Connected to Learner, waiting for params...")
    else:
        print_green("🎯 评估模式：跳过 learner 连接")

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    trajectory = []

    # 🔍 Debug: 保存 actor obs，每10步保存一次（由 --save_debug_pkl 控制）
    if FLAGS.save_debug_pkl:
        _actor_debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(_actor_debug_dir, exist_ok=True)

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions, action_embeddings = agent.sample_actions(
                    observations=jax.device_put(obs),
                    tasks=jax.device_put(tasks),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))
                # 裁剪 action 到 action_space 范围内（从 env config 读取）
                # actions = np.clip(actions, env.action_space.low, env.action_space.high)
                # actions = actions * env.action_scale # haoyuan
                # 🔍 Debug: 打印 action shape
                if step % 100 == 0:
                    print_green(f"[Actor] Step {step}: actions shape = {actions.shape}")

                # 🔍 Debug: 每10步保存 actor obs 到 debug 文件夹
                if FLAGS.save_debug_pkl and step % 10 == 0:
                    try:
                        _obs_save = {}
                        for k, v in obs.items():
                            if hasattr(v, 'shape'):
                                _obs_save[k] = np.asarray(v)
                            else:
                                _obs_save[k] = v
                        _obs_save['_step'] = step
                        _obs_save['_actions'] = actions
                        with open(os.path.join(_actor_debug_dir, f"actor_obs_step{step}.pkl"), "wb") as f:
                            pkl.dump(_obs_save, f)
                        print_green(f"🔍 [Debug] Saved actor obs to {_actor_debug_dir}/actor_obs_step{step}.pkl")
                    except Exception as e:
                        print_green(f"🔍 [Debug] Failed to save actor obs: {e}")

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # 🎯 处理 chunk 提前结束的情况（用0填充未执行的动作）
            if 'executed_actions' in info and 'total_chunk_size' in info:
                executed = info.pop('executed_actions')
                total = info.pop('total_chunk_size')
                if executed < total:
                    # 动作提前结束，用0填充剩余部分
                    if actions.ndim == 2:  # shape: (chunk_size, action_dim)
                        padding = np.zeros((total - executed, actions.shape[1]), dtype=actions.dtype)
                        actions = np.concatenate([actions[:executed], padding], axis=0)
                    # 如果是1维，不需要填充（说明 chunk_size=None）
                    
            # override the action with the intervention action
            if "intervene_action_eef" in info:
                actions = info.pop("intervene_action_eef")  # 单步干预动作 (action_dim,)
                # 干预时直接保存单步动作，在episode结束后统一重组】
                #Debug：dhy 打印actions
                print_green(f"[Actor][intervene_action_eef] step={step} shape={actions.shape} values={np.round(actions, 4)}")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                intervened=already_intervened,
                embeddings=action_embeddings,
            )
            # 🔍 Debug: 打印 transition action shape
            if step % 100 == 0:
                print_green(f"[Actor] Step {step}: Transition actions shape = {transition['actions'].shape}, intervened = {already_intervened}")
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']

            trajectory.append(transition)

            obs = next_obs
            if done or truncated:
                trajectory = add_mc_returns_to_trajectory(trajectory, FLAGS.gamma,
                                                          FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.reward_neg, is_sparse_reward=False
                                                          )
                trajectory = add_next_embeddings_to_trajectory(trajectory)
                
                # � 保存 mc_return 到文件
                # if FLAGS.checkpoint_path:
                #     mc_return_dir = os.path.join(FLAGS.checkpoint_path, "mc_returns")
                #     os.makedirs(mc_return_dir, exist_ok=True)
                    
                #     # 提取 mc_return 数据
                #     mc_returns = [t.get('mc_returns', None) for t in trajectory]
                #     rewards = [t.get('rewards', None) for t in trajectory]
                    
                #     # 保存为文本文件（易读）
                #     txt_path = os.path.join(mc_return_dir, f"mc_return_step{step}.txt")
                #     with open(txt_path, 'w') as f:
                #         f.write(f"Episode at step {step}\n")
                #         f.write(f"Trajectory length: {len(trajectory)}\n")
                #         f.write(f"Total return: {sum(rewards):.4f}\n")
                #         f.write(f"Success: {info.get('succeed', 'N/A')}\n")
                #         f.write("=" * 60 + "\n\n")
                #         f.write(f"{'Timestep':<10} {'Reward':<12} {'MC Return':<12}\n")
                #         f.write("-" * 60 + "\n")
                #         for i, (r, mc) in enumerate(zip(rewards, mc_returns)):
                #             f.write(f"{i:<10} {r:<12.4f} {mc if mc is not None else 'None':<12}\n")
                    
                #     # 保存为 pickle 文件（完整数据）
                #     pkl_data = {
                #         'step': step,
                #         'trajectory_length': len(trajectory),
                #         'mc_returns': mc_returns,
                #         'rewards': rewards,
                #         'total_return': sum(rewards),
                #         'succeed': info.get('succeed', None),
                #         'intervention_count': intervention_count,
                #         'intervention_steps': intervention_steps,
                #     }sf
                #     pkl_path = os.path.join(mc_return_dir, f"mc_return_step{step}.pkl")
                #     with open(pkl_path, 'wb') as f:
                #         pkl.dump(pkl_data, f)
                    
                #     print_green(f"💾 [MC Return] Saved to {mc_return_dir}/mc_return_step{step}.[txt|pkl]")
                
                # 📦� 检查是否需要重组：如果有任何单步动作，需要重组整个trajectory
                if hasattr(config, 'action_chunk_size') and config.action_chunk_size:
                    # 检查是否有单步动作（干预产生）
                    has_single_step = any(t['actions'].ndim == 1 for t in trajectory)
                    if has_single_step:
                        # 有单步动作，需要重组整个trajectory
                        trajectory = reorganize_transitions_to_chunks(trajectory, config.action_chunk_size)
                        if step % 100 == 0:
                            print_green(f"📦 Episode结束，重组为action chunks (size={config.action_chunk_size})")
                    # # else:
                    #     # 所有actions已经是chunk格式（策略输出），仍需重组以处理最后几帧不满的情况
                    #     trajectory = reorganize_transitions_to_chunks(trajectory, config.action_chunk_size)
                    #     if step % 100 == 0:
                    #         print_green(f"✅ 重组chunk trajectory，填充最后几帧")
                
                episode_intvn = []
                for transition in trajectory:
                    ####Debug之前：写入data_store之前打印actions
                    print_green(
                        f"[Actor][pre-insert] step={step} "
                        f"actions shape={transition['actions'].shape} "
                        f"values={np.round(transition['actions'], 4)} "
                        f"intervened={transition['intervened']}"
                    )
                    data_store.insert(transition)
                    if transition['intervened']:
                        intvn_data_store.insert(transition)
                        episode_intvn.append(transition)

                # 💾 每个 episode 结束后立即写 pkl，避免内存累积
                if FLAGS.checkpoint_path and config.buffer_period > 0:
                    buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                    os.makedirs(buffer_path, exist_ok=True)
                    with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                        pkl.dump(trajectory, f)
                    if episode_intvn:
                        demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                        os.makedirs(demo_buffer_path, exist_ok=True)
                        with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                            pkl.dump(episode_intvn, f)

                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["succeed"] = int(info['succeed'])
                info["episode"]["total_steps"] = step
                
                # 🎯 评估模式：只打印统计，不发送给 learner
                if is_eval_mode:
                    print_green(f"Episode {step}: return={running_return:.2f}, succeed={info['episode']['succeed']}")
                else:
                    # send stats to the learner to log
                    stats = {"environment": info}
                    
                    # 🔍 监控统计发送性能
                    send_start = time.time()
                    send_success = False
                    try:
                        client.request("send-stats", stats)
                        send_time = time.time() - send_start
                        network_stats["send_times"].append(send_time)
                        send_success = True
                        
                        if step % 100 == 0:  # 每100步打印一次
                            avg_send = np.mean(network_stats["send_times"][-10:])
                            max_send = np.max(network_stats["send_times"][-10:])
                            print_green(
                                f"📤 [Actor] Sent episode stats to Learner "
                                f"(return={running_return:.2f}, "
                                f"send_time={send_time*1000:.1f}ms, "
                                f"avg={avg_send*1000:.1f}ms, max={max_send*1000:.1f}ms)")
                    except Exception as e:
                        send_time = time.time() - send_start
                        network_stats["send_failures"] += 1
                        print_green(
                            f"⚠️  Failed to send stats to learner: {e}")
                        print_green(
                            f"   Attempt time: {send_time:.2f}s, "
                            f"Total failures: {network_stats['send_failures']}")
                        # 继续运行，不因为统计发送失败而中断训练
                
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                
                if not is_eval_mode:
                    client.update()
                
                trajectory = []
                gc.collect()  # 立即归还图像/embedding 内存给 OS
                obs, _ = env.reset()

        # pkl 已在 episode 结束时按 episode 写入，无需按 buffer_period 批量 dump

        timer.tock("total")

        if step % config.log_period == 0 and not is_eval_mode:
            stats = {"timer": timer.get_average_times()}
            
            # 🔍 添加网络诊断信息
            send_start = time.time()
            try:
                client.request("send-stats", stats)
                send_time = time.time() - send_start
                network_stats["send_times"].append(send_time)
            except Exception as e:
                network_stats["send_failures"] += 1
                print_green(f"⚠️  Failed to send timer stats: {e}")
            
            # 🔍 定期打印网络健康状况（每50个log_period）
            if step % (config.log_period * 50) == 0 and step > 0:
                print_green("\n" + "=" * 70)
                print_green("📊 网络通信健康检查")
                print_green("=" * 70)
                
                # 参数更新统计
                time_since_param = time.time() - network_stats["last_param_update"]
                intervals = network_stats["param_update_intervals"]
                if len(intervals) > 0:
                    avg_interval = np.mean(intervals)
                    max_interval = np.max(intervals)
                    min_interval = np.min(intervals)
                    print_green("📥 参数更新:")
                    print_green(f"   最后更新: {time_since_param:.1f}s 前")
                    print_green(f"   平均间隔: {avg_interval:.2f}s")
                    print_green(f"   最大间隔: {max_interval:.2f}s")
                    print_green(f"   最小间隔: {min_interval:.2f}s")
                    
                    # 🚨 警告：参数更新间隔过长
                    if time_since_param > 60:
                        msg = f"⚠️  警告: 超过 {time_since_param:.0f}s "
                        msg += "未收到参数更新！"
                        print_green(msg)
                        print_green("   Learner 可能阻塞或网络中断")
                else:
                    print_green("📥 参数更新: 尚未收到任何参数")
                    print_green(f"⚠️  警告: 已等待 {time_since_param:.0f}s")
                
                # 统计发送性能
                send_times = network_stats["send_times"]
                if len(send_times) > 0:
                    avg_send = np.mean(send_times) * 1000
                    max_send = np.max(send_times) * 1000
                    min_send = np.min(send_times) * 1000
                    p95_send = np.percentile(send_times, 95) * 1000
                    print_green("📤 统计发送:")
                    print_green(f"   平均延迟: {avg_send:.1f}ms")
                    print_green(f"   P95 延迟: {p95_send:.1f}ms")
                    print_green(f"   最大延迟: {max_send:.1f}ms")
                    print_green(f"   最小延迟: {min_send:.1f}ms")
                    fail_count = network_stats['send_failures']
                    print_green(f"   失败次数: {fail_count}")
                    
                    # 🚨 警告：发送延迟过高
                    if avg_send > 500:
                        msg = f"⚠️  警告: 平均发送延迟 {avg_send:.0f}ms 过高！"
                        print_green(msg)
                        print_green("   网络可能存在拥塞")
                    if max_send > 2000:
                        msg = f"⚠️  警告: 最大发送延迟 {max_send:.0f}ms！"
                        print_green(msg)
                        print_green("   可能有阻塞发生")
                
                print_green("=" * 70 + "\n")


##############################################################################


def learner(rng, tasks, agent, replay_buffer, demo_buffer, wandb_logger=None,
            sampling_strategy=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path) if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) else None
    start_step = (
        int(os.path.basename(latest_ckpt)[11:]) + 1
        if latest_ckpt is not None
        else 0
    )
    step = start_step
    online_start_step = start_step

    # 创建txt日志文件用于保存训练指标
    log_file_path = None
    if FLAGS.checkpoint_path:
        log_dir = os.path.join(FLAGS.checkpoint_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"training_metrics_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        print_green(f"Training metrics will be saved to: {log_file_path}")
        # 写入文件头
        with open(log_file_path, 'w') as f:
            f.write(f"Training started at step {start_step}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    # 📊 成功率统计缓冲区：收集自上次记录以来的所有轨迹成功情况
    success_buffer = []
    # 📊 干预率统计缓冲区：收集自上次记录以来的所有轨迹干预率
    intervention_rate_buffer = []

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        
        # 🔍 打印接收到的统计信息
        if "environment" in payload:
            env_info = payload["environment"]
            if "episode" in env_info:
                ep_info = env_info["episode"]
                # 安全地提取并格式化数值
                return_val = ep_info.get('r', None)
                if return_val is not None:
                    # 转换 numpy 数组为标量
                    return_val = float(np.asarray(return_val).item() if hasattr(return_val, 'item') else return_val)
                    return_str = f"{return_val:.2f}"
                else:
                    return_str = "N/A"
                
                succeed_val = ep_info.get('succeed', 'N/A')
                step_val = ep_info.get('total_steps', 'N/A')
                
                # 📊 收集成功率数据到缓冲区
                if 'succeed' in ep_info:
                    success_value = ep_info['succeed']
                    # 转换为标量（处理可能的 numpy 数组）
                    if hasattr(success_value, 'item'):
                        success_value = float(success_value.item())
                    else:
                        success_value = float(success_value)
                    success_buffer.append(success_value)
                
                # 📊 收集干预率数据到缓冲区
                if 'intervention_steps' in ep_info and 'total_steps' in ep_info:
                    intervention_steps = ep_info['intervention_steps']
                    total_steps = ep_info['total_steps']
                    # 转换为标量
                    if hasattr(intervention_steps, 'item'):
                        intervention_steps = float(intervention_steps.item())
                    else:
                        intervention_steps = float(intervention_steps)
                    if hasattr(total_steps, 'item'):
                        total_steps = float(total_steps.item())
                    else:
                        total_steps = float(total_steps)
                    
                    # 计算该轨迹的干预率
                    if total_steps > 0:
                        intervention_rate = intervention_steps / total_steps
                        intervention_rate_buffer.append(intervention_rate)
                
                print_green(
                    f"📥 [Learner] Received stats from Actor: "
                    f"return={return_str}, "
                    f"succeed={succeed_val}, "
                    f"step={step_val}"
                )
        elif "timer" in payload:
            # Timer 统计较多，只在 verbose 模式打印
            if step % 1000 == 0:
                print_green(f"📥 [Learner] Received timer stats from Actor")
        
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(),
                           request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    train_critic_networks_to_update = frozenset({"critic"})
    train_actor_networks_to_update = frozenset({"actor"})
    train_networks_to_update = frozenset({"critic", "actor"})

    def create_batch_tasks(data_dict, batch_size):
        batch_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):  # Handling nested dictionary (e.g., language_instruction)
                batch_dict[key] = {k: np.tile(
                    v, (batch_size, *([1] * (v.ndim - 1)))) for k, v in value.items()}
            else:
                # For non-dictionary values, repeat along batch dimension (axis=0)
                batch_dict[key] = np.tile(
                    value, (batch_size, *([1] * (value.ndim - 1))))  # Repeat along axis 0

        return batch_dict

    # Pretrain the model with the demo data
    if step < FLAGS.pretrain_steps:
        print_green("Pretraining the model with demo data")
        for step in tqdm.tqdm(range(start_step, FLAGS.pretrain_steps + 1), desc="pretraining"):
            for _ in range(config.cta_ratio - 1):
                batch = next(demo_buffer.get_iterator(
                    sample_args={"batch_size": config.batch_size,
                                 "pack_obs": True, },
                    device=sharding.replicate(),
                ))

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
                batch = frozen_dict.freeze(batch)
                agent, critics_info = agent.update_calql(
                    batch, networks_to_update=train_critic_networks_to_update,)

            batch = next(demo_buffer.get_iterator(
                sample_args={"batch_size": config.batch_size,
                             "pack_obs": True, },
                device=sharding.replicate(),
            ))

            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)

            agent, update_info = agent.update_calql(
                batch, networks_to_update=train_networks_to_update,)

            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=step)
            
            # 同时保存到txt文件（预训练阶段）
            if step % config.log_period == 0 and log_file_path:
                with open(log_file_path, 'a') as f:
                    f.write(f"[Pretrain] Step {step}:\n")
                    for key, value in sorted(update_info.items()):
                        if isinstance(value, (int, float, np.number)):
                            f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n")

            if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path, agent.state, step=step, keep=200)

        print_green("Pretraining done")
        return  # after pretraining, return and exit
    else:
        print_green(
            "Existing pretrained checkpoint model found. Skipping pretraining")

    agent = jax.block_until_ready(agent)
    server.publish_network(agent.state.params)
    print_green("📤 [Learner] Published initial network parameters")

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    sampling_rng = jax.random.PRNGKey(42)

    # 🔬 对齐诊断：只在第一个 train step 保存一次采样的 batch
    _diag_batch_saved = False

    # 🔍 Debug: 保存 learner sample batch，每10步保存一次（由 --save_debug_pkl 控制）
    if FLAGS.save_debug_pkl:
        _learner_debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(_learner_debug_dir, exist_ok=True)

    # Start online training after offline pretraining
    online_start_step = FLAGS.pretrain_steps + \
        1 if online_start_step < FLAGS.pretrain_steps else online_start_step
    for step in tqdm.tqdm(range(online_start_step, config.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)

                # 🔍 Debug: 每100步保存 learner sample batch 到 debug 文件夹（在合并前分别保存）
                if FLAGS.save_debug_pkl and step % 100 == 0:
                    try:
                        def _serialize_batch(src_batch):
                            """将 batch dict 递归转为可 pickle 的 numpy dict。"""
                            out = {}
                            for k, v in src_batch.items():
                                if k == "tasks":
                                    continue
                                if hasattr(v, 'shape'):
                                    out[k] = np.asarray(v)
                                elif isinstance(v, dict):
                                    out[k] = {
                                        sk: np.asarray(sv) if hasattr(sv, 'shape') else sv
                                        for sk, sv in v.items()
                                    }
                                else:
                                    out[k] = v
                            return out

                        _batch_save = {
                            '_step': step,
                            '_critic_step': critic_step,
                            'online_batch': _serialize_batch(batch),       # 合并前的 online replay 部分
                            'demo_batch': _serialize_batch(demo_batch),    # 合并前的 demo 部分
                        }

                        # 🔍 统计干预数据（分别检查 online 和 demo）
                        half_size = config.batch_size // 2
                        _intervention_stats = {
                            'online_size': half_size,
                            'demo_size': half_size,
                        }
                        has_intervention_info = False

                        if 'intervened' in batch:
                            online_intervened_arr = np.asarray(batch['intervened'])
                            online_intervened = int(np.sum(online_intervened_arr))
                            _intervention_stats['online_has_intervened'] = True
                            _intervention_stats['online_intervened_count'] = online_intervened
                            _intervention_stats['online_intervened_ratio'] = online_intervened / half_size
                            has_intervention_info = True
                        else:
                            _intervention_stats['online_has_intervened'] = False

                        if 'intervened' in demo_batch:
                            demo_intervened_arr = np.asarray(demo_batch['intervened'])
                            demo_intervened = int(np.sum(demo_intervened_arr))
                            _intervention_stats['demo_has_intervened'] = True
                            _intervention_stats['demo_intervened_count'] = demo_intervened
                            _intervention_stats['demo_intervened_ratio'] = demo_intervened / half_size
                            has_intervention_info = True
                        else:
                            _intervention_stats['demo_has_intervened'] = False

                        _batch_save['_intervention_stats'] = _intervention_stats

                        if has_intervention_info:
                            online_str = (f"intervened={_intervention_stats.get('online_intervened_count', 'N/A')}"
                                          if _intervention_stats['online_has_intervened'] else "no intervened field")
                            demo_str = (f"intervened={_intervention_stats.get('demo_intervened_count', 'N/A')}"
                                        if _intervention_stats['demo_has_intervened'] else "no intervened field")
                            print_green(
                                f"🔍 [Debug] Batch composition: "
                                f"online={half_size} ({online_str}), "
                                f"demo={half_size} ({demo_str})"
                            )

                        with open(os.path.join(_learner_debug_dir, f"learner_sample_batch_step{step}.pkl"), "wb") as f:
                            pkl.dump(_batch_save, f)
                        print_green(f"🔍 [Debug] Saved learner sample batch to {_learner_debug_dir}/learner_sample_batch_step{step}.pkl")
                    except Exception as e:
                        print_green(f"🔍 [Debug] Failed to save learner sample batch: {e}")

                batch = concat_batches(batch, demo_batch, axis=0)

                # ---- 采样策略过滤 ----
                if sampling_strategy is not None:
                    sampling_rng, _key = jax.random.split(sampling_rng)
                    batch = sampling_strategy.apply(batch, _key, agent=agent)

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }

            batch = frozen_dict.freeze(batch)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_ql(
                    batch, networks_to_update=train_critic_networks_to_update,)

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)

            # ---- 采样策略过滤 ----
            if sampling_strategy is not None:
                sampling_rng, _key = jax.random.split(sampling_rng)
                batch = sampling_strategy.apply(batch, _key, agent=agent)

            # 🔬 对齐诊断：第一次 train step 时保存 replay 和 demo 的原始 batch
            if not _diag_batch_saved and FLAGS.checkpoint_path:
                try:
                    diag_dir = os.path.join(FLAGS.checkpoint_path, "diag_batches")
                    os.makedirs(diag_dir, exist_ok=True)
                    # 保存 online replay batch（可能混有干预样本 intervened=True）
                    _replay_save = {
                        k: np.asarray(v) if hasattr(v, 'shape') else v
                        for k, v in batch.items()
                        if k not in ('tasks', 'embeddings', 'next_embeddings')
                    }
                    with open(os.path.join(diag_dir, "online_replay_batch.pkl"), "wb") as f:
                        pkl.dump(_replay_save, f)
                    # 保存 demo batch
                    _demo_save = {
                        k: np.asarray(v) if hasattr(v, 'shape') else v
                        for k, v in demo_batch.items()
                        if k not in ('tasks', 'embeddings', 'next_embeddings')
                    }
                    with open(os.path.join(diag_dir, "demo_batch.pkl"), "wb") as f:
                        pkl.dump(_demo_save, f)
                    print_green(f"🔬 [Diag] Saved alignment batch to {diag_dir}")
                    _diag_batch_saved = True
                except Exception as e:
                    print_green(f"🔬 [Diag] Failed to save batch: {e}")

            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            # 🔍 Debug: 打印 batch action shape
            if step % 100 == 0:
                print_green(f"[Learner] Step {step}: batch['actions'] shape = {batch['actions'].shape}")
                ##Debug: dhy learner读取batch之后打印actions的数值分布，检查是否合理
                _actions_np = np.asarray(batch['actions'])
                print_green(
                    f"[Learner][batch actions] step={step} "
                    f"shape={_actions_np.shape} "
                    f"min={_actions_np.min():.4f} max={_actions_np.max():.4f} "
                    f"mean={_actions_np.mean():.4f}\n"
                    f"  sample[0]={np.round(_actions_np[0], 4)}"
                )
            batch = frozen_dict.freeze(batch)
            agent, update_info = agent.update_ql(
                batch, networks_to_update=train_networks_to_update,)
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)
            if step % 100 == 0:  # 每100步打印一次，避免刷屏
                print_green(f"📤 [Learner] Published network at step {step}")

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            
            # 📊 计算并记录平均成功率
            if len(success_buffer) > 0:
                avg_success_rate = np.mean(success_buffer)
                wandb_logger.log({
                    "metrics/success_rate": avg_success_rate,
                    "metrics/trajectories_count": len(success_buffer),
                }, step=step)
                print_green(f"📊 [Learner] Step {step}: Success Rate = {avg_success_rate:.2%} (based on {len(success_buffer)} trajectories)")
                # 清空缓冲区，准备下一个记录周期
                success_buffer.clear()
            
            # 📊 计算并记录平均干预率
            if len(intervention_rate_buffer) > 0:
                avg_intervention_rate = np.mean(intervention_rate_buffer)
                wandb_logger.log({
                    "metrics/intervention_rate": avg_intervention_rate,
                }, step=step)
                print_green(f"📊 [Learner] Step {step}: Intervention Rate = {avg_intervention_rate:.2%} (based on {len(intervention_rate_buffer)} trajectories)")
                # 清空缓冲区，准备下一个记录周期
                intervention_rate_buffer.clear()
        
        # 同时保存到txt文件（在线训练阶段）
        if step % config.log_period == 0 and log_file_path:
            with open(log_file_path, 'a') as f:
                f.write(f"[Online] Step {step}:\n")
                for key, value in sorted(update_info.items()):
                    if isinstance(value, (int, float, np.number)):
                        f.write(f"  {key}: {value:.6f}\n")
                f.write("  Timer:\n")
                for key, value in sorted(timer.get_average_times().items()):
                    f.write(f"    {key}: {value:.6f}\n")
                f.write("\n")

        if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=step, keep=200)


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner, 
        save_video=FLAGS.eval_checkpoint_step, 
        classifier=False, 
        stack_obs_num=2,
        eval_mode=bool(FLAGS.eval_checkpoint_step))  # 🎯 评估模式：禁用干预和同步
    env = RecordEpisodeStatistics(env)

    FLAGS.reward_neg = config.reward_neg

    rng, sampling_rng = jax.random.split(rng)

    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])

    # 🔧 从 demo 数据获取正确的 action 形状（支持 chunked actions）
    sample_action = env.action_space.sample()
    # if FLAGS.demo_path is not None and len(FLAGS.demo_path) > 0:
    #     try:
    #         with open(FLAGS.demo_path[0], "rb") as f:
    #             demo_transitions = pkl.load(f)
    #             if len(demo_transitions) > 0 and 'actions' in demo_transitions[0]:
    #                 sample_action = demo_transitions[0]['actions']
    #                 print_green(f"📊 使用 demo action 形状: {sample_action.shape}")
    #     except Exception as e:
    #         print_green(f"⚠️  无法加载 demo 数据，使用默认 action 形状: {e}")

    if config.setup_mode == 'single-arm-fixed-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=sample_action,
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            fix_gripper=True,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
        )
        include_grasp_penalty = False
        include_octo_embeddings = True
        include_mc_returns = True
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=sample_action,
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
        )
        include_grasp_penalty = True
        include_octo_embeddings = True
        include_mc_returns = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # ---- 注入 Cov Actor Loss 配置到 agent.config ----
    if FLAGS.use_cov_actor_loss:
        agent = agent.replace(
            config={
                **agent.config,
                "use_cov_actor_loss": True,
                "cov_K": FLAGS.cov_K,
                "cov_q_low": FLAGS.cov_q_low,
                "cov_q_high": FLAGS.cov_q_high,
                "_cov_fn_factory": make_cov_policy_loss_fn_diffusion,
            }
        )
        print_banner(
            "🔬 Diffusion-adapted Cov Actor Loss ENABLED",
            [
                f"K (MC samples)  = {FLAGS.cov_K}",
                f"q_low           = {FLAGS.cov_q_low}",
                f"q_high          = {FLAGS.cov_q_high}",
                f"sigma_eval      = agent.config['sigma_min']",
                "proxy log-prob  = -||f_θ(a+σε,σ|s) - a||²",
            ],
            color="cyan",
        )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_util.tree_map(
        jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        # 🎯 评估模式：加载指定的 checkpoint
        if FLAGS.eval_checkpoint_step > 0:
            ckpt = checkpoints.restore_checkpoint(
                FLAGS.checkpoint_path, 
                agent.state, 
                step=FLAGS.eval_checkpoint_step)  # ✅ 加载指定步数
            print_green(f"Loaded checkpoint at step {FLAGS.eval_checkpoint_step} for evaluation.")
        else:
            # 训练模式：加载最新的 checkpoint
            latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
            if latest_ckpt is not None:
                if not FLAGS.learner:
                    input("Checkpoint path already exists. Press Enter to resume training.")
                ckpt = checkpoints.restore_checkpoint(
                    FLAGS.checkpoint_path, agent.state,)
                ckpt_number = os.path.basename(latest_ckpt)[11:]
                print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
            else:
                ckpt = None
        
        # Update params only, ignore the optimizer states
        if ckpt is not None:
            new_params = ckpt.params
            new_target_params = ckpt.target_params

            agent = agent.replace(state=agent.state.replace(
                params=new_params, target_params=new_target_params))

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        # set up wandb and logging

        wandb_logger = make_wandb_logger(
            project="conrft",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(
            sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        assert FLAGS.demo_path is not None

        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        # ---- 构建采样策略 ----
        import json as _json
        _ss_kwargs = _json.loads(FLAGS.sampling_strategy_kwargs) if FLAGS.sampling_strategy_kwargs else {}
        _sampling_strategy = make_sampling_strategy(FLAGS.sampling_strategy, **_ss_kwargs)
        if FLAGS.sampling_strategy == "none":
            print_green(f"sampling strategy: none (no filtering)")
        else:
            print_banner(
                "[ACTIVE] SAMPLING STRATEGY",
                [
                    f"strategy  = {FLAGS.sampling_strategy}",
                    f"kwargs    = {_ss_kwargs if _ss_kwargs else '(default)'}",
                ],
                color="yellow",
            )

        # learner loop
        print_green("starting learner loop")
        learner(sampling_rng,
                tasks,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                wandb_logger=wandb_logger,
                sampling_strategy=_sampling_strategy,
                )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        # 容量不宜过大：每条 transition 含图像，50000 条会占 20-80GB RAM
        # client.update() 每个 episode 结束即同步，2000 足以缓冲 3-4 个 episode
        data_store = QueuedDataStore(20000)
        intvn_data_store = QueuedDataStore(20000)

        # actor loop
        print_green("starting actor loop")
        actor(tasks,
              agent,
              data_store,
              intvn_data_store,
              env,
              sampling_rng,
              )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
