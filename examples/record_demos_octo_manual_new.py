###
###```
###  cd conrft
# ``python /home/dungeon_master/conrft/examples/record_demos_octo_manual_new.py \\n    --exp_name a1x_pick_banana \\n    --successes_needed 10`
###
import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING
from data_util import add_mc_returns_to_trajectory, add_embeddings_to_trajectory, add_next_embeddings_to_trajectory

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20,
                     "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_string(
    "demo_data_subdir", "20260208", 
    "Subdirectory name under demo_data for saving trajectories.")


def reorganize_transitions_to_chunks(transitions, chunk_size):
    """
    📦 离线重组transitions为action chunks
    
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
            else:
                # Episode结束，用单步的0填充
                action_chunk.append(np.zeros(action_dim, dtype=first_action.dtype))
        
        # 更新action为chunk
        trans['actions'] = np.array(action_chunk)  # Shape: [chunk_size, action_dim]
        
        chunked_transitions.append(trans)
    
    return chunked_transitions


def print_instructions():
    """打印操作说明"""
    print("\n" + "="*60)
    print("🎮 录制演示数据 - 操作说明 (新版本 - 延后处理embeddings)")
    print("="*60)
    print("📖 基本操作:")
    print("   - 使用 Gello 控制机器人")
    print("   - 完成任务后等待自动检测成功")
    print("   - 按 's' 键: 手动标记当前轨迹为成功（来自 GelloIntervention wrapper）")
    print()
    print("⌨️  退出方式:")
    print("   - Ctrl+C: 退出程序")
    print()
    print("🎯 任务目标:")
    print("   - 需要收集 {} 个成功的演示轨迹".format(FLAGS.successes_needed))
    print("   - 每次 episode 结束时会自动判断是否成功")
    print()
    print("💡 提示:")
    print("   - 绿色文字表示成功")
    print("   - 红色文字表示失败")
    print("   - 进度条显示已收集的成功轨迹数量")
    print("   - ⚡ embeddings 将在所有数据采集完成后统一处理")
    print("="*60)
    print()


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 🎯 关键修复：保存原始 action_chunk_size，然后临时禁用
    original_chunk_size = config.action_chunk_size
    print(f"⚙️  配置: 原始 action_chunk_size = {original_chunk_size}")
    print(f"📝 采集模式: 禁用 chunking (采集单步数据)")
    print(f"📦 离线处理: 将在采集完成后重组为 chunk_size={original_chunk_size}")
    
    config.action_chunk_size = None  # 采集时不使用 chunking
    
    # 🎯 创建环境：使用数据采集模式（Reset时同步，按空格不同步）
    env = config.get_environment(
        fake_env=False, 
        save_video=False, 
        classifier=False, 
        stack_obs_num=2,
        data_collection_mode=True  # ✅ 数据采集模式
    )
    
    # 恢复原始配置（用于后续离线重组）
    config.action_chunk_size = original_chunk_size
    
    print("加载预训练 Octo 模型...", config.octo_path)
    model = OctoModel.load_pretrained(config.octo_path)
    tasks = model.create_tasks(texts=[config.task_desc])

    try:
        obs, info = env.reset()
        print("观察空间键:", obs.keys())
        print("Reset done")
        
        print_instructions()

        # 用于存储所有成功的轨迹（未处理embeddings）
        all_trajectories = []
        success_count = 0
        success_needed = FLAGS.successes_needed
        pbar = tqdm(total=success_needed, desc="成功轨迹")
        trajectory = []
        returns = 0
        episode_count = 0

        # 使用实验目录下的 demo_data 文件夹
        demo_data_dir = f"./examples/experiments/{FLAGS.exp_name}/demo_data/{FLAGS.demo_data_subdir}"
        if not os.path.exists(demo_data_dir):
            os.makedirs(demo_data_dir)
            print(f"📁 创建目录: {demo_data_dir}")
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"📁 数据采集完成后将统一保存到: {demo_data_dir}/\n")

        while success_count < success_needed:
            # 🎯 执行 env.step()（采集单步数据，不使用 chunking）
            # 传入零动作（让 Gello 干预提供真实动作）
            dummy_action = np.zeros(7)  # 单步动作 [7维]
            next_obs, rew, done, truncated, info = env.step(dummy_action)
            returns += rew
            
            # 🔍 调试：打印奖励和 done 状态
            if rew > 0 or done:
                print(f"\n🔍 调试: rew={rew}, done={done}, succeed={info.get('succeed', 'N/A')}")
            
            # 🎯 使用 EEF 动作空间（已实现完整转换）
            if "intervene_action_eef" in info and info["intervene_action_eef"] is not None:
                actions = info["intervene_action_eef"]  # EEF 空间 [dx,dy,dz,drx,dry,drz,gripper]
                # 验证 EEF 动作是否有效
                if np.sum(np.abs(actions[:6])) < 1e-6:
                    print("⚠️  检测到 EEF 动作前6维全0，可能是机器人未移动")
            elif "intervene_action" in info:
                # Fallback: 使用关节空间动作
                actions = info["intervene_action"]  # A1X 关节空间 [7]
                print("ℹ️  EEF 动作不可用，使用关节空间动作")
            else:
                # 如果没有干预动作，使用零动作
                actions = np.zeros(env.action_space.sample().shape)
            
            # 数据验证检查
            data_valid = info.get("data_valid", True)
            if not data_valid:
                if not hasattr(main, '_data_warning_count'):
                    main._data_warning_count = 0
                main._data_warning_count += 1
                if main._data_warning_count % 50 == 1:
                    print(f"\n⚠️ 警告: 检测到无效数据 (已忽略 {main._data_warning_count} 次)")
            
            # 检查动作有效性
            action_valid = True
            if actions is not None:
                if np.any(np.isnan(actions)):
                    print(f"\n❌ 错误: actions 包含 NaN，跳过此帧")
                    action_valid = False
                elif len(actions) not in [7, 14]:  # 单臂7维，双臂14维
                    print(f"\n❌ 错误: actions 维度错误 {len(actions)}，跳过此帧")
                    action_valid = False
            else:
                print(f"\n❌ 错误: actions 为 None，跳过此帧")
                action_valid = False
            
            # 输出数据有效信息（定期输出，避免刷屏）
            if action_valid:
                if not hasattr(main, '_valid_data_count'):
                    main._valid_data_count = 0
                main._valid_data_count += 1
                if main._valid_data_count % 50 == 1:
                    print(f"✅ 数据有效 (已采集 {main._valid_data_count} 帧有效数据)")
            
            # 无效数据时更新 obs 并跳过
            if not action_valid:
                obs = next_obs if 'next_obs' in locals() else obs
                # 如果 done=True（可能是手动成功），继续处理 episode 结束
                if not done:
                    continue
                
            # 只有数据有效才添加 transition
            if action_valid:
                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=rew,
                        masks=1.0 - done,
                        dones=done,
                        infos=info,
                    )
                )
                trajectory.append(transition)

            pbar.set_description(f"Episode {episode_count+1} | Return: {returns:.2f} | 成功: {success_count}/{success_needed}")

            obs = next_obs
            
            # 检查是否应该结束 episode（wrapper 会处理手动成功）
            if done:
                # 🎯 如果是负奖励（'f'键标记失败），直接重新采集
                if rew < 0:
                    print(f"\n🔄 检测到负奖励 (rew={rew}) -> 重新采集 (不计入统计)")
                    trajectory = []
                    returns = 0
                    obs, info = env.reset()
                    continue  # 跳过后续处理
                
                episode_count += 1
                
                # 从 info 读取成功状态（可能是自动检测或手动标记）
                episode_success = info.get("succeed", False)
                
                # 显示结果
                if episode_success:
                    print(f"\n✅ Episode {episode_count} 成功!")
                    
                    # 验证轨迹数据完整性
                    traj_valid = True
                    if len(trajectory) == 0:
                        print("   ⚠️ 警告: 轨迹为空，跳过保存")
                        traj_valid = False
                    else:
                        # 检查所有 transition 的动作
                        invalid_count = 0
                        for i, trans in enumerate(trajectory):
                            if trans["actions"] is None or np.any(np.isnan(trans["actions"])):
                                invalid_count += 1
                        if invalid_count > 0:
                            print(f"   ⚠️ 警告: {invalid_count}/{len(trajectory)} 帧包含无效动作")
                            if invalid_count > len(trajectory) * 0.1:  # 超过10%无效
                                print("   ❌ 无效帧过多，跳过此轨迹")
                                traj_valid = False
                    
                    if traj_valid:
                        # 🔧 如果是成功episode，更新最后一帧的状态
                        # if len(trajectory) > 0:
                            # last_frame = trajectory[-1]
                            # # 更新 succeed 标志
                            # last_frame["infos"]["succeed"] = True
                            # # 🎯 更新 rewards, dones, masks
                            # last_frame["rewards"] = 1  # 成功奖励
                            # last_frame["dones"] = True
                            # last_frame["masks"] = 0.0  # masks = 1.0 - done
                            # print(f"   🔧 已更新最后一帧: rewards=1, dones=True, masks=0.0")
                        
                        # 只计算 MC returns，不处理 embeddings
                        print("   🔄 计算MC returns...")
                        trajectory = add_mc_returns_to_trajectory(
                            trajectory, config.discount, FLAGS.reward_scale, FLAGS.reward_bias, 
                            config.reward_neg, is_sparse_reward=True)
                        
                        # 保存未处理embeddings的轨迹
                        all_trajectories.append(copy.deepcopy(trajectory))
                        success_count += 1
                        print(f"   💾 轨迹已暂存 (轨迹 {success_count}/{success_needed}，长度: {len(trajectory)} 帧)")
                        pbar.update(1)
                    else:
                        print(f"   ⏭️ 跳过无效轨迹")
                else:
                    print(f"\n❌ Episode {episode_count} 失败")
                
                # 重置状态
                trajectory = []
                returns = 0
                obs, info = env.reset()

        pbar.close()
        
        # ==================== 数据采集完成，清理环境 ====================
        print("\n" + "="*60)
        print("✅ 数据采集完成！正在清理环境...")
        print("="*60)
        
        # 🔧 关键修复：在处理 embeddings 之前清理环境
        print("🧹 关闭环境...")
        try:
            if hasattr(env, 'close'):
                env.close()
            print("   ✅ 环境已关闭")
        except Exception as e:
            print(f"   ⚠️ 关闭环境时出错: {e}")
        

        print("\n" + "="*60)
        print("开始批量处理数据...")
        print("="*60)
        print(f"📊 共收集 {len(all_trajectories)} 条成功轨迹")
        
        # 🔧 步骤1: Action Chunk 重组（如果启用）
        # 使用 original_chunk_size 而不是 config.action_chunk_size（已被设为 None）
        if original_chunk_size is not None and original_chunk_size > 0:
            print(f"\n📦 步骤1: 重组为 Action Chunks (chunk_size={original_chunk_size})")
            chunk_pbar = tqdm(total=len(all_trajectories), desc="重组轨迹")
            
            chunked_trajectories = []
            for idx, trajectory in enumerate(all_trajectories):
                original_len = len(trajectory)
                chunked_traj = reorganize_transitions_to_chunks(trajectory, original_chunk_size)
                chunked_trajectories.append(chunked_traj)
                
                # 验证重组结果
                if idx == 0:  # 只打印第一条轨迹的详细信息
                    first_action = chunked_traj[0]['actions']
                    print(f"   ✅ 示例轨迹: 长度={original_len} -> {len(chunked_traj)}, "
                          f"action shape: {first_action.shape}")
                
                chunk_pbar.update(1)
            
            chunk_pbar.close()
            all_trajectories = chunked_trajectories
            print(f"   ✅ Action Chunk 重组完成\n")
        else:
            print(f"   ⏭️  跳过 Action Chunk 重组 (chunk_size={original_chunk_size})\n")
        
        # 🔧 步骤2: 添加 embeddings
        # 计算总帧数
        total_frames = sum(len(traj) for traj in all_trajectories)
        print(f"📊 总帧数: {total_frames} 帧")
        print(f"⏱️  预计处理时间: {total_frames * 2 / 60:.1f} 分钟\n")
        
        # 批量处理每条轨迹
        transitions = []
        process_pbar = tqdm(total=len(all_trajectories), desc="添加embeddings")
        
        for idx, trajectory in enumerate(all_trajectories):
            trajectory_num = idx + 1
            traj_len = len(trajectory)
            
            print(f"\n🔄 处理轨迹 {trajectory_num}/{len(all_trajectories)} (长度: {traj_len} 帧)...")
            
            # 打印action shape信息
            sample_action = trajectory[0]['actions']
            action_shape_str = f"shape={sample_action.shape}"
            if sample_action.ndim == 2:
                action_shape_str += f" (chunked)"
            print(f"   📊 动作格式: {action_shape_str}")
            
            # 添加 embeddings
            print(f"   🔄 添加embeddings (预计 {traj_len * 2:.0f} 秒)...")
            start_time = time.time()
            trajectory = add_embeddings_to_trajectory(trajectory, model, tasks=tasks)
            elapsed = time.time() - start_time
            print(f"   ✅ embeddings完成，耗时 {elapsed:.1f} 秒")
            
            # 添加 next embeddings
            print(f"   🔄 添加next embeddings...")
            trajectory = add_next_embeddings_to_trajectory(trajectory)
            print(f"   ✅ next embeddings完成")
            
            # 保存单条轨迹到独立文件
            trajectory_file = f"{demo_data_dir}/traj_{trajectory_num:03d}_{uuid}.pkl"
            
            try:
                with open(trajectory_file, "wb") as f:
                    pkl.dump(trajectory, f)
                print(f"   💾 已保存: {os.path.basename(trajectory_file)}")
            except Exception as e:
                print(f"   ⚠️ 保存失败: {e}")
            else:
                # 只有保存成功才添加到总列表
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
            
            process_pbar.update(1)
        
        process_pbar.close()
        
        # 最终确认
        print("\n" + "="*60)
        print(f"🎉 所有数据处理完成！")
        print("="*60)
        print(f"   📊 总计: {success_needed} 个成功 episode")
        print(f"   📦 总transitions数: {len(transitions)}")
        if original_chunk_size is not None and original_chunk_size > 0:
            print(f"   🎯 Action format: Chunked (size={original_chunk_size})")
        else:
            print(f"   🎯 Action format: Single-step")
        print(f"   📁 保存目录: {demo_data_dir}")
        print(f"   📄 共 {len(all_trajectories)} 个轨迹文件")
        print("="*60)
        
        print("\n✅ 程序正常结束")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")
        if success_count > 0:
            print(f"   💾 已收集 {success_count} 个成功 episode（未处理embeddings）")
            print(f"   ⚠️ 如需使用这些数据，请手动处理embeddings")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        if success_count > 0:
            print(f"   💾 已收集 {success_count} 个成功 episode（未处理embeddings）")
    finally:
        # 最终清理：确保环境和监听器都被清理
        print("\n🧹 执行最终清理...")
        
        # 清理环境
        if 'env' in locals():
            try:
                env.close()
                print("   ✅ 环境已关闭")
            except:
                pass


if __name__ == "__main__":
    app.run(main)
