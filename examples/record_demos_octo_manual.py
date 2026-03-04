import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import threading
import sys
import select
import termios
import tty

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
flags.DEFINE_boolean("manual_success", False, "Enable manual success detection by pressing 's' key")


class KeyboardListener:
    """监听键盘输入的类"""
    
    def __init__(self):
        self.manual_success = False
        self.running = True
        
        # 保存原始终端设置
        self.old_settings = termios.tcgetattr(sys.stdin)
        
    def start_listening(self):
        """开始监听键盘输入"""
        # 设置终端为非阻塞模式
        tty.setraw(sys.stdin.fileno())
        
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()
        return thread
        
    def _listen_loop(self):
        """监听循环"""
        while self.running:
            try:
                # 检查是否有输入
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    # 检查 Ctrl+C (ASCII 3) 和 Ctrl+\ (ASCII 28)
                    if ord(key) == 3:  # Ctrl+C
                        print("\n⚠️ 检测到 Ctrl+C，正在退出...")
                        self.running = False
                        # 恢复终端设置后重新发送中断信号
                        self.cleanup()
                        raise KeyboardInterrupt
                    elif ord(key) == 28:  # Ctrl+\
                        print("\n⚠️ 检测到 Ctrl+\\，强制退出...")
                        self.cleanup()
                        import os
                        os._exit(1)
                    elif key.lower() == 's':
                        print("\n🎯 手动设置成功! Manual success triggered!")
                        self.manual_success = True
                    elif key.lower() == 'q':
                        print("\n❌ 退出程序...")
                        self.running = False
                        break
                    elif key.lower() == 'r':
                        print("\n🔄 重置手动成功标志...")
                        self.manual_success = False
            except KeyboardInterrupt:
                # 传播 KeyboardInterrupt
                raise
            except Exception as e:
                # 在某些情况下可能出现异常，继续循环
                time.sleep(0.1)
                
    def reset_success(self):
        """重置成功标志"""
        self.manual_success = False
        
    def cleanup(self):
        """清理终端设置"""
        self.running = False
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except:
            pass


def print_instructions():
    """打印操作说明"""
    print("\n" + "="*60)
    print("🎮 录制演示数据 - 操作说明")
    print("="*60)
    print("📖 基本操作:")
    print("   - 使用 Gello 控制机器人")
    print("   - 完成任务后等待自动检测成功")
    print()
    if FLAGS.manual_success:
        print("⌨️  手动成功模式已启用:")
        print("   - 按 's' 键: 手动标记当前轨迹为成功并结束当前 episode")
        print("   - 按 'r' 键: 重置手动成功标志")
        print("   - 按 'q' 键: 退出程序")
        print("   - Ctrl+C: 立即退出程序")
        print("   - Ctrl+\\: 强制退出程序")
        print()
    else:
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
    print("="*60)
    print()


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    print("加载预训练 Octo 模型...", config.octo_path)
    model = OctoModel.load_pretrained(config.octo_path)
    tasks = model.create_tasks(texts=[config.task_desc])

    # 初始化键盘监听器
    keyboard_listener = None
    if FLAGS.manual_success:
        keyboard_listener = KeyboardListener()
        keyboard_thread = keyboard_listener.start_listening()

    try:
        obs, info = env.reset()
        print("观察空间键:", obs.keys())
        print("Reset done")
        
        print_instructions()

        transitions = []
        success_count = 0
        success_needed = FLAGS.successes_needed
        pbar = tqdm(total=success_needed, desc="成功轨迹")
        trajectory = []
        returns = 0
        episode_count = 0

        # 使用实验目录下的 demo_data 文件夹
        demo_data_dir = f"./examples/experiments/{FLAGS.exp_name}/demo_data"
        if not os.path.exists(demo_data_dir):
            os.makedirs(demo_data_dir)
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        manual_suffix = "_manual" if FLAGS.manual_success else ""
        print(f"📁 每条成功轨迹将单独保存到: {demo_data_dir}/\n")

        while success_count < success_needed:
            if keyboard_listener and not keyboard_listener.running:
                print("键盘监听器停止，正在退出...")
                raise KeyboardInterrupt
            
            # 🆕 在 step 之前检查手动成功标志，实现立即结束
            manual_success_before_step = False
            if FLAGS.manual_success and keyboard_listener and keyboard_listener.manual_success:
                manual_success_before_step = True
                # 如果已经有轨迹数据，立即结束 episode
                if len(trajectory) > 0:
                    print("\n🎯 检测到手动成功标志，立即结束当前 episode...")
                    done = True
                    # 使用最后一次观测作为 next_obs
                    next_obs = obs
                    rew = 0
                    info = {"succeed": False}  # 将在后面设置为手动成功
                    # 跳过 env.step()，直接进入 episode 结束处理
                else:
                    # 如果还没有轨迹数据，重置标志继续采集
                    keyboard_listener.reset_success()
                    manual_success_before_step = False
            
            if not manual_success_before_step:
                actions = np.zeros(env.action_space.sample().shape)
                next_obs, rew, done, truncated, info = env.step(actions)
                returns += rew
            
            # 🎯 修复：优先使用关节空间动作（A1X joint delta）
            # EEF 空间转换目前未完全实现，会导致前6维为0
            if not manual_success_before_step:
                if "intervene_action_eef" in info:
                    actions = info["intervene_action_eef"]  # A1X 关节空间 [7]（推荐）
                elif "intervene_action_eef" in info:
                    actions = info["intervene_action_eef"]  # EEF空间（备用）
            else:
                # 如果是手动成功触发，使用最后一次的动作
                if len(trajectory) > 0:
                    actions = trajectory[-1]["actions"]
                else:
                    actions = np.zeros(env.action_space.sample().shape)
            
            # 🆕 数据验证检查
            data_valid = info.get("data_valid", True)
            if not data_valid:
                if not hasattr(main, '_data_warning_count'):
                    main._data_warning_count = 0
                main._data_warning_count += 1
                if main._data_warning_count % 50 == 1:
                    print(f"\n⚠️ 警告: 检测到无效数据 (已忽略 {main._data_warning_count} 次)")
            
            # 🆕 检查动作有效性
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
            
            # 在跳过之前检查是否有手动成功标志
            if not action_valid:
                # 检查手动成功，即使数据无效也要能立即结束 episode
                manual_success_check = False
                if FLAGS.manual_success and keyboard_listener and keyboard_listener.manual_success:
                    manual_success_check = True
                    print("\n🎯 检测到手动成功标志（数据无效但强制结束）")
                    # 强制触发 episode 结束
                    done = True
                    episode_done = True
                    if episode_done and manual_success_check:
                        # 立即跳到 episode 结束处理
                        episode_count += 1
                        episode_success = True
                        
                        if len(trajectory) > 0:
                            # 处理并保存轨迹（与下面的逻辑相同）
                            print(f"\n✅ Episode {episode_count} 成功! (手动标记)")
                            print("   🔄 [无效数据分支] 步骤1: 计算MC returns...")
                            trajectory = add_mc_returns_to_trajectory(
                                trajectory, config.discount, FLAGS.reward_scale, FLAGS.reward_bias, 
                                config.reward_neg, is_sparse_reward=True)
                            print(f"   🔄 [无效数据分支] 步骤2: 添加embeddings（共 {len(trajectory)} 帧，可能需要 {len(trajectory)*2} 秒）...")
                            import time as time_module
                            start_time = time_module.time()
                            trajectory = add_embeddings_to_trajectory(
                                trajectory, model, tasks=tasks)
                            elapsed = time_module.time() - start_time
                            print(f"   ✅ embeddings完成，耗时 {elapsed:.1f} 秒")
                            print("   🔄 [无效数据分支] 步骤3: 添加next embeddings...")
                            trajectory = add_next_embeddings_to_trajectory(trajectory)
                            print("   ✅ [无效数据分支] 轨迹处理完成")
                            
                            success_count += 1
                            trajectory_file = f"{demo_data_dir}/traj_{success_count:03d}{manual_suffix}_{uuid}.pkl"
                            
                            try:
                                with open(trajectory_file, "wb") as f:
                                    pkl.dump(trajectory, f)
                                print(f"   💾 已保存轨迹到: {os.path.basename(trajectory_file)}")
                                print(f"   📊 轨迹长度: {len(trajectory)} 帧，累计成功: {success_count}/{success_needed}")
                                for transition in trajectory:
                                    transitions.append(copy.deepcopy(transition))
                                pbar.update(1)
                            except Exception as e:
                                print(f"   ⚠️ 保存失败: {e}")
                                success_count -= 1
                        
                        # Reset
                        print("   ⏳ 1秒后进入下一个 episode...")
                        time.sleep(1.0)
                        print("   🔄 开始reset环境...")
                        trajectory = []
                        returns = 0
                        obs, info = env.reset()
                        print("   ✅ Reset完成")
                        keyboard_listener.reset_success()
                        print("   🔄 继续下一个循环\n")
                        # 重新开始循环，不要继续执行后面的代码
                        continue
                else:
                    # 没有手动成功，正常跳过此帧
                    obs = next_obs
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
            
            # 检查是否应该结束 episode：环境返回 done 或手动标记成功
            manual_success = manual_success_before_step
            if not manual_success and FLAGS.manual_success and keyboard_listener:
                manual_success = keyboard_listener.manual_success
            
            episode_done = done or manual_success
            
            if episode_done:
                episode_count += 1
                
                # 判断成功条件
                episode_success = False
                
                # 1. 原始自动检测
                auto_success = info.get("succeed", False)
                
                # 2. 综合判断（已经在上面检查过了）
                episode_success = auto_success or manual_success
                
                # 显示结果
                if episode_success:
                    print(f"\n✅ Episode {episode_count} 成功! ", end="")
                    if auto_success:
                        print("(自动检测)")
                    elif manual_success:
                        print("(手动标记)")
                    
                    # 🆕 验证轨迹数据完整性
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
                        # 处理成功轨迹
                        print("   🔄 步骤1: 计算MC returns...")
                        trajectory = add_mc_returns_to_trajectory(
                            trajectory, config.discount, FLAGS.reward_scale, FLAGS.reward_bias, 
                            config.reward_neg, is_sparse_reward=True)
                        print(f"   🔄 步骤2: 添加embeddings（共 {len(trajectory)} 帧，可能需要 {len(trajectory)*2} 秒）...")
                        import time as time_module
                        start_time = time_module.time()
                        trajectory = add_embeddings_to_trajectory(
                            trajectory, model, tasks=tasks)
                        elapsed = time_module.time() - start_time
                        print(f"   ✅ embeddings完成，耗时 {elapsed:.1f} 秒")
                        print("   🔄 步骤3: 添加next embeddings...")
                        trajectory = add_next_embeddings_to_trajectory(trajectory)
                        print("   ✅ 轨迹处理完成")
                        
                        # 保存单条轨迹到独立文件
                        success_count += 1
                        trajectory_file = f"{demo_data_dir}/traj_{success_count:03d}{manual_suffix}_{uuid}.pkl"
                        
                        try:
                            with open(trajectory_file, "wb") as f:
                                pkl.dump(trajectory, f)
                            print(f"   💾 已保存轨迹到: {os.path.basename(trajectory_file)}")
                            print(f"   📊 轨迹长度: {len(trajectory)} 帧，累计成功: {success_count}/{success_needed}")
                        except Exception as e:
                            print(f"   ⚠️ 保存失败: {e}")
                            success_count -= 1  # 保存失败则不计数
                        else:
                            # 只有保存成功才添加到总列表
                            for transition in trajectory:
                                transitions.append(copy.deepcopy(transition))
                        
                        pbar.update(1)
                    else:
                        print(f"   ⏭️ 跳过无效轨迹")
                else:
                    print(f"\n❌ Episode {episode_count} 失败")
                
                # 如果是手动标记成功，延迟1秒后再reset
                if manual_success:
                    print("   ⏳ 1秒后进入下一个 episode...")
                    time.sleep(1.0)
                
                # 重置状态
                print("   🔄 [正常分支] 开始reset环境...")
                trajectory = []
                returns = 0
                obs, info = env.reset()
                print("   ✅ [正常分支] Reset完成\n")
                
                # 重置手动成功标志
                if keyboard_listener:
                    keyboard_listener.reset_success()
                    
                # time.sleep(2.0)

        # 最终确认
        print(f"\n🎉 数据收集完成！")
        print(f"   📊 总计: {success_needed} 个成功 episode")
        print(f"   📦 总transitions数: {len(transitions)}")
        print(f"   📁 保存目录: {demo_data_dir}")
        print(f"   📄 共 {success_count} 个轨迹文件")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")
        if success_count > 0:
            print(f"   💾 已收集 {success_count} 个成功 episode")
            print(f"   📁 保存在: {demo_data_dir}")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        if success_count > 0:
            print(f"   💾 已收集 {success_count} 个成功 episode")
            print(f"   📁 保存在: {demo_data_dir}")
    finally:
        # 清理键盘监听器
        if keyboard_listener:
            keyboard_listener.cleanup()
        pbar.close()


if __name__ == "__main__":
    app.run(main)