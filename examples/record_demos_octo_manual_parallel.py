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
import queue

from experiments.mappings import CONFIG_MAPPING
from data_util import (
    add_mc_returns_to_trajectory,
    add_embeddings_to_trajectory,
    add_next_embeddings_to_trajectory
)

# OctoModel 只在处理线程中按需导入

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20,
                     "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_boolean("manual_success", False, "Enable manual success detection by pressing 's' key")
flags.DEFINE_integer("num_workers", 2, "Number of parallel workers for embedding processing")


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


class EmbeddingProcessor:
    """并行处理 embeddings 的工作线程"""
    
    def __init__(self, octo_path, task_desc, demo_data_dir, manual_suffix, uuid, num_workers=2):
        # 延迟加载：只存储模型路径和任务描述
        self.octo_path = octo_path
        self.task_desc = task_desc
        self.demo_data_dir = demo_data_dir
        self.manual_suffix = manual_suffix
        self.uuid = uuid
        
        # 模型实例（延迟加载）
        self.model = None
        self.tasks = None
        self.model_loaded = False
        
        # 任务队列：存放待处理的轨迹
        self.task_queue = queue.Queue()
        
        # 结果统计
        self.processed_count = 0
        self.total_count = 0
        self.lock = threading.Lock()
        
        # 工作线程
        self.workers = []
        self.running = True
        
        # 启动工作线程
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        print(f"🔧 启动 {num_workers} 个并行处理线程 (模型将在第一次处理时加载)")
    
    def _load_model_if_needed(self, worker_id):
        """延迟加载模型（只在第一次处理时加载）"""
        with self.lock:
            if not self.model_loaded:
                print(f"[Worker-{worker_id}] 📥 首次处理，正在加载 Octo 模型...")
                print(f"[Worker-{worker_id}]    模型路径: {self.octo_path}")
                
                # 延迟导入：只在需要时才导入 OctoModel
                from octo.model.octo_model import OctoModel
                
                self.model = OctoModel.load_pretrained(self.octo_path)
                self.tasks = self.model.create_tasks(texts=[self.task_desc])
                self.model_loaded = True
                print(f"[Worker-{worker_id}] ✅ 模型加载完成，开始处理...")
    
    def add_trajectory(self, trajectory, trajectory_num):
        """添加待处理的轨迹"""
        with self.lock:
            self.total_count += 1
        self.task_queue.put((trajectory, trajectory_num))
    
    def _worker_loop(self, worker_id):
        """工作线程主循环"""
        while self.running:
            try:
                # 从队列获取任务（超时1秒避免阻塞）
                try:
                    trajectory, trajectory_num = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 🔥 延迟加载：只在第一次处理时加载模型
                self._load_model_if_needed(worker_id)
                
                traj_len = len(trajectory)
                print(f"\n[Worker-{worker_id}] 🔄 处理轨迹 {trajectory_num} (长度: {traj_len} 帧)...")
                
                try:
                    # 添加 embeddings
                    start_time = time.time()
                    trajectory = add_embeddings_to_trajectory(trajectory, self.model, tasks=self.tasks)
                    elapsed = time.time() - start_time
                    print(f"[Worker-{worker_id}]    ✅ embeddings完成，耗时 {elapsed:.1f} 秒")
                    
                    # 添加 next embeddings
                    trajectory = add_next_embeddings_to_trajectory(trajectory)
                    print(f"[Worker-{worker_id}]    ✅ next embeddings完成")
                    
                    # 保存单条轨迹到独立文件
                    trajectory_file = f"{self.demo_data_dir}/traj_{trajectory_num:03d}{self.manual_suffix}_{self.uuid}.pkl"
                    
                    with open(trajectory_file, "wb") as f:
                        pkl.dump(trajectory, f)
                    print(f"[Worker-{worker_id}]    💾 已保存: {os.path.basename(trajectory_file)}")
                    
                    # 更新统计
                    with self.lock:
                        self.processed_count += 1
                        progress = self.processed_count / self.total_count * 100 if self.total_count > 0 else 0
                        print(f"[Worker-{worker_id}]    📊 进度: {self.processed_count}/{self.total_count} ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"[Worker-{worker_id}]    ⚠️ 处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    self.task_queue.task_done()
                    
            except Exception as e:
                print(f"[Worker-{worker_id}] ⚠️ 工作线程错误: {e}")
                time.sleep(0.1)
    
    def wait_completion(self):
        """等待所有任务完成"""
        print("\n⏳ 等待所有轨迹处理完成...")
        self.task_queue.join()
        print(f"✅ 所有轨迹处理完成！共处理 {self.processed_count} 条轨迹")
    
    def stop(self):
        """停止所有工作线程"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=2.0)


def print_instructions():
    """打印操作说明"""
    print("\n" + "="*60)
    print("🎮 录制演示数据 - 操作说明 (并行处理版本)")
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
        print()
    else:
        print("⌨️  退出方式:")
        print("   - Ctrl+C: 退出程序")
        print()
    print("🎯 任务目标:")
    print("   - 需要收集 {} 个成功的演示轨迹".format(FLAGS.successes_needed))
    print()
    print("💡 提示:")
    print("   - ⚡ 成功的轨迹会立即放入后台队列进行并行处理")
    print("   - 🔧 使用 {} 个并行线程处理 embeddings".format(FLAGS.num_workers))
    print("   - 📊 采集和处理同时进行，互不阻塞")
    print("="*60)
    print()


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    print("⚠️ 采集模式：模型将在第一次处理轨迹时才加载，采集期间不占用 GPU")

    # 初始化键盘监听器
    keyboard_listener = None
    if FLAGS.manual_success:
        keyboard_listener = KeyboardListener()
        keyboard_listener.start_listening()

    # 创建数据目录
    demo_data_dir = f"./examples/experiments/{FLAGS.exp_name}/demo_data"
    if not os.path.exists(demo_data_dir):
        os.makedirs(demo_data_dir)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manual_suffix = "_manual" if FLAGS.manual_success else ""
    
    # 初始化并行处理器（延迟加载模型）
    processor = EmbeddingProcessor(
        octo_path=config.octo_path,
        task_desc=config.task_desc,
        demo_data_dir=demo_data_dir,
        manual_suffix=manual_suffix,
        uuid=uuid,
        num_workers=FLAGS.num_workers
    )

    try:
        obs, info = env.reset()
        print("观察空间键:", obs.keys())
        print("Reset done")
        
        print_instructions()

        success_count = 0
        success_needed = FLAGS.successes_needed
        pbar = tqdm(total=success_needed, desc="成功轨迹 (采集中)")
        trajectory = []
        returns = 0
        episode_count = 0

        while success_count < success_needed:
            if keyboard_listener and not keyboard_listener.running:
                print("键盘监听器停止，正在退出...")
                raise KeyboardInterrupt
            
            # 在 step 之前检查手动成功标志
            manual_success_before_step = False
            if FLAGS.manual_success and keyboard_listener and keyboard_listener.manual_success:
                manual_success_before_step = True
                if len(trajectory) > 0:
                    print("\n🎯 检测到手动成功标志，立即结束当前 episode...")
                    done = True
                    next_obs = obs
                    rew = 0
                    info = {"succeed": False}
                else:
                    keyboard_listener.reset_success()
                    manual_success_before_step = False
            
            if not manual_success_before_step:
                actions = np.zeros(env.action_space.sample().shape)
                next_obs, rew, done, truncated, info = env.step(actions)
                returns += rew
            
            # 🎯 修复：优先使用关节空间动作（A1X joint delta）
            # EEF 空间转换目前未完全实现，会导致前6维为0
            if not manual_success_before_step:
                if "intervene_action" in info:
                    actions = info["intervene_action"]  # A1X 关节空间 [7]（推荐）
                elif "intervene_action_eef" in info:
                    actions = info["intervene_action_eef"]  # EEF空间（备用）
            else:
                if len(trajectory) > 0:
                    actions = trajectory[-1]["actions"]
                else:
                    actions = np.zeros(env.action_space.sample().shape)
            
            # 数据验证
            action_valid = True
            if actions is not None:
                if np.any(np.isnan(actions)):
                    action_valid = False
                elif len(actions) not in [7, 14]:
                    action_valid = False
            else:
                action_valid = False
            
            # 只添加有效数据
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

            pbar.set_description(f"Episode {episode_count+1} | Return: {returns:.2f} | 采集: {success_count}/{success_needed}")

            obs = next_obs
            
            # 检查 episode 结束
            manual_success = manual_success_before_step
            if not manual_success and FLAGS.manual_success and keyboard_listener:
                manual_success = keyboard_listener.manual_success
            
            episode_done = done or manual_success
            
            if episode_done:
                episode_count += 1
                
                # 判断成功
                auto_success = info.get("succeed", False)
                episode_success = auto_success or manual_success
                
                if episode_success and len(trajectory) > 0:
                    print(f"\n✅ Episode {episode_count} 成功! ", end="")
                    if auto_success:
                        print("(自动检测)")
                    elif manual_success:
                        print("(手动标记)")
                    
                    # 验证轨迹完整性
                    invalid_count = sum(1 for trans in trajectory 
                                       if trans["actions"] is None or np.any(np.isnan(trans["actions"])))
                    
                    if invalid_count > len(trajectory) * 0.1:
                        print(f"   ❌ 无效帧过多 ({invalid_count}/{len(trajectory)})，跳过")
                    else:
                        # 计算 MC returns
                        print("   🔄 计算MC returns...")
                        trajectory = add_mc_returns_to_trajectory(
                            trajectory, config.discount, FLAGS.reward_scale, FLAGS.reward_bias, 
                            config.reward_neg, is_sparse_reward=True)
                        
                        # 🚀 立即提交到并行处理队列
                        success_count += 1
                        processor.add_trajectory(copy.deepcopy(trajectory), success_count)
                        print(f"   ⚡ 已提交到处理队列 (轨迹 {success_count}/{success_needed}，长度: {len(trajectory)} 帧)")
                        pbar.update(1)
                
                elif episode_success:
                    print(f"\n⚠️ Episode {episode_count} 成功但轨迹为空，跳过")
                else:
                    print(f"\n❌ Episode {episode_count} 失败")
                
                # 重置
                if manual_success:
                    time.sleep(1.0)
                
                trajectory = []
                returns = 0
                obs, info = env.reset()
                
                if keyboard_listener:
                    keyboard_listener.reset_success()

        pbar.close()
        
        # 等待所有处理完成
        print("\n" + "="*60)
        print("✅ 数据采集完成！等待后台处理...")
        print("="*60)
        processor.wait_completion()
        
        print("\n" + "="*60)
        print(f"🎉 所有数据处理完成！")
        print("="*60)
        print(f"   📊 总计: {success_needed} 个成功 episode")
        print(f"   📁 保存目录: {demo_data_dir}")
        print("="*60)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")
        print("   ⏳ 等待已提交的轨迹处理完成...")
        processor.wait_completion()
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        processor.stop()
        if keyboard_listener:
            keyboard_listener.cleanup()


if __name__ == "__main__":
    app.run(main)
