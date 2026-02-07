#!/usr/bin/env python3
"""
测试模型推理输出与真实动作的对比

用法：
    python test_model_inference.py \
        --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/conrft \
        --checkpoint_step=30000 \
        --demo_path=/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/demo_data/traj_001_manual_*.pkl \
        --num_samples=100
"""

import pickle as pkl
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags
from flax.training import checkpoints
from pathlib import Path
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from tqdm import tqdm

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.launcher import (
    make_conrft_octo_cp_pixel_agent_single_arm,
)
from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "a1x_pick_banana", "实验名称")
flags.DEFINE_string("checkpoint_path", None, "Checkpoint 路径", required=True)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint 步数", required=True)
flags.DEFINE_string("demo_path", None, "演示数据路径 (支持通配符)", required=True)
flags.DEFINE_integer("num_samples", 100, "测试样本数量")
flags.DEFINE_boolean("visualize", True, "是否可视化结果")
flags.DEFINE_boolean("verbose", False, "是否打印详细信息")


def load_demo_data(demo_path: str):
    """加载演示数据"""
    print(f"\n📂 加载演示数据...")
    print(f"   路径模式: {demo_path}")
    
    # 支持通配符
    files = natsorted(glob.glob(demo_path))
    
    if not files:
        print(f"❌ 未找到匹配的文件: {demo_path}")
        return []
    
    print(f"   找到 {len(files)} 个文件")
    
    all_transitions = []
    for file_path in files:
        print(f"   加载: {Path(file_path).name}")
        with open(file_path, "rb") as f:
            trajectory = pkl.load(f)
            all_transitions.extend(trajectory)
    
    print(f"✅ 共加载 {len(all_transitions)} 个 transitions")
    return all_transitions


def create_agent_and_load_checkpoint(config, checkpoint_path, checkpoint_step):
    """创建 agent 并加载 checkpoint"""
    print(f"\n🤖 创建 Agent...")
    
    # 创建假环境用于获取 observation/action space
    env = config.get_environment(
        fake_env=True, 
        save_video=False, 
        classifier=False, 
        stack_obs_num=2
    )
    
    # 加载 Octo 模型
    print(f"📥 加载 Octo 模型: {config.octo_path}")
    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])
    
    # 创建 agent
    print(f"🔧 创建 Agent...")
    agent = make_conrft_octo_cp_pixel_agent_single_arm(
        seed=42,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        sample_tasks=tasks,
        octo_model=octo_model,
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        fix_gripper=(config.setup_mode == 'single-arm-fixed-gripper'),
    )
    
    # 复制到所有设备
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), 
        sharding.replicate()
    )
    
    # 加载 checkpoint
    print(f"\n📦 加载 Checkpoint...")
    print(f"   路径: {checkpoint_path}")
    print(f"   步数: {checkpoint_step}")
    
    ckpt = checkpoints.restore_checkpoint(
        checkpoint_path,
        agent.state,
        step=checkpoint_step,
    )
    agent = agent.replace(state=ckpt)
    
    print(f"✅ Checkpoint 加载成功")
    
    return agent, tasks, env


def test_model_inference(agent, tasks, transitions, num_samples, verbose=False):
    """测试模型推理"""
    print(f"\n🧪 开始测试模型推理...")
    print(f"   测试样本数: {num_samples}")
    
    # 随机选择样本
    if len(transitions) > num_samples:
        indices = np.random.choice(len(transitions), num_samples, replace=False)
        test_samples = [transitions[i] for i in indices]
    else:
        test_samples = transitions
        print(f"   ⚠️  样本数不足，使用全部 {len(transitions)} 个样本")
    
    # 存储结果
    results = {
        'predicted_actions': [],
        'true_actions': [],
        'errors': [],
        'observations': []
    }
    
    print(f"\n🔄 开始推理...")
    for i, sample in enumerate(tqdm(test_samples, desc="推理进度")):
        obs = sample['observations']
        true_action = sample['actions']
        
        # 模型推理（argmax mode for evaluation）
        try:
            predicted_action = agent.sample_actions(
                observations=jax.device_put(obs),
                tasks=jax.device_put(tasks),
                argmax=True,  # 确定性输出
                seed=jax.random.PRNGKey(0)
            )
            predicted_action = np.asarray(jax.device_get(predicted_action))
            
            # 计算误差
            error = np.abs(predicted_action - true_action)
            
            results['predicted_actions'].append(predicted_action)
            results['true_actions'].append(true_action)
            results['errors'].append(error)
            results['observations'].append(obs)
            
            if verbose and i < 5:
                print(f"\n样本 {i}:")
                print(f"  真实动作: {true_action}")
                print(f"  预测动作: {predicted_action}")
                print(f"  误差:     {error}")
                
        except Exception as e:
            print(f"\n❌ 样本 {i} 推理失败: {e}")
            continue
    
    # 转换为 numpy 数组
    results['predicted_actions'] = np.array(results['predicted_actions'])
    results['true_actions'] = np.array(results['true_actions'])
    results['errors'] = np.array(results['errors'])
    
    return results


def analyze_results(results):
    """分析结果"""
    print(f"\n" + "="*70)
    print("📊 结果分析")
    print("="*70)
    
    errors = results['errors']
    
    # 整体统计
    print(f"\n整体误差统计:")
    print(f"  样本数: {len(errors)}")
    print(f"  平均误差 (MAE): {np.mean(errors):.4f}")
    print(f"  误差标准差:     {np.std(errors):.4f}")
    print(f"  最大误差:       {np.max(errors):.4f}")
    print(f"  最小误差:       {np.min(errors):.4f}")
    
    # 各维度统计（假设是7维动作：6个关节 + 夹爪）
    if errors.shape[1] == 7:
        print(f"\n各维度误差 (MAE):")
        joint_names = ["关节1", "关节2", "关节3", "关节4", "关节5", "关节6", "夹爪"]
        for i, name in enumerate(joint_names):
            mae = np.mean(errors[:, i])
            std = np.std(errors[:, i])
            print(f"  {name}: {mae:.4f} ± {std:.4f}")
    
    # 误差分布
    print(f"\n误差分布:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(errors, p)
        print(f"  {p}th percentile: {val:.4f}")


def visualize_results(results, save_path=None):
    """可视化结果"""
    print(f"\n📈 生成可视化...")
    
    predicted = results['predicted_actions']
    true = results['true_actions']
    errors = results['errors']
    
    num_dims = predicted.shape[1]
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('模型推理 vs 真实动作对比', fontsize=16)
    
    joint_names = [f"Joint {i+1}" for i in range(6)] + ["Gripper"]
    
    for i in range(min(num_dims, 7)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 散点图：预测 vs 真实
        ax.scatter(true[:, i], predicted[:, i], alpha=0.5, s=10)
        
        # 添加理想线 y=x
        min_val = min(true[:, i].min(), predicted[:, i].min())
        max_val = max(true[:, i].max(), predicted[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        
        ax.set_xlabel('True Action')
        ax.set_ylabel('Predicted Action')
        ax.set_title(f'{joint_names[i]}\nMAE: {np.mean(errors[:, i]):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(num_dims, 9):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 图表已保存: {save_path}")
    else:
        plt.savefig('model_inference_comparison.png', dpi=150, bbox_inches='tight')
        print(f"   💾 图表已保存: model_inference_comparison.png")
    
    # 误差直方图
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle('各维度误差分布', fontsize=16)
    
    axes2 = axes2.flatten()
    for i in range(min(num_dims, 7)):
        axes2[i].hist(errors[:, i], bins=30, alpha=0.7, edgecolor='black')
        axes2[i].set_xlabel('Absolute Error')
        axes2[i].set_ylabel('Frequency')
        axes2[i].set_title(f'{joint_names[i]}')
        axes2[i].grid(True, alpha=0.3)
        axes2[i].axvline(np.mean(errors[:, i]), color='r', linestyle='--', 
                        label=f'Mean: {np.mean(errors[:, i]):.4f}')
        axes2[i].legend()
    
    # 隐藏多余的子图
    for i in range(num_dims, 8):
        axes2[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        error_path = save_path.replace('.png', '_errors.png')
        plt.savefig(error_path, dpi=150, bbox_inches='tight')
        print(f"   💾 误差分布图已保存: {error_path}")
    else:
        plt.savefig('model_inference_errors.png', dpi=150, bbox_inches='tight')
        print(f"   💾 误差分布图已保存: model_inference_errors.png")


def main(_):
    print("="*70)
    print("🔬 模型推理测试工具")
    print("="*70)
    
    # 加载配置
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 加载演示数据
    transitions = load_demo_data(FLAGS.demo_path)
    
    if not transitions:
        print("❌ 没有可用的演示数据，退出")
        return
    
    # 创建 agent 并加载 checkpoint
    agent, tasks, env = create_agent_and_load_checkpoint(
        config, 
        FLAGS.checkpoint_path, 
        FLAGS.checkpoint_step
    )
    
    # 测试模型推理
    results = test_model_inference(
        agent, 
        tasks, 
        transitions, 
        FLAGS.num_samples,
        verbose=FLAGS.verbose
    )
    
    # 分析结果
    analyze_results(results)
    
    # 可视化
    if FLAGS.visualize:
        save_path = f"inference_comparison_step{FLAGS.checkpoint_step}.png"
        visualize_results(results, save_path)
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)


if __name__ == "__main__":
    app.run(main)
