"""
诊断为什么buffer里有intervened键，但sample出来没有
"""
import pickle
import numpy as np
from serl_launcher.serl_launcher.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer


def analyze_buffer_file():
    """分析buffer文件的内容"""
    print("=" * 80)
    print("步骤1: 分析buffer文件内容")
    print("=" * 80)
    
    buffer_file = '/home/dungeon_master/conrft/examples/experiments/insert_block/conrft/0225/buffer/transitions_105.pkl'
    with open(buffer_file, 'rb') as f:
        transitions = pickle.load(f)
    
    print(f"Buffer文件包含 {len(transitions)} 个transition")
    print(f"第一个transition的键: {list(transitions[0].keys())}")
    print(f"intervened字段存在: {'intervened' in transitions[0]}")
    
    if 'intervened' in transitions[0]:
        print(f"intervened值: {[t['intervened'] for t in transitions]}")
    
    return transitions


def check_replay_buffer_init():
    """检查ReplayBuffer初始化时是否包含intervened字段"""
    print("\n" + "=" * 80)
    print("步骤2: 检查ReplayBuffer初始化")
    print("=" * 80)
    
    # 读取ReplayBuffer的__init__代码
    with open('/home/dungeon_master/conrft/serl_launcher/serl_launcher/data/replay_buffer.py', 'r') as f:
        content = f.read()
    
    print("检查ReplayBuffer.__init__中是否有'intervened'字段:")
    if 'intervened' in content:
        print("  ✓ 找到'intervened'关键字")
        # 找到包含intervened的行
        for i, line in enumerate(content.split('\n'), 1):
            if 'intervened' in line:
                print(f"    第{i}行: {line.strip()}")
    else:
        print("  ✗ 未找到'intervened'关键字")
        print("\n  这就是问题所在！")
        print("  ReplayBuffer在初始化时没有为'intervened'字段创建存储空间。")
        print("  虽然insert时可以传入intervened，但它不会被保存到dataset_dict中。")


def simulate_insert_and_sample():
    """模拟insert和sample过程"""
    print("\n" + "=" * 80)
    print("步骤3: 模拟insert和sample过程")
    print("=" * 80)
    
    from gymnasium import spaces
    
    # 创建简单的observation和action空间
    obs_space = spaces.Dict({
        'state': spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        'pixels': spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8),
    })
    action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
    
    # 创建ReplayBuffer
    print("创建ReplayBuffer...")
    buffer = MemoryEfficientReplayBuffer(
        obs_space, 
        action_space, 
        capacity=100,
        pixel_keys=('pixels',)
    )
    
    print(f"Buffer的dataset_dict键: {list(buffer.dataset_dict.keys())}")
    print(f"'intervened' 在 dataset_dict 中: {'intervened' in buffer.dataset_dict}")
    
    # 创建一个带有intervened字段的transition
    print("\n尝试插入一个带有'intervened'字段的transition...")
    transition = {
        'observations': {
            'state': np.random.randn(10).astype(np.float32),
            'pixels': np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8),
        },
        'next_observations': {
            'state': np.random.randn(10).astype(np.float32),
            'pixels': np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8),
        },
        'actions': np.random.randn(7).astype(np.float32),
        'rewards': 0.5,
        'masks': 1.0,
        'dones': False,
        'intervened': True,  # 这个字段会丢失！
    }
    
    print(f"插入的transition包含的键: {list(transition.keys())}")
    print(f"intervened值: {transition['intervened']}")
    
    try:
        buffer.insert(transition)
        print("✓ 插入成功")
    except Exception as e:
        print(f"✗ 插入失败: {e}")
    
    # 尝试sample
    print("\n从buffer中sample一个batch...")
    try:
        batch = buffer.sample(batch_size=1)
        print(f"Sample出来的batch包含的键: {list(batch.keys())}")
        print(f"'intervened' 在 batch 中: {'intervened' in batch}")
        
        if 'intervened' not in batch:
            print("\n❌ 问题确认：'intervened'字段在sample时丢失了！")
    except Exception as e:
        print(f"✗ Sample失败: {e}")


def propose_solution():
    """提出解决方案"""
    print("\n" + "=" * 80)
    print("解决方案")
    print("=" * 80)
    
    print("""
问题原因：
- ReplayBuffer.__init__() 中没有为'intervened'字段初始化存储空间
- 虽然在insert transition时，transition包含'intervened'字段，但该字段不会被保存
- 因为insert方法只会保存dataset_dict中已经初始化的字段
- Sample时自然也就取不到这个字段

解决方案：
需要在ReplayBuffer的__init__方法中添加对'intervened'字段的初始化。

修改文件: /home/dungeon_master/conrft/serl_launcher/serl_launcher/data/replay_buffer.py

在__init__方法中，添加：
    dataset_dict['intervened'] = np.empty((capacity,), dtype=bool)

类似于已有的'dones'字段的处理方式。

或者，如果想要可选的intervened字段，可以添加一个参数：
    include_intervened: Optional[bool] = False

然后在初始化时：
    if include_intervened:
        dataset_dict['intervened'] = np.empty((capacity,), dtype=bool)
""")


if __name__ == "__main__":
    transitions = analyze_buffer_file()
    check_replay_buffer_init()
    simulate_insert_and_sample()
    propose_solution()
