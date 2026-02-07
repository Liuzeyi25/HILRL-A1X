# Gello干预 - 评估模式说明

## 📝 修改摘要

已为 `GelloIntervention` wrapper 添加 `eval_mode` 参数，用于在评估模式下完全禁用干预功能。

---

## 🔧 修改内容

### 1. 新增参数

在 `serl_robot_infra/franka_env/envs/wrappers.py` 的 `GelloIntervention.__init__()` 中添加：

```python
def __init__(
    self, 
    env, 
    left_config_path: str,
    # ... 其他参数 ...
    eval_mode: bool = False,  # 🎯 新增：评估模式（禁用干预）
):
```

###  2. 功能实现

#### (1) 初始化时强制禁用干预

```python
self.always_intervene = always_intervene and not eval_mode  # 评估模式强制禁用
self.eval_mode = eval_mode
```

#### (2) 跳过后台控制线程

```python
# 评估模式下不启动后台控制线程
if self.threaded_control and not eval_mode:
    self._start_control_thread()
```

#### (3) `step()` 方法直接返回

```python
def step(self, action):
    # 评估模式：跳过所有干预检测
    if self.eval_mode:
        obs, rew, done, truncated, info = self.env.step(action)
        self.last_obs = obs
        info["gello_intervened"] = False
        info["eval_mode"] = True
        return obs, rew, done, truncated, info
    
    # 正常干预逻辑...
```

#### (4) 禁用空格键切换

```python
def _on_key_press(self, key):
    if key == keyboard.Key.space:
        if self.eval_mode:
            print("⚠️  评估模式下干预已禁用")
            return
        # 正常切换逻辑...
```

---

## 🎯 使用方法

### 方法 1: 修改配置文件的 `get_environment()`

在实验配置中添加 `eval_mode` 参数支持：

```python
# examples/experiments/a1x_pick_banana/config.py

def get_environment(self, fake_env=False, save_video=False, classifier=False, 
                   stack_obs_num=2, eval_mode=False):  # 新增参数
    
    if fake_env:
        env = DefaultFrankaEnv()
    else:
        env = A1XPickBananaEnv(...)
        
        # 添加 Gello 干预 wrapper（支持评估模式）
        env = GelloIntervention(
            env,
            left_config_path="path/to/config.yaml",
            eval_mode=eval_mode,  # 🎯 传递评估模式标志
            threaded_control=True,
            # ... 其他参数
        )
        
        if save_video:
            env = VideoRecordingWrapper(env, ...)
            
    return env
```

### 方法 2: 在 `train_conrft_octo.py` 中判断

```python
# examples/train_conrft_octo.py

def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 判断是否为评估模式
    is_eval_mode = (FLAGS.actor and FLAGS.eval_checkpoint_step > 0)
    
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.eval_checkpoint_step,
        classifier=False,
        stack_obs_num=2,
        eval_mode=is_eval_mode,  # 🎯 自动启用评估模式
    )
    
    # ...
```

---

## 📊 运行效果

### 评估模式（`eval_mode=True`）

```bash
$ bash run_actor_conrft.sh

✅ GelloIntervention initialized
   - Agent: BimanualAgent
   - Control rate: 500 Hz
   - Bimanual: False
   - 🚀 双线程控制: 禁用
   - 快速干预模式: 启用
   🎯 评估模式: 干预已禁用
   ⚠️  Gello 设备将被忽略，只使用 Agent 策略
   ⚪ 双向控制已禁用

# 如果按空格键
⚠️  评估模式下干预已禁用

# 每个 step 返回
info = {
    "gello_intervened": False,
    "eval_mode": True,
    "succeed": True/False,
    # ... 其他信息
}
```

### 训练模式（`eval_mode=False`）

```bash
$ bash run_actor_conrft.sh  # 不带 --eval_checkpoint_step

✅ GelloIntervention initialized
   - Agent: BimanualAgent
   - Control rate: 500 Hz
   - Bimanual: False
   - 🚀 双线程控制: 启用
   - 快速干预模式: 启用
   🎮 按空格键切换Gello干预 (当前: 禁用)
   🔄 双向控制已启用：Reset 时 Gello 跟随机器人
🚀 后台控制线程已启动 (目标频率: 500 Hz)

# 可以按空格键切换干预
🎮 Gello干预已🟢 启用
🎮 Gello干预已🔴 禁用
```

---

## ✅ 优势

### 1. **节省资源**
- 评估模式下不启动后台控制线程（500Hz）
- 不读取 Gello 设备
- 不发送额外的机器人命令

### 2. **避免意外干预**
- 即使误触 Gello 设备也不会影响评估
- 确保评估结果纯粹来自 Agent 策略

### 3. **清晰的状态标记**
- `info["eval_mode"] = True` 明确标记评估状态
- `info["gello_intervened"] = False` 确保数据记录正确

### 4. **兼容性好**
- 向后兼容：不传 `eval_mode` 参数时默认为 `False`
- 不影响现有训练代码

---

## 🔍 验证方法

### 检查评估模式是否生效

```python
# 在评估循环中添加检查
for episode in range(FLAGS.eval_n_trajs):
    obs, _ = env.reset()
    done = False
    
    while not done:
        actions, _ = agent.sample_actions(...)
        obs, reward, done, truncated, info = env.step(actions)
        
        # ✅ 验证评估模式
        assert info.get("eval_mode") == True, "评估模式未启用!"
        assert info.get("gello_intervened") == False, "检测到意外干预!"
        
        if done:
            print(f"✅ Episode {episode} 完成，无干预")
```

### 检查 info 字典内容

```python
# 训练模式（可能有干预）
info = {
    "gello_intervened": True/False,
    "intervene_action": [...],           # 如果有干预
    "intervene_action_eef": [...],       # 如果有干预
    "succeed": True/False,
    # eval_mode 不存在
}

# 评估模式（无干预）
info = {
    "gello_intervened": False,
    "eval_mode": True,
    "succeed": True/False,
    # 没有 intervene_action 相关字段
}
```

---

## 🎯 总结

| 特性 | 训练模式 | 评估模式 (`eval_mode=True`) |
|------|----------|----------------------------|
| Gello 干预 | ✅ 启用（可切换） | ❌ 禁用 |
| 后台控制线程 | ✅ 运行 | ❌ 不启动 |
| 空格键切换 | ✅ 可用 | ❌ 禁用 |
| Gello 设备读取 | ✅ 读取 | ❌ 不读取 |
| `info["gello_intervened"]` | True/False | False |
| `info["eval_mode"]` | 不存在 | True |

修改完成后，评估模式下将完全忽略 Gello 设备，确保评估结果纯粹反映 Agent 的策略性能！

---

**修改日期**: 2026-02-03  
**修改文件**: `serl_robot_infra/franka_env/envs/wrappers.py`  
**影响范围**: `GelloIntervention` wrapper
