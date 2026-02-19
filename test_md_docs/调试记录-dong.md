# /1/25 
## 一些变量的定位
[训练的checkpoint修改位于config.py下的TrainConfig](experiments/a1x_pick_banana/config.py#L110)
## 环境读取调用链
[banana实验的配置文件](experiments/a1x_pick_banana/config.py)
[训练脚本读取配置](train_conrft_octo.py#L455)
```python
def main(_):
    # 加载配置
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # config.teleoperation_device = "gello"  ← 配置已加载
    
    # 创建环境
    env = config.get_environment(  # ← 步骤 2: 调用
        fake_env=FLAGS.learner,    #定义为false，意为真机
        save_video=FLAGS.eval_checkpoint_step,
        classifier=True,
        stack_obs_num=2
    )
```
[在get_environment()中使用](experiments/a1x_pick_banana/config.py#L128)
 ```python
def get_environment(self, fake_env=False, ...):
    env = A1XTaskEnv(...)
    
    # ← 步骤 3: 使用 self.teleoperation_device
    if not fake_env:
        if self.teleoperation_device == "gello":  # ← 这里!
            env = GelloIntervention(env, port=self.gello_port)
        elif self.teleoperation_device == "spacemouse":  # ← 这里!
            env = SpacemouseIntervention(env)
    
    return env
```

## 干预逻辑
/home/dungeon_master/conrft/serl_robot_infra/franka_env/envs/wrappers.py
[ GelloIntervention类定位](../serl_robot_infra/franka_env/envs/wrappers.py#L434)

[SpacemouseIntervention](../serl_robot_infra/franka_env/envs/wrappers.py#L373)

## obs调用流程

[最底层](../serl_robot_infra/franka_env/envs/a1x_env.py#L355)
A1XEnv._get_obs()
    ↓ 返回
{
    "images": {...},
    "state": {
        "joint_positions": [7,],      ← 包含夹爪位置
        "joint_velocities": [7,],
        "ee_pos_rot_gripper": [7,],
    }
}
    ↓ 传入
SERLObsWrapper.observation(obs)
    ↓ 根据 proprio_keys 提取并扁平化
{
    "state": [扁平化的数组],  ← 根据proprio_keys选择的键合并
    "wrist_1": [...],
    "side_policy_256": [...],
}
    ↓ 传入
ChunkingWrapper
    ↓ 添加时间维度
{
    "state": [obs_horizon, state_dim],
    "wrist_1": [...],
    "side_policy_256": [...],
}
    ↓ 传入
A1XGripperPenaltyWrapper.reset()  ← 你的包装器在这里


python /home/dungeon_master/conrft/examples/record_demos_octo_manual.py \
    --exp_name a1x_pick_banana \
    --successes_needed 20 \
    --manual_success



# 02/3
### 夹爪数据

[当前夹爪获取为 0-100]（）