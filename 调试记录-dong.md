
 #### todo list
1.机械臂抖动(完成)

2.干预结束手动奖励写入wrapper中

3.服务器通讯

4.gello映射改为关节增量控制

# /1/25 
## 初始启动
1. a1x通电
根目录下
```bash
./a1_x_joint.bash
```
2. rostopic话题
```bash
ros2 topic echo /hdas/pose_ee_arm
```
3. gello跟随模式
```bash
➜  ~ cd conrft/Gello/gello_software
➜  gello_software git:(main) ✗ source .venv/bin/activate
(gello_software) ➜  gello_software git:(main) ✗ python experiments/launch_yaml.py --left-config-path configs/yam_A1_X.yaml
```
1. 夹香蕉采数据
```bash
cd /home/dungeon_master/conrft
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
python /home/dungeon_master/conrft/examples/record_demos_octo_manual_new.py \
     --exp_name a1x_pick_banana \
     --successes_needed 30 \
     --demo_data_subdir 20260216

python /home/dungeon_master/conrft/examples/record_demos_octo_manual_new.py \
     --exp_name insert_block \
     --successes_needed 1  \
     --demo_data_subdir 20260213

python /home/dungeon_master/conrft/examples/record_demos_octo_manual_new.py \
     --exp_name insert_network_cable \
     --successes_needed 2  \
     --demo_data_subdir 20260218

python /home/dungeon_master/conrft/examples/record_demos_octo_manual_new.py \
     --exp_name fold_towel \
     --successes_needed 2  \
     --demo_data_subdir 20260218
```
2. 检查数据

```bash
python /home/dungeon_master/conrft/examples/diagnose_bc_loss.py \
  --demo_path=/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/demo_data/20260216/traj_002_2026-02-16_15-40-13.pkl \
  --exp_name=a   \
  --detailed 



```


3.在线训练
actor启动 


```bash
 conda activate conrft
cd /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana  ###进入对应任务的目录下
bash run_actor_conrft.sh

```
learner启动
远程连接193.193.193.201
```bash
 conda activate conrft
cd /home/luka/Haoyuan/Safevla_RL/examples/experiments/a1x_pick_banana  ###进入对应任务的目录下
xvfb-run -a bash run_learner_conrft.sh
```
在run_learner_conrft.sh
```python

python ../../train_conrft_octo.py "$@" \
    --exp_name=a1x_pick_banana \             ##改为训练任务名
    --checkpoint_path=/mnt/data/Haoyuan/0212/  \    ##改为预训练的checkpoint
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=./demo_data/a1x_pick_banana_36_demos.pkl \ ###改预训练时用的pkl
    --pretrain_steps=100000 \     ####和所用预训练的checkpoint步数对齐
    --debug=False \
    --learner

```


```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```
采集脚本时需要使用上述命令
[ ] 加入环境激活中(未完成)

5. 


## 一些变量的定位

[训练的checkpoint修改位于config.py下的TrainConfig](./examples/experiments/a1x_pick_banana/config.py#L110)

## 环境读取调用链

[banana实验的配置文件](./examples/experiments/a1x_pick_banana/config.py)

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
```python
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
```

### gello Agent(读取位置)

```python
self.agent = instantiate_from_dict(left_cfg["agent"])
```
作用: 读取 Gello 遥操作设备的关节位置
方法: agent.act(None) 返回 Gello 7个关节位置 (6臂+1夹爪)
不负责: 机器人控制（底层环境已处理）
### GelloFollower(用于双向控制)
```python 
from gello.agents.gello_follower import GelloFollower
self.gello_follower = GelloFollower(dynamixel_robot)
```

作用: Reset 时让 Gello 跟随机器人位置
方法:

`start()` - 进入跟随模式
`stop()` - 退出跟随模式
`command_follow(joints) `- 发送目标位置
`get_current_position()` - 获取当前位置



### learner 和 actor之间的通信问题

方案一： learner在10.97.216.200服务器上
本机：
```bash
nc -l 3333
ssh -R 3333:localhost:3333 e230112@10.97.216.200
```
服务器：
```bash
echo "hello" | nc localhost 3333
```
本机能收到消息

依据上面逻辑修改train.py


### 训练脚本中--checkpoint_path参数问题详细解答

启动 Learner
    ↓
1. 检查 checkpoint_path 中是否有已有的 checkpoint
    ↓
2. 如果有：加载最新的 checkpoint (在 main 函数中)
    ↓
3. 确定起始步数 (start_step)
    ↓
4. 判断训练阶段
    ├─ 如果 start_step < pretrain_steps → 预训练阶段
    └─ 如果 start_step >= pretrain_steps → 在线训练阶段
    ↓
5. 保存新的 checkpoints 到同一个 checkpoint_path

tips:他再先训练之后的命名是接着预训练的继续往上加的

## 在线训练的逻辑 /home/dungeon_master/conrft/examples/train_conrft_octo.py

启动与模式选择

main() 根据 --learner / --actor 进入两条进程；learner 负责更新参数并发布，actor 负责采样并上传数据。
环境由 config.get_environment(...) 构建，并包一层 RecordEpisodeStatistics，评估模式 eval_mode 会关闭干预。
在线训练前准备（learner）

恢复 checkpoint，初始化 TrainerServer，注册两个数据源：在线 replay_buffer 和干预/示教 demo_buffer。
如果 step < pretrain_steps，先做离线预训练（只用 demo_buffer），预训练结束直接退出；否则进入在线训练。
actor 在线采样

循环执行：从策略采样动作 → env.step() → 处理干预动作（若有） → 生成 transition。
回合结束时计算 MC return、补齐 next embedding，并把数据写入 data_store（在线）和 intvn_data_store（干预）。
TrainerClient 会把环境统计（success、steps、episode length 等）上报给 learner 侧。
learner 在线更新

等待 replay_buffer 填充到 training_starts。
训练迭代中，每步采样 半在线 + 半示教：
replay_buffer 取 batch_size/2
demo_buffer 取 batch_size/2
合并后更新 critic（cta_ratio - 1 次），再做一次 critic+actor 更新。
每 steps_per_update 发布一次最新参数给 actor；每 log_period 记录指标；每 checkpoint_period 保存 checkpoint。

### 干预动作处理
####

[GelloIntervention类定位](./serl_robot_infra/franka_env/envs/wrappers.py#L458)

1. 整体干预逻辑

现在干预逻辑使用的双线程模式，[原本的step()函数跳转到threadedstep](./serl_robot_infra/franka_env/envs/wrappers.py#1116)
```python
        if self.threaded_control and self.intervention_enabled:
            return self._threaded_step(action)
```
_threaded_step为主线程，观测，奖励等最终打包为info的都在这个函数里面

后台控制线程：[control_loop](./serl_robot_infra/franka_env/envs/wrappers.py#705)
2. 奖励问题
类内原本的奖励逻辑为
```python
       rew = 0
        done = False
        truncated = False
        if env is not None and hasattr(env, 'compute_reward'):
            rew = env.compute_reward(obs)
            done = rew > 0
        if env is not None and hasattr(env, 'curr_path_length'):
            env.curr_path_length += 1
            if hasattr(env, 'max_episode_length'):
                done = done or (env.curr_path_length >= env.max_episode_length)
```
hasattr(env, 'compute_reward'):判断这个环境里面是否有这个函数
添加一个手动奖励标志，将原来的compute_reward逻辑替换掉 self.manual_reward_value

将奖励放进干预类内，干预逻辑为空格控制`intervention_enabled`,当未启用干预时`aciton()`内直接返回action, `step()`函数直接执行最后的奖励逻辑
```python
        obs, rew, done, truncated, info = self.env.step(action)
        self.last_obs = obs
        info["gello_intervened"] = False
        manual_succeed = None  # 用于后续设置 info
        if self.manual_success_flag:
            rew = 1.0
            self.manual_success_flag = False  # 重置标志
            done = True  # 手动标记成功后结束 episode
            manual_succeed = True
            print(f"✅ 手动奖励已应用: reward={rew}, succeed=True")
        elif self.manual_failure_flag:
            rew = -1.0
            self.manual_failure_flag = False  # 重置标志
            done = True  # 手动标记失败后结束 episode
            manual_succeed = False
            print(f"❌ 手动奖励已应用: reward={rew}, succeed=False")
        base_env = self._get_cached_base_env()
        if base_env is not None and hasattr(base_env, 'curr_path_length'):
            base_env.curr_path_length += 1  # ✅ 明确使用底层环境
            if hasattr(base_env, 'max_episode_length'):
                done = done or (base_env.curr_path_length >= base_env.max_episode_length)
        info = {
            "succeed": rew > 0 if manual_succeed is None else manual_succeed,  # 🎯 手动标记优先
        }
        return obs, rew, done, truncated, info
```
当进行干预时直接进入`_threaded_step()`干预的奖励也写在里面




#### 数据流向图
/home/dungeon_master/conrft/test_md_docs/data_flow_visualization.md
该文档仅供参考，由ai生成，有些内容不对

## action chunk修改

config.py中 进行了action chunk的修改
[None为原本动作,4为最新的动作](./examples/experiments/a1x_pick_banana/config.py#L123) 


修改为4个chunk后,chunkwrapper的step()会循环四次,写在gellointervene类内的奖励逻辑容易被覆盖,因此进行修改
chunking.py
```python
        executed_actions = 0
        for i in range(act_exec_horizon):##原始代码
            obs, reward, done, trunc, info = self.env.step(action[i], *args)##原始代码
            self.current_obs.append(obs)##原始代码
            executed_actions = i + 1
            
            # 🎯 如果 episode 结束，立即停止
            if done or trunc:
                # 填充剩余的观测（保持 obs_horizon 大小一致）
                remaining = act_exec_horizon - i - 1
                if remaining > 0:
                    for _ in range(remaining):
                        self.current_obs.append(obs)
                break
        
        # 🎯 记录实际执行的动作数量
        info['executed_actions'] = executed_actions
        info['total_chunk_size'] = act_exec_horizon
        
        return (stack_obs(self.current_obs), reward, done, trunc, info)##原始代码
```

这是train_conrft_octo的修改
```python
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
```


```python
    if FLAGS.demo_path is not None and len(FLAGS.demo_path) > 0:
        try:
            with open(FLAGS.demo_path[0], "rb") as f:
                demo_transitions = pkl.load(f)
                if len(demo_transitions) > 0 and 'actions' in demo_transitions[0]:
                    sample_action = demo_transitions[0]['actions']
                    print_green(f"📊 使用 demo action 形状: {sample_action.shape}")
        except Exception as e:
            print_green(f"⚠️  无法加载 demo 数据，使用默认 action 形状: {e}")
```

### 固定夹爪逻辑
a1x_env.py `def interpolate_move(）`

1. 零动作检测
a1x_env.py  `step()`里面
2. 固定rxryrz
a1x_env.py 
```python
     self.reset_ee_rotation = self.curr_ee_pos_rot_gripper[3:6].copy()
        print(f"🔒 Reset时保存末端姿态 (rx, ry, rz): {np.rad2deg(self.reset_ee_rotation)} degrees")
        print(f"   后续step将保持该姿态不变（所有旋转delta强制为0）")

###############################

            # 🔒 固定旋转：将旋转delta设置为0，保持当前姿态不变
    scaled_action = scaled_action.copy()
    scaled_action[3:6] = 0.0  # 强制 drx, dry, drz = 0
``` 


  Host 6000pro
    HostName 10.97.216.208
    User e230112
    Port 34802
