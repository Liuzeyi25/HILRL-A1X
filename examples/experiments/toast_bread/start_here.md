## 实验步骤说明

## 仓库目录说明
- 仓库根目录：`/home/dungeon_master/liuzeyi/HILRL-A1X`
- 本实验目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/`

## 机器人与环境准备
1. 机器人通电，检查 CAN 盒连接、wrist camera 连接；实验过程中可能断连，需随时确认。

2. 打开终端，在仓库根目录启动机器人节点：
```bash
./a1_x_joint.bash
```

3. 启动 rostopic，确认反馈正常：
```bash
ros2 topic echo /hdas/feedback_arm
```
观察到持续读数，说明机器人与环境准备就绪。

4. 打开另一个终端，进入 conrft 虚拟环境。后续所有运行 python 脚本的终端都需先激活：
```bash
conda activate conrft
```

## 方法训练

### 1) 采集 demo 数据
脚本路径：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/record_demos_octo_manual_new.py`

数据目录约定：
- Demo 根目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/`
- Demo 子目录（当天日期）：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/YYYYMMDD/`

执行：
```bash
python /home/dungeon_master/liuzeyi/HILRL-A1X/examples/record_demos_octo_manual_new.py \
    --exp_name toast_bread \  # 实验名称（路径会自动映射到 experiments/toast_bread/）
    --successes_needed 20 \  # 需要采集的成功 demo 数量
    --demo_data_subdir $(date +%Y%m%d)  # demo 数据存放子目录
```

补充说明：
- Demo 数据会生成多个 `traj_*.pkl` 文件，存放在 `demo_data/YYYYMMDD/`。
- 采集中可按空格启用/禁用干预（依赖任务配置中的 `teleoperation_device`）。
- 若摄像头断开，需重启对应节点并重新运行采集脚本。

### 2) 训练 progress model
脚本路径：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/train_progress_model.py`

输出目录约定：
- Buffer 目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/buffer/`
- 训练输出目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/progress_model/`

执行：
```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X/examples

python train_progress_model.py \
    --demo_dir experiments/toast_bread/demo_data/20260430 \  # demo 数据目录（日期需替换）
    --buffer_dir experiments/toast_bread/buffer \           # 输出 buffer 目录
    --output_dir experiments/toast_bread/progress_model \   # 模型输出目录
    --hidden_dim 128 \
    --epochs 300 \
    --device cpu
```

补充说明：
- `--demo_dir` 需指向实际采集日期的子目录。
- 若 GPU 可用，可将 `--device` 改为 `cuda`。

### 3) 合并 demo 数据为单个 pkl
脚本路径：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/merge_trajectories.py`

输入/输出目录约定：
- 输入目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/YYYYMMDD/`
- 合并输出文件：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/YYYYMMDD/traj_XX.pkl`

执行：
```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X/examples

python merge_trajectories.py \
    /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/20260430 \  # demo 数据目录
    /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/20260430/traj_20.pkl  # 输出文件
```

补充说明：
- 输出文件名可自定义，如 `traj_all.pkl`。
- 合并后的单文件通常用于后续训练或离线评估。

### 4) 训练 multi-bc agent
脚本目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/`

目录内脚本：
- `run_learner_zeyi.sh`
- `run_actor_zeyi.sh`

执行前需要确认参数配置：
- `run_learner_zeyi.sh` 内的 `RUN_TAG` 需与当前实验名称一致（如 `0502_2_multi-bc`）。
- `run_actor_zeyi.sh` 内的 `RUN_TAG` 也需与 learner 中一致。
- checkpoint_path 是模型节点的保存路径，需确认与实验名称一致。
- demo_path 需指向合并后的 demo 数据文件（如 `demo_data/YYYYMMDD/traj_20.pkl`）。
- progress_model_path使用对应demo数据训练的progress模型，请确保 `run_learner_zeyi.sh` 与 `run_actor_zeyi.sh` 内的路径一致。
- state_stats_path 需指向 demo 数据对应的 state_stats 文件（如 `demo_data/YYYYMMDD/state_stats.npz`）。


执行：
```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread

# 先启动 learner
bash run_learner_zeyi.sh

# 另开终端（同样激活 conrft），再启动 actor
bash run_actor_zeyi.sh
```



补充说明：
- learner 与 actor 需同时运行。
- 若使用新数据或新模型，请确保 `run_learner_zeyi.sh` 与 `run_actor_zeyi.sh` 内的路径与实验名称一致。
