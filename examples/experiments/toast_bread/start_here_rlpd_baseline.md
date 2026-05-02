## RLPD Baseline 实验步骤说明（toast_bread）

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

## 数据准备（Demo）
### 1) 采集 demo 数据
脚本路径：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/record_demos_octo_manual_new.py`

数据目录约定：
- Demo 根目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/`
- Demo 子目录（当天日期）：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/YYYYMMDD/`

执行：
```bash
python /home/dungeon_master/liuzeyi/HILRL-A1X/examples/record_demos_octo_manual_new.py \
    --exp_name toast_bread \
    --successes_needed 20 \
    --demo_data_subdir $(date +%Y%m%d)
```

### 2) 合并 demo 数据为单个 pkl
脚本路径：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/merge_trajectories.py`

输入/输出目录约定：
- 输入目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/YYYYMMDD/`
- 合并输出文件：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/YYYYMMDD/traj_XX.pkl`

执行：
```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X/examples

python merge_trajectories.py \
    /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/20260430 \
    /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/demo_data/20260430/traj_20.pkl
```

补充说明：
- 输出文件名可自定义，如 `traj_all.pkl`。
- 合并后的 pkl 会作为 RLPD 的 `--demo_path` 输入。

## RLPD 训练（Learner + Actor）
本任务使用 insert_block 的启动脚本进行 RLPD 训练（hilserl 版本）：

脚本路径：
- `examples/experiments/insert_block/run_learner_hilserl.sh`
- `examples/experiments/insert_block/run_actor_hilserl.sh`

路径约定（来自脚本内置默认）：
- Checkpoint 根目录：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/hilserl/<RUN_TAG>/`
- Demo 数据路径：`/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/demo_data/20260421/all_demos_merged.pkl`

运行前需要确认（在脚本内修改或通过环境变量覆盖）：
- `RUN_TAG`（默认 `hilserl-0422_1`）决定 checkpoint 目录。
- `run_learner_hilserl.sh` 内的 `--demo_path` 是否指向当前 demo 文件。

### 1) 启动 learner（终端 A）
```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block

bash run_learner_hilserl.sh
```

### 2) 启动 actor（终端 B）
```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block

bash run_actor_hilserl.sh
```

补充说明：
- Learner 与 Actor 需同时运行。
- 续训时可通过环境变量覆盖 `RUN_TAG`，例如：`RUN_TAG=hilserl-0502_1 bash run_learner_hilserl.sh`。
- actor 脚本默认连接 `localhost`，如跨机器需在脚本内添加 `--ip <learner_ip>`。

## 评估（可选）
当需要评估某个 checkpoint：
```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X/examples

python train_rlpd.py \
    --exp_name toast_bread \
    --eval \
    --checkpoint_path /home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/ckpt_rlpd \
    --eval_steps 10000 \
    --eval_episodes 10
```

补充说明：
- `--eval_steps` 必须指定已保存的 step。
- 评估时可以按 `s` 标记成功、按 `f` 提前 reset（失败）。
