
# 训练分为四个步骤：
1. 采集demo data：
```bash
python /home/dungeon_master/liuzeyi/HILRL-A1X/examples/record_demos_octo_manual_new.py \
    --exp_name <YOUR_EXP_NAME> \
    --successes_needed 20 \
    --demo_data_subdir $(date +%Y%m%d)
```

2. 训练progress model：
```bash
cd examples

python train_progress_model.py \
    --demo_dir  experiments/a1x_pick_banana/demo_data/20260429 \
    --buffer_dir experiments/a1x_pick_banana/buffer \
    --output_dir experiments/a1x_pick_banana/progress_model \
    --hidden_dim 128 \
    --epochs 300 \
    --device cpu
```

3. 合并demo data为一个pkl文件：脚本在 examples/merge_trajectories.py
```bash
cd /home/dungeon_master/conrft/examples/
   python merge_trajectories.py \
       //home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/demo_data/20260429 \
       //home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/demo_data/20260429/traj_20.pkl 