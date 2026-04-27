# python ./examples/train_progress_model.py\
#     --demo_dir  examples/experiments/insert_block/demo_data/20260421 \
#     --buffer_dir examples/experiments/insert_block/buffer \
#     --output_dir examples/experiments/insert_block/progress_model \
#     --hidden_dim 128 \
#     --epochs 300 \
#     --device cpu


# 在 HILRL-A1X 根目录下运行,每个实验可以新建一个命令
python ./examples/train_progress_model.py\
    --demo_dir  ./examples/experiments/a1x_pick_banana/demo_data/20260427 \
    --buffer_dir examples/experiments/a1x_pick_banana/buffer \
    --output_dir examples/experiments/a1x_pick_banana/progress_model \
    --hidden_dim 128 \
    --epochs 300 \
    --device cpu