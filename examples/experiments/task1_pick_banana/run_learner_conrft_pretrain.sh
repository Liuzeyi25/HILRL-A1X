rm -rf ~/.cache/jax_cache/
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=task1_pick_banana \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/task1_pick_banana/conrft\
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=./demo_data/task1_pick_banana_30_demos.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner 

# export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.75 && \
# export TF_FORCE_GPU_ALLOW_GROWTH=true && \
# export LD_LIBRARY_PATH=/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH && \
# export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" && \
# export XLA_FLAGS=--xla_gpu_autotune_level=0 && \
# export JAX_PLATFORMS=gpu && \
# python ../../train_conrft_octo.py "$@" \
#     --exp_name=task1_pick_banana \
#     --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/task1_pick_banana/conrft\
#     --q_weight=0.1 \
#     --bc_weight=1.0 \
#     --demo_path=./demo_data/task1_pick_banana_30_demos.pkl \
#     --pretrain_steps=20000 \
#     --debug=False \
#     --learner