JAX_PLATFORMS=cpu python /home/wenkai001/hil-serl/examples/train_rlpd.py \
  --exp_name="push_cube" \
  --learner=True \
  --demo_path=/home/wenkai001/Hil-serl/classifier_data/push_cube_50_success_transitions_2025-06-23_20-13-33.pkl \
  --demo_path=/home/wenkai001/Hil-serl/classifier_data/push_cube_109_failure_transitions_2025-06-23_20-13-33.pkl \
  --eval_n_trajs=1 \
  --checkpoint_path=/home/wenkai001/Hil-serl/sac_ckpt