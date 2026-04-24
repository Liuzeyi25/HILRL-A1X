python scripts/evaluate_q_values.py \
    --checkpoint_a "examples/experiments/insert_block/experiments/insert_block/checkpoints_hil/0423_2_multi-bc/checkpoint_12000"  \
    --checkpoint_b "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000"  \
    --trajectory_path "examples/experiments/insert_block/experiments/insert_block/checkpoints_hil/0423_2_multi-bc/buffer/transitions_1774.pkl" \
    --exp_name insert_block \
    --output_dir ./examples/experiments/insert_block/q_eval_multi_model 
    