帮我看一下是我设计的算法包含的三个模块，逻辑中是如何处理第七维度夹爪的训练的，由于原始的hilserl的前六维度和第七维度是分开训练的，用了两个critic网络，所以我的算法逻辑是不是跟这个有冲突。
目前我在启动训练代码的时候，learner端运行会报错：
re/lift.py:310: RuntimeWarning: kwargs are not supported in vmap, so "train" is(are) ignored
  warnings.warn(msg.format(name, ', '.join(kwargs.keys())), RuntimeWarning)
learner_bc:   0%|                                   | 0/1000000 [00:01<?, ?it/s]
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/../../train_rlpd_hil_bc.py", line 803, in <module>
  File "/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/../../train_rlpd_hil_bc.py", line 771, in main
    wandb_logger=wandb_logger,
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/../../train_rlpd_hil_bc.py", line 493, in learner_bc
    agent, _ = agent.update_with_correction_bc(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 802, in update_with_correction_bc
    new_state, info = self.state.apply_loss_fns(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/common/common.py", line 210, in apply_loss_fns
    grads_and_aux = jax.tree_util.tree_map(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/common/common.py", line 211, in <lambda>
    lambda loss_fn, rng: jax.grad(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 259, in grasp_critic_loss_fn
    grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
  File "/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/flax/core/frozen_dict.py", line 70, in __getitem__
    v = self._dict[key]
KeyError: 'grasp_penalty'
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/../../train_rlpd_hil_bc.py", line 803, in <module>
  File "/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/../../train_rlpd_hil_bc.py", line 771, in main
    wandb_logger=wandb_logger,
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/../../train_rlpd_hil_bc.py", line 493, in learner_bc
    agent, _ = agent.update_with_correction_bc(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 802, in update_with_correction_bc
    new_state, info = self.state.apply_loss_fns(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/common/common.py", line 210, in apply_loss_fns
    grads_and_aux = jax.tree_util.tree_map(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/common/common.py", line 211, in <lambda>
    lambda loss_fn, rng: jax.grad(
  File "/home/dungeon_master/liuzeyi/HILRL-A1X/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 259, in grasp_critic_loss_fn
    grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
  File "/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/flax/core/frozen_dict.py", line 70, in __getitem__
    v = self._dict[key]
KeyError: 'grasp_penalty'