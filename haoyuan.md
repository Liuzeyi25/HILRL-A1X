### Todo list

/home/dungeon_master/conrft/serl_robot_infra/franka_env/envs/a1x_env.py

172 def step(self, action: np.ndarray) -> tuple: 

# 默认（不过滤）
bash run_learner_hilserl.sh

# 策略 B: 随机丢弃
SAMPLING_STRATEGY=random_drop bash run_learner_hilserl.sh

# 策略 A: 空间过滤（带参数）
SAMPLING_STRATEGY=workspace_filtering \
  SAMPLING_KWARGS='{"x_range":[0.38,0.45],"y_range":[0.09,0.13]"z_range":[0.12,0.20]}' \
  bash run_learner_hilserl.sh
0.39724547, 0.10054581, 0.19048584
# 策略 C: PER（带参数）
SAMPLING_STRATEGY=per \
  SAMPLING_KWARGS='{"alpha":0.6,"beta":0.4}' \
  bash run_learner_hilserl.sh
  
  
 # 标准 SAC（默认）
bash run_learner_hilserl.sh

# 启用 Cov Actor Loss
USE_COV_ACTOR_LOSS=true bash run_learner_hilserl.sh

# 自定义超参
USE_COV_ACTOR_LOSS=true COV_K=16 COV_Q_HIGH=0.95 bash run_learner_hilserl.sh