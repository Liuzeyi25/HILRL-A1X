bash: /home/dungeon_master/miniconda3/envs/conrft/lib/libtinfo.so.6: no version information available (required by bash)
/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/../../train_conrft_octo.py:70: DeprecationWarning: jax.sharding.PositionalSharding is deprecated. Use jax.NamedSharding instead.
  sharding = jax.sharding.PositionalSharding(devices)
/bin/zsh: /home/dungeon_master/miniconda3/envs/conrft/lib/libtinfo.so.6: no version information available (required by /bin/zsh)
[INFO] [1770559977.834243924] [a1x_serl_node]: FK solver initialized (DOF: 8, EE frame: gripper_link)
[INFO] [1770559977.834871746] [a1x_serl_node]: 🚀 A1XRobotZMQNode initialized with dual sockets:
[INFO] [1770559977.834983066] [a1x_serl_node]:    Command port: 6100
[INFO] [1770559977.835077737] [a1x_serl_node]:    State port: 6101
Saving videos!
Initializing A1_X robot...
Starting ROS2 node subprocess...
Connecting to ROS2 node on ports 6100 (cmd) and 6101 (state)...
🚀 [A1XRobotBridge] 双Socket模式初始化
   Command port: 6100
   State port: 6101
Waiting for joint states from A1_X robot...
A1_X robot connected successfully
Joint names: ['arm']
[RSCapture] Hardware reset RealSense...
[RSCapture] Hardware reset RealSense...
Initialized A1_X Environment
📟 Initializing Gello Agent (device reader only)...
attempting to connect to port: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0
Attempting to initialize Dynamixel driver (attempt 1/3)
Successfully initialized Dynamixel driver on /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0
   ✅ Gello Agent created successfully
🎹 键盘监听器已启动 (监听 's'/'f' 键和空格键)
   ✅ GelloFollower initialized for bidirectional control
✅ GelloIntervention initialized
   - Agent: GelloAgent
   - Control rate: 500 Hz
   - Bimanual: False
   - 🚀 双线程控制: 启用
   - 快速干预模式: 启用
   - 🔄 同步验证: 最大重试=3, 误差阈值=0.150 rad
   🎯 评估模式: 干预已禁用
   ⚠️  Gello 设备将被忽略，只使用 Agent 策略
   🔄 双向控制已启用：Reset 时 Gello 跟随机器人
🎯 GelloIntervention 已创建（评估模式：干预已禁用）
Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]Fetching 6 files: 100%|██████████| 6/6 [00:00<00:00, 29606.85it/s]
I0208 22:13:09.900936 123160943875904 octo_module.py:219] repeating task tokens at each timestep to perform cross-modal attention
I0208 22:13:14.025161 123160943875904 checkpointer.py:164] Restoring item from /home/dungeon_master/.cache/huggingface/hub/models--rail-berkeley--octo-small-1.5/snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58/300000/default.
I0208 22:13:14.345588 123160943875904 checkpointer.py:166] Finished restoring checkpoint from /home/dungeon_master/.cache/huggingface/hub/models--rail-berkeley--octo-small-1.5/snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58/300000/default.
I0208 22:13:15.369024 123160943875904 schedule.py:75] A polynomial schedule was set with a non-positive `transition_steps` value; this results in a constant schedule with value `init_value`.
I0208 22:13:15.369201 123160943875904 schedule.py:75] A polynomial schedule was set with a non-positive `transition_steps` value; this results in a constant schedule with value `init_value`.
W0208 22:13:23.031311 123160943875904 tokenizers.py:25] No pad_mask_dict found. Nothing will be masked.
W0208 22:13:24.832504 123160943875904 tokenizers.py:25] No pad_mask_dict found. Nothing will be masked.
I0208 22:13:25.217207 123160943875904 octo_module.py:219] repeating task tokens at each timestep to perform cross-modal attention
/home/dungeon_master/miniconda3/envs/conrft/lib/python3.10/site-packages/flax/core/lift.py:310: RuntimeWarning: kwargs are not supported in vmap, so "train" is(are) ignored
  warnings.warn(msg.format(name, ', '.join(kwargs.keys())), RuntimeWarning)
I0208 22:13:42.902847 123160943875904 checkpoints.py:1108] Restoring legacy Flax checkpoint from /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/conrft/0208/checkpoint_40000
warning, comm failed: -3002
warning, comm failed: -3002
warning, comm failed: -3001
warning, comm failed: -3002
warning, comm failed: -3001
warning, comm failed: -3002
warning, comm failed: -3002
warning, comm failed: -3002
warning, comm failed: -3002
The ResNet-10 weights already exist at '/home/dungeon_master/.serl/resnet10_params.pkl'.
Loaded 5.418792M parameters from ResNet-10 pretrained on ImageNet-1K
replaced conv_init in pretrained_encoder
replaced norm_init in pretrained_encoder
replaced ResNetBlock_0 in pretrained_encoder
replaced ResNetBlock_1 in pretrained_encoder
replaced ResNetBlock_2 in pretrained_encoder
replaced ResNetBlock_3 in pretrained_encoder
warning, comm failed: -3002
warning, comm failed: -3002
warning, comm failed: -3001
[92m Loaded checkpoint at step 40000 for evaluation.[00m
[92m starting actor loop[00m
[92m 🎯 评估模式：跳过 learner 连接[00m
CURRENT JOINTS: [ 0.0687234   2.30234043 -1.35574468  0.26489362  0.11148936  0.05468085
  0.        ]
Interpolating move TO [-1.5310e-02  1.8255e+00 -1.1390e+00  8.6800e-01 -5.3000e-02 -1.0300e-01
  1.0000e+02] over 20 steps (2.0s).
warning, comm failed: -3002
FINAL JOINTS: [-1.53191489e-02  1.82808511e+00 -1.13914894e+00  8.62978723e-01
 -5.27659574e-02 -1.01914894e-01  8.63168183e+01]
⚠️  Warning: Large position error detected!
   Position error: [9.14893617e-06 2.58510638e-03 1.48936170e-04 5.02127660e-03
 2.34042553e-04 1.08510638e-03 1.36831817e+01]
   Max error: 13.6832 rad
🎯 评估模式: 跳过 Gello 同步
  0%|          | 0/1000000 [00:00<?, ?it/s]W0208 22:13:51.541828 123160943875904 tokenizers.py:25] No pad_mask_dict found. Nothing will be masked.
W0208 22:13:51.559761 123160943875904 tokenizers.py:25] No pad_mask_dict found. Nothing will be masked.
I0208 22:13:51.561933 123160943875904 octo_module.py:219] repeating task tokens at each timestep to perform cross-modal attention
[INFO] [1770560038.258736471] [a1x_serl_node]: Published EEF command: pos=[0.2641, -0.0062, 0.1914], quat=[-0.049, 0.695, 0.010, 0.717]
  0%|          | 1/1000000 [00:07<2073:56:59,  7.47s/it][INFO] [1770560038.444822355] [a1x_serl_node]: Published EEF command: pos=[0.2668, -0.0026, 0.1900], quat=[-0.045, 0.696, 0.012, 0.717]
  0%|          | 2/1000000 [00:07<884:22:05,  3.18s/it] [INFO] [1770560038.598929103] [a1x_serl_node]: Published EEF command: pos=[0.2681, -0.0066, 0.1887], quat=[-0.045, 0.691, 0.009, 0.721]
  0%|          | 3/1000000 [00:07<500:05:58,  1.80s/it][INFO] [1770560038.752306909] [a1x_serl_node]: Published EEF command: pos=[0.2704, -0.0020, 0.1907], quat=[-0.040, 0.689, 0.007, 0.724]
  0%|          | 4/1000000 [00:07<319:32:20,  1.15s/it][INFO] [1770560038.906154067] [a1x_serl_node]: Published EEF command: pos=[0.2717, 0.0005, 0.1858], quat=[-0.040, 0.689, 0.011, 0.723]
  0%|          | 5/1000000 [00:08<227:59:04,  1.22it/s][INFO] [1770560039.143215320] [a1x_serl_node]: Published EEF command: pos=[0.2710, 0.0001, 0.1908], quat=[-0.032, 0.683, 0.008, 0.729]
  0%|          | 6/1000000 [00:08<165:12:45,  1.68it/s][INFO] [1770560039.299507156] [a1x_serl_node]: Published EEF command: pos=[0.2715, -0.0008, 0.1888], quat=[-0.037, 0.688, 0.013, 0.724]
  0%|          | 7/1000000 [00:08<125:14:42,  2.22it/s][INFO] [1770560039.452222460] [a1x_serl_node]: Published EEF command: pos=[0.2781, 0.0009, 0.1919], quat=[-0.028, 0.681, 0.008, 0.732]
  0%|          | 8/1000000 [00:08<98:53:19,  2.81it/s] [INFO] [1770560039.606172699] [a1x_serl_node]: Published EEF command: pos=[0.2756, -0.0064, 0.1921], quat=[-0.032, 0.685, 0.010, 0.728]
  0%|          | 9/1000000 [00:08<81:18:00,  3.42it/s][INFO] [1770560039.759300675] [a1x_serl_node]: Published EEF command: pos=[0.2814, -0.0023, 0.1948], quat=[-0.021, 0.676, 0.006, 0.736]
  0%|          | 10/1000000 [00:08<69:20:59,  4.01it/s][INFO] [1770560039.912660171] [a1x_serl_node]: Published EEF command: pos=[0.2807, -0.0012, 0.1932], quat=[-0.024, 0.678, 0.008, 0.735]
  0%|          | 11/1000000 [00:09<61:08:52,  4.54it/s][INFO] [1770560040.067032071] [a1x_serl_node]: Published EEF command: pos=[0.2810, 0.0026, 0.1928], quat=[-0.013, 0.671, 0.004, 0.741]
  0%|          | 12/1000000 [00:09<55:41:46,  4.99it/s][INFO] [1770560040.228347984] [a1x_serl_node]: Published EEF command: pos=[0.2803, 0.0025, 0.1926], quat=[-0.017, 0.672, 0.006, 0.740]
  0%|          | 13/1000000 [00:09<52:17:51,  5.31it/s][INFO] [1770560040.381710721] [a1x_serl_node]: Published EEF command: pos=[0.2842, 0.0031, 0.1946], quat=[-0.002, 0.666, 0.001, 0.746]
  0%|          | 14/1000000 [00:09<49:24:08,  5.62it/s][INFO] [1770560040.534462936] [a1x_serl_node]: Published EEF command: pos=[0.2851, -0.0014, 0.1932], quat=[-0.006, 0.671, 0.000, 0.742]
  0%|          | 15/1000000 [00:09<47:20:08,  5.87it/s][INFO] [1770560040.689559368] [a1x_serl_node]: Published EEF command: pos=[0.2884, -0.0001, 0.1910], quat=[0.004, 0.660, -0.004, 0.752]
  0%|          | 16/1000000 [00:09<46:00:21,  6.04it/s][INFO] [1770560040.843115526] [a1x_serl_node]: Published EEF command: pos=[0.2841, 0.0036, 0.1932], quat=[0.001, 0.664, -0.001, 0.748]
  0%|          | 17/1000000 [00:10<44:59:53,  6.17it/s][INFO] [1770560040.996720953] [a1x_serl_node]: Published EEF command: pos=[0.2907, -0.0016, 0.1910], quat=[0.010, 0.657, -0.005, 0.754]
  0%|          | 18/1000000 [00:10<44:16:54,  6.27it/s][INFO] [1770560041.151095554] [a1x_serl_node]: Published EEF command: pos=[0.2875, -0.0032, 0.1909], quat=[0.009, 0.663, -0.007, 0.749]
  0%|          | 19/1000000 [00:10<43:53:08,  6.33it/s][INFO] [1770560041.307810552] [a1x_serl_node]: Published EEF command: pos=[0.2910, -0.0009, 0.1929], quat=[0.015, 0.655, -0.008, 0.756]
  0%|          | 20/1000000 [00:10<43:45:52,  6.35it/s][INFO] [1770560041.461591741] [a1x_serl_node]: Published EEF command: pos=[0.2908, -0.0028, 0.1913], quat=[0.013, 0.656, -0.009, 0.754]
  0%|          | 21/1000000 [00:10<43:26:22,  6.39it/s][INFO] [1770560041.615159129] [a1x_serl_node]: Published EEF command: pos=[0.2901, -0.0031, 0.1920], quat=[0.020, 0.657, -0.012, 0.754]
  0%|          | 22/1000000 [00:10<43:08:46,  6.44it/s][INFO] [1770560041.768685097] [a1x_serl_node]: Published EEF command: pos=[0.2926, -0.0075, 0.1928], quat=[0.019, 0.656, -0.016, 0.754]
  0%|          | 23/1000000 [00:10<43:07:20,  6.44it/s][INFO] [1770560041.928660716] [a1x_serl_node]: Published EEF command: pos=[0.2953, -0.0070, 0.1911], quat=[0.028, 0.653, -0.018, 0.757]
  0%|          | 24/1000000 [00:11<43:28:55,  6.39it/s][INFO] [1770560042.086585888] [a1x_serl_node]: Published EEF command: pos=[0.2899, -0.0044, 0.1916], quat=[0.025, 0.655, -0.020, 0.755]
  0%|          | 25/1000000 [00:11<43:36:58,  6.37it/s][INFO] [1770560042.239873026] [a1x_serl_node]: Published EEF command: pos=[0.2955, -0.0099, 0.1912], quat=[0.032, 0.651, -0.022, 0.758]
  0%|          | 26/1000000 [00:11<43:13:46,  6.43it/s]warning, comm failed: -3002
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00176426 0.0033601  0.00157236], rot=[ 0.00090255 -0.0111597  -0.00242921], gripper: 0.863 -> 0.867 (86.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 1, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00409328 0.00683782 0.00012523], rot=[ 0.00619255 -0.00625157 -0.0037971 ], gripper: 0.863 -> 0.870 (87.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=8.0mm
Step done: False, reward: False, path length: 2, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00366287  0.00252305 -0.002067  ], rot=[ 0.00304919 -0.00923022 -0.00716753], gripper: 0.863 -> 0.867 (86.7mm)
⏱️  ✓ 执行耗时=99ms, 误差=4.9mm
Step done: False, reward: False, path length: 3, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00443568 0.00445638 0.00013255], rot=[ 0.00298028 -0.01180053 -0.01338326], gripper: 0.863 -> 0.876 (87.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.3mm
Step done: False, reward: False, path length: 4, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.0034165   0.00763338 -0.00485526], rot=[ 0.0079365  -0.00070593 -0.00737373], gripper: 0.863 -> 0.872 (87.2mm)
⏱️  ✓ 执行耗时=182ms, 误差=7.9mm
Step done: False, reward: False, path length: 5, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00042988  0.00498049 -0.00017929], rot=[ 0.01056289 -0.00853555 -0.01056859], gripper: 0.863 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.0mm
Step done: False, reward: False, path length: 6, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00035121  0.00287636 -0.00026464], rot=[ 0.00649371  0.00636528 -0.00187795], gripper: 0.862 -> 0.863 (86.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.9mm
Step done: False, reward: False, path length: 7, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00604393 0.0034254  0.00185581], rot=[ 0.00487135 -0.01006592 -0.00630875], gripper: 0.862 -> 0.866 (86.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.2mm
Step done: False, reward: False, path length: 8, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00308622 -0.00406097  0.00200384], rot=[ 0.00288292  0.00341086 -0.00936417], gripper: 0.862 -> 0.865 (86.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.6mm
Step done: False, reward: False, path length: 9, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00584436 -0.00085976  0.00391634], rot=[ 0.00649681 -0.00745403 -0.01302108], gripper: 0.862 -> 0.868 (86.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.1mm
Step done: False, reward: False, path length: 10, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00565575 0.00182981 0.0023503 ], rot=[ 0.00748449 -0.00829442 -0.01664065], gripper: 0.862 -> 0.873 (87.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.4mm
Step done: False, reward: False, path length: 11, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00236609 0.0052724  0.00075923], rot=[ 0.01191467 -0.008941   -0.01412766], gripper: 0.862 -> 0.865 (86.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.8mm
Step done: False, reward: False, path length: 12, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00139632 0.00516068 0.00028198], rot=[ 0.0084482  -0.00608741 -0.01339061], gripper: 0.862 -> 0.867 (86.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 13, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00294885 0.00371012 0.00184561], rot=[ 0.01183933 -0.01039036 -0.0207046 ], gripper: 0.862 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.1mm
Step done: False, reward: False, path length: 14, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00398147 -0.00120917  0.00026624], rot=[ 0.00535543  0.00101977 -0.02176351], gripper: 0.862 -> 0.868 (86.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.2mm
Step done: False, reward: False, path length: 15, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00445885 -0.00090736 -0.00296296], rot=[ 0.00377729 -0.01429736 -0.01885479], gripper: 0.862 -> 0.867 (86.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 16, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00022     0.00482543 -0.00025025], rot=[ 0.0089049  -0.00592661 -0.01446082], gripper: 0.862 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.8mm
Step done: False, reward: False, path length: 17, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00280573 -0.00050433 -0.00272493], rot=[ 0.00913613 -0.00590648 -0.01554502], gripper: 0.862 -> 0.863 (86.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 18, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00154436 -0.00349128 -0.00278401], rot=[ 0.00365915 -0.00036068 -0.02069617], gripper: 0.862 -> 0.868 (86.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.7mm
Step done: False, reward: False, path length: 19, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00112303 -0.00061715 -0.00075289], rot=[ 0.00836453 -0.00235094 -0.02043336], gripper: 0.862 -> 0.863 (86.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.5mm
Step done: False, reward: False, path length: 20, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00246653 -0.00056407 -0.001601  ], rot=[ 0.00467875 -0.01011571 -0.01147289], gripper: 0.862 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 21, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00116662 -0.0010748  -0.0011472 ], rot=[ 0.00302488  0.00657135 -0.01564465], gripper: 0.862 -> 0.867 (86.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 22, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00197017 -0.00424495 -0.00061995], rot=[-0.00074149  0.00127908 -0.01896448], gripper: 0.862 -> 0.871 (87.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.7mm
Step done: False, reward: False, path length: 23, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00380433 -0.00341709 -0.00195298], rot=[ 0.00662099 -0.00206473 -0.02315608], gripper: 0.862 -> 0.871 (87.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.5mm
Step done: False, reward: False, path length: 24, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00235774  0.00194629 -0.00162851], rot=[ 0.00469579  0.00506033 -0.01465091], gripper: 0.862 -> 0.869 (86.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.5mm
Step done: False, reward: False, path length: 25, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00091704 -0.00333052 -0.0016817 ], rot=[ 0.00387214  0.00299511 -0.01518673], gripper: 0.862 -> 0.870 (87.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 26, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560042.392257540] [a1x_serl_node]: Published EEF command: pos=[0.2966, -0.0106, 0.1929], quat=[0.032, 0.654, -0.023, 0.755]
  0%|          | 27/1000000 [00:11<42:57:56,  6.46it/s][INFO] [1770560042.544720575] [a1x_serl_node]: Published EEF command: pos=[0.2957, -0.0125, 0.1925], quat=[0.039, 0.651, -0.026, 0.758]
  0%|          | 28/1000000 [00:11<42:48:18,  6.49it/s][INFO] [1770560042.698284603] [a1x_serl_node]: Published EEF command: pos=[0.2951, -0.0117, 0.1904], quat=[0.039, 0.651, -0.026, 0.758]
  0%|          | 29/1000000 [00:11<42:49:02,  6.49it/s][INFO] [1770560042.852589874] [a1x_serl_node]: Published EEF command: pos=[0.2994, -0.0161, 0.1901], quat=[0.044, 0.645, -0.031, 0.762]
  0%|          | 30/1000000 [00:12<42:46:57,  6.49it/s][INFO] [1770560043.004848578] [a1x_serl_node]: Published EEF command: pos=[0.2968, -0.0128, 0.1892], quat=[0.044, 0.649, -0.027, 0.759]
  0%|          | 31/1000000 [00:12<42:37:02,  6.52it/s][INFO] [1770560043.157758244] [a1x_serl_node]: Published EEF command: pos=[0.2988, -0.0146, 0.1893], quat=[0.043, 0.643, -0.031, 0.764]
  0%|          | 32/1000000 [00:12<42:34:48,  6.52it/s][INFO] [1770560043.311079582] [a1x_serl_node]: Published EEF command: pos=[0.2989, -0.0099, 0.1886], quat=[0.045, 0.642, -0.026, 0.765]
  0%|          | 33/1000000 [00:12<42:35:28,  6.52it/s][INFO] [1770560043.467768841] [a1x_serl_node]: Published EEF command: pos=[0.3040, -0.0145, 0.1910], quat=[0.045, 0.637, -0.029, 0.769]
  0%|          | 34/1000000 [00:12<42:51:34,  6.48it/s][INFO] [1770560043.622033782] [a1x_serl_node]: Published EEF command: pos=[0.3025, -0.0147, 0.1880], quat=[0.047, 0.636, -0.027, 0.770]
  0%|          | 35/1000000 [00:12<42:52:20,  6.48it/s][INFO] [1770560043.774644807] [a1x_serl_node]: Published EEF command: pos=[0.3030, -0.0111, 0.1912], quat=[0.046, 0.630, -0.027, 0.775]
  0%|          | 36/1000000 [00:12<42:43:23,  6.50it/s][INFO] [1770560043.927374773] [a1x_serl_node]: Published EEF command: pos=[0.3067, -0.0122, 0.1898], quat=[0.051, 0.629, -0.028, 0.775]
  0%|          | 37/1000000 [00:13<42:39:06,  6.51it/s][INFO] [1770560044.080964892] [a1x_serl_node]: Published EEF command: pos=[0.3061, -0.0135, 0.1923], quat=[0.052, 0.624, -0.030, 0.779]
  0%|          | 38/1000000 [00:13<42:36:27,  6.52it/s][INFO] [1770560044.232463024] [a1x_serl_node]: Published EEF command: pos=[0.3113, -0.0169, 0.1939], quat=[0.049, 0.620, -0.033, 0.783]
  0%|          | 39/1000000 [00:13<42:30:04,  6.54it/s][INFO] [1770560044.386916066] [a1x_serl_node]: Published EEF command: pos=[0.3098, -0.0178, 0.1924], quat=[0.052, 0.621, -0.033, 0.781]
  0%|          | 40/1000000 [00:13<42:35:44,  6.52it/s][INFO] [1770560044.540157404] [a1x_serl_node]: Published EEF command: pos=[0.3149, -0.0162, 0.1955], quat=[0.049, 0.616, -0.033, 0.786]
  0%|          | 41/1000000 [00:13<42:33:55,  6.53it/s][INFO] [1770560044.696068081] [a1x_serl_node]: Published EEF command: pos=[0.3124, -0.0162, 0.1952], quat=[0.052, 0.617, -0.034, 0.784]
  0%|          | 42/1000000 [00:13<42:47:55,  6.49it/s][INFO] [1770560044.849190168] [a1x_serl_node]: Published EEF command: pos=[0.3169, -0.0138, 0.1930], quat=[0.051, 0.611, -0.034, 0.789]
  0%|          | 43/1000000 [00:14<42:44:32,  6.50it/s][INFO] [1770560045.002236925] [a1x_serl_node]: Published EEF command: pos=[0.3124, -0.0145, 0.1954], quat=[0.053, 0.612, -0.034, 0.789]
  0%|          | 44/1000000 [00:14<42:39:09,  6.51it/s][INFO] [1770560045.156339387] [a1x_serl_node]: Published EEF command: pos=[0.3131, -0.0153, 0.1920], quat=[0.050, 0.613, -0.033, 0.788]
  0%|          | 45/1000000 [00:14<42:42:30,  6.50it/s][INFO] [1770560045.309505035] [a1x_serl_node]: Published EEF command: pos=[0.3184, -0.0156, 0.1962], quat=[0.053, 0.614, -0.034, 0.787]
  0%|          | 46/1000000 [00:14<42:38:09,  6.51it/s][INFO] [1770560045.463529935] [a1x_serl_node]: Published EEF command: pos=[0.3171, -0.0143, 0.1963], quat=[0.052, 0.614, -0.035, 0.787]
  0%|          | 47/1000000 [00:14<42:40:54,  6.51it/s][INFO] [1770560045.616459763] [a1x_serl_node]: Published EEF command: pos=[0.3150, -0.0136, 0.1969], quat=[0.055, 0.612, -0.035, 0.788]
  0%|          | 48/1000000 [00:14<42:52:08,  6.48it/s][INFO] [1770560045.773924085] [a1x_serl_node]: Published EEF command: pos=[0.3136, -0.0149, 0.1922], quat=[0.052, 0.613, -0.034, 0.787]
  0%|          | 49/1000000 [00:14<42:51:44,  6.48it/s][INFO] [1770560045.927202533] [a1x_serl_node]: Published EEF command: pos=[0.3165, -0.0140, 0.1973], quat=[0.058, 0.612, -0.037, 0.788]
  0%|          | 50/1000000 [00:15<43:01:53,  6.45it/s][INFO] [1770560046.083418852] [a1x_serl_node]: Published EEF command: pos=[0.3169, -0.0129, 0.1924], quat=[0.051, 0.611, -0.035, 0.789]
  0%|          | 51/1000000 [00:15<42:55:57,  6.47it/s][INFO] [1770560046.236979791] [a1x_serl_node]: Published EEF command: pos=[0.3152, -0.0104, 0.1937], quat=[0.057, 0.610, -0.037, 0.789]
  0%|          | 52/1000000 [00:15<42:52:28,  6.48it/s]EEF delta: pos=[ 0.00407941 -0.00391679  0.00075194], rot=[ 0.00680745  0.00104833 -0.01510797], gripper: 0.862 -> 0.867 (86.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.7mm
Step done: False, reward: False, path length: 27, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.000957   -0.00420225  0.00035158], rot=[ 0.00670153  0.00262317 -0.01952159], gripper: 0.862 -> 0.867 (86.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.3mm
Step done: False, reward: False, path length: 28, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00018188 -0.00268698 -0.00175819], rot=[ 0.00794745  0.00082812 -0.01534372], gripper: 0.862 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.2mm
Step done: False, reward: False, path length: 29, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00339435 -0.00531419 -0.00192592], rot=[ 0.00250276 -0.00743077 -0.01609274], gripper: 0.862 -> 0.863 (86.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.6mm
Step done: False, reward: False, path length: 30, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00116236 -0.00180112 -0.00260735], rot=[ 0.00707117 -0.00089377 -0.01060703], gripper: 0.862 -> 0.861 (86.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.4mm
Step done: False, reward: False, path length: 31, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 1.20956247e-05 -5.78992767e-04 -2.49443855e-03], rot=[ 0.00056637 -0.00276185 -0.0021636 ], gripper: 0.862 -> 0.862 (86.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 32, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00073722  0.00315294 -0.00292884], rot=[ 0.00326269 -0.00888107  0.00151078], gripper: 0.862 -> 0.859 (85.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.4mm
Step done: False, reward: False, path length: 33, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00380152 -0.00039202 -0.00032305], rot=[ 0.00450197 -0.01240738 -0.00054507], gripper: 0.862 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 34, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00228029 -0.00238372 -0.00318945], rot=[ 0.00359481 -0.0163643  -0.00314646], gripper: 0.862 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.6mm
Step done: False, reward: False, path length: 35, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00037923  0.00254098 -0.00117376], rot=[ 0.00459916 -0.01487588 -0.0009617 ], gripper: 0.862 -> 0.860 (86.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.8mm
Step done: False, reward: False, path length: 36, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00214106  0.00143752 -0.00227252], rot=[ 0.00676855 -0.01197127 -0.00751717], gripper: 0.862 -> 0.866 (86.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.4mm
Step done: False, reward: False, path length: 37, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00086213 -0.00052968 -0.00047005], rot=[ 0.0033971  -0.0184052  -0.01264712], gripper: 0.862 -> 0.864 (86.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.1mm
Step done: False, reward: False, path length: 38, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00389761 -0.00387253  0.00062245], rot=[-0.00896475 -0.01970717 -0.00678201], gripper: 0.862 -> 0.858 (85.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.5mm
Step done: False, reward: False, path length: 39, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00209572 -0.00354837 -0.00192103], rot=[-0.00162538 -0.01030079 -0.00667371], gripper: 0.862 -> 0.862 (86.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.5mm
Step done: False, reward: False, path length: 40, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00364017 -0.00034946 -0.00038346], rot=[-0.00197198 -0.00556352 -0.000978  ], gripper: 0.862 -> 0.863 (86.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 41, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 1.11583341e-03  8.41293950e-05 -3.40071972e-04], rot=[ 0.00038307 -0.00380596 -0.00411721], gripper: 0.862 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.1mm
Step done: False, reward: False, path length: 42, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00337285  0.00275982 -0.00307457], rot=[ 0.00238507 -0.00891505 -0.00230899], gripper: 0.858 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.3mm
Step done: False, reward: False, path length: 43, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00080958  0.00177804 -0.00086294], rot=[ 0.00383059 -0.00899408 -0.00592941], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.1mm
Step done: False, reward: False, path length: 44, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00329023  0.00062077 -0.00396257], rot=[-0.00014774  0.00606299  0.00235448], gripper: 0.854 -> 0.850 (85.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 45, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00330273 0.00018734 0.00017265], rot=[ 0.00085035  0.00308251 -0.0024392 ], gripper: 0.854 -> 0.851 (85.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.3mm
Step done: False, reward: False, path length: 46, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.0010759  0.0016206  0.00114524], rot=[ 0.00029861  0.00620019 -0.00356837], gripper: 0.854 -> 0.851 (85.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.3mm
Step done: False, reward: False, path length: 47, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00135591  0.00197722  0.00167437], rot=[ 0.00289673  0.00016738 -0.00631028], gripper: 0.854 -> 0.850 (85.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.9mm
Step done: False, reward: False, path length: 48, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00298726  0.00109302 -0.00260002], rot=[ 0.00168897  0.00613136 -0.00206908], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 49, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00087511 0.00115482 0.00231405], rot=[ 3.36912367e-03 -7.14135822e-05 -9.18839965e-03], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 50, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00063023  0.00302098 -0.00188574], rot=[-0.00185238 -0.00072068 -0.00044453], gripper: 0.854 -> 0.852 (85.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.6mm
Step done: False, reward: False, path length: 51, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00099586  0.00505255 -0.00094579], rot=[ 0.00060012 -0.00327945 -0.00251318], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 52, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560046.390781552] [a1x_serl_node]: Published EEF command: pos=[0.3177, -0.0114, 0.1969], quat=[0.050, 0.605, -0.036, 0.794]
  0%|          | 53/1000000 [00:15<42:46:02,  6.49it/s][INFO] [1770560046.543194278] [a1x_serl_node]: Published EEF command: pos=[0.3156, -0.0112, 0.1984], quat=[0.060, 0.611, -0.034, 0.788]
  0%|          | 54/1000000 [00:15<42:38:02,  6.52it/s][INFO] [1770560046.696831638] [a1x_serl_node]: Published EEF command: pos=[0.3165, -0.0078, 0.1942], quat=[0.053, 0.609, -0.032, 0.790]
  0%|          | 55/1000000 [00:15<42:40:18,  6.51it/s][INFO] [1770560046.849209883] [a1x_serl_node]: Published EEF command: pos=[0.3124, -0.0086, 0.1953], quat=[0.060, 0.610, -0.030, 0.790]
  0%|          | 56/1000000 [00:16<42:33:00,  6.53it/s][INFO] [1770560047.001430809] [a1x_serl_node]: Published EEF command: pos=[0.3176, -0.0083, 0.1950], quat=[0.056, 0.607, -0.030, 0.792]
  0%|          | 57/1000000 [00:16<42:31:17,  6.53it/s][INFO] [1770560047.154861668] [a1x_serl_node]: Published EEF command: pos=[0.3182, -0.0074, 0.1978], quat=[0.059, 0.611, -0.027, 0.789]
  0%|          | 58/1000000 [00:16<42:32:21,  6.53it/s][INFO] [1770560047.307894327] [a1x_serl_node]: Published EEF command: pos=[0.3179, -0.0099, 0.1959], quat=[0.057, 0.606, -0.031, 0.793]
  0%|          | 59/1000000 [00:16<42:29:17,  6.54it/s][INFO] [1770560047.461307496] [a1x_serl_node]: Published EEF command: pos=[0.3181, -0.0063, 0.1966], quat=[0.057, 0.608, -0.026, 0.791]
  0%|          | 60/1000000 [00:16<42:35:25,  6.52it/s][INFO] [1770560047.637014400] [a1x_serl_node]: Published EEF command: pos=[0.3170, -0.0101, 0.1973], quat=[0.057, 0.604, -0.030, 0.794]
  0%|          | 61/1000000 [00:16<44:24:33,  6.25it/s][INFO] [1770560047.791787434] [a1x_serl_node]: Published EEF command: pos=[0.3191, -0.0042, 0.1960], quat=[0.055, 0.609, -0.024, 0.791]
  0%|          | 62/1000000 [00:16<44:00:55,  6.31it/s][INFO] [1770560047.945868106] [a1x_serl_node]: Published EEF command: pos=[0.3168, -0.0094, 0.1969], quat=[0.056, 0.605, -0.029, 0.794]
  0%|          | 63/1000000 [00:17<43:35:07,  6.37it/s][INFO] [1770560048.099052585] [a1x_serl_node]: Published EEF command: pos=[0.3186, -0.0025, 0.1956], quat=[0.055, 0.608, -0.024, 0.792]
  0%|          | 64/1000000 [00:17<43:17:44,  6.42it/s][INFO] [1770560048.254337431] [a1x_serl_node]: Published EEF command: pos=[0.3184, -0.0085, 0.1980], quat=[0.052, 0.607, -0.027, 0.792]
  0%|          | 65/1000000 [00:17<43:11:47,  6.43it/s][INFO] [1770560048.407384009] [a1x_serl_node]: Published EEF command: pos=[0.3188, -0.0014, 0.1952], quat=[0.052, 0.607, -0.024, 0.792]
  0%|          | 66/1000000 [00:17<43:03:28,  6.45it/s][INFO] [1770560048.561827013] [a1x_serl_node]: Published EEF command: pos=[0.3200, -0.0071, 0.1984], quat=[0.054, 0.605, -0.029, 0.794]
  0%|          | 67/1000000 [00:17<42:59:30,  6.46it/s][INFO] [1770560048.716030765] [a1x_serl_node]: Published EEF command: pos=[0.3158, 0.0002, 0.1962], quat=[0.051, 0.607, -0.026, 0.792]
  0%|          | 68/1000000 [00:17<42:55:07,  6.47it/s][INFO] [1770560048.869537666] [a1x_serl_node]: Published EEF command: pos=[0.3183, -0.0042, 0.1995], quat=[0.056, 0.605, -0.030, 0.793]
  0%|          | 69/1000000 [00:18<42:55:58,  6.47it/s]Current EE Pos: [ 0.26230882 -0.00955732  0.18977827], Rot (quat): [-0.04962576  0.69909497  0.01112716  0.71321785]
Current EE Pos: [ 0.26272917 -0.00947524  0.18983998], Rot (quat): [-0.04897797  0.69818021  0.01098321  0.71416029]
Current EE Pos: [ 0.26446735 -0.00913184  0.19076833], Rot (quat): [-0.04862969  0.69449783  0.01084894  0.71776753]
Current EE Pos: [ 0.26600549 -0.00644905  0.19055876], Rot (quat): [-0.04519384  0.69302354  0.01129006  0.71940838]
Current EE Pos: [ 0.26827515 -0.0071796   0.19064819], Rot (quat): [-0.04576284  0.68923359  0.01082563  0.7230115 ]
Current EE Pos: [ 0.27057574 -0.00484756  0.1909939 ], Rot (quat): [-0.03953925  0.68628988  0.00830138  0.72620516]
Current EE Pos: [ 0.27186263 -0.00363641  0.18906007], Rot (quat): [-0.04045927  0.6858867   0.01118351  0.72649667]
Current EE Pos: [ 0.27206766 -0.0024779   0.19007488], Rot (quat): [-0.03224276  0.6843592   0.00838196  0.72838357]
Current EE Pos: [ 0.27254385 -0.00236041  0.19006458], Rot (quat): [-0.03658153  0.68359444  0.0124355   0.72883866]
Current EE Pos: [ 0.27557662 -0.00145821  0.19088265], Rot (quat): [-0.0281906   0.67890117  0.00835721  0.73364068]
Current EE Pos: [ 0.27502981 -0.00302022  0.19083356], Rot (quat): [-0.03270726  0.68077985  0.01136659  0.73166921]
Current EE Pos: [ 0.27865197 -0.00266857  0.19206919], Rot (quat): [-0.02208045  0.67412673  0.0057925   0.73826286]
Current EE Pos: [ 0.27885931 -0.00261748  0.19231787], Rot (quat): [-0.02415826  0.67449199  0.00841519  0.73783881]
Current EE Pos: [ 0.28126911 -0.00065196  0.19276457], Rot (quat): [-0.0132877   0.66977583  0.00479749  0.74242896]
Current EE Pos: [ 2.81166056e-01 -1.80214953e-04  1.92944951e-01], Rot (quat): [-0.01556268  0.67006866  0.00655945  0.74210698]
Current EE Pos: [0.28390048 0.00082687 0.19399859], Rot (quat): [-0.00342835  0.66495949  0.00205045  0.74686874]
Current EE Pos: [ 0.28389918 -0.00123117  0.19346293], Rot (quat): [-0.00687357  0.66647703  0.00157797  0.74549221]
Current EE Pos: [ 0.28792112 -0.00109675  0.19368134], Rot (quat): [ 0.00137073  0.65962179 -0.00261091  0.75159191]
Current EE Pos: [2.85980608e-01 2.42817810e-04 1.93659772e-01], Rot (quat): [ 0.00115162  0.66289586 -0.00095078  0.74871012]
Current EE Pos: [ 2.89881122e-01 -2.80132472e-04  1.93683190e-01], Rot (quat): [ 0.00563716  0.65585043 -0.00257758  0.75486542]
Current EE Pos: [ 0.28832019 -0.00225497  0.19285345], Rot (quat): [ 0.00791592  0.66022855 -0.00641018  0.75099568]
Current EE Pos: [ 0.29125863 -0.00202681  0.19313274], Rot (quat): [ 0.01369809  0.65434451 -0.00717639  0.75603844]
Current EE Pos: [ 0.29066267 -0.00324132  0.19346113], Rot (quat): [ 0.01283654  0.65577693 -0.0089713   0.75479227]
Current EE Pos: [ 0.29144987 -0.00357918  0.19309894], Rot (quat): [ 0.01777047  0.65395252 -0.01120236  0.75624388]
Current EE Pos: [ 0.29230489 -0.00638606  0.19320215], Rot (quat): [ 0.01797145  0.6535681  -0.01562217  0.75649303]
Current EE Pos: [ 0.29456737 -0.0065769   0.19290251], Rot (quat): [ 0.02580773  0.65022225 -0.01720951  0.75911054]
Current EE Pos: [ 0.2925112  -0.00672111  0.19211727], Rot (quat): [ 0.02475738  0.65391611 -0.01946536  0.7559113 ]
Current EE Pos: [ 0.29477005 -0.00826939  0.1921694 ], Rot (quat): [ 0.03040769  0.64983908 -0.02093092  0.75917484]
Current EE Pos: [ 0.2949635  -0.00905539  0.19220293], Rot (quat): [ 0.03122054  0.65092382 -0.0224318   0.75816902]
Current EE Pos: [ 0.2959788  -0.01083214  0.19198502], Rot (quat): [ 0.03766102  0.64853274 -0.02574551  0.75981846]
Current EE Pos: [ 0.29561602 -0.01098005  0.19179037], Rot (quat): [ 0.03822068  0.64932466 -0.02575592  0.7591135 ]
Current EE Pos: [ 0.29882645 -0.01398569  0.19184247], Rot (quat): [ 0.0422306   0.64399162 -0.030164    0.76327027]
Current EE Pos: [ 0.29820902 -0.0130834   0.19154417], Rot (quat): [ 0.04383998  0.64569747 -0.02832233  0.76180751]
Current EE Pos: [ 0.3001619  -0.01409552  0.19128041], Rot (quat): [ 0.04277293  0.64169695 -0.03011265  0.76517236]
Current EE Pos: [ 0.30022726 -0.01232543  0.19115861], Rot (quat): [ 0.04476313  0.64196802 -0.02746099  0.76493086]
Current EE Pos: [ 0.30337552 -0.01367129  0.19235424], Rot (quat): [ 0.04414086  0.63580912 -0.02829427  0.77006349]
Current EE Pos: [ 0.3045502  -0.01365876  0.19210906], Rot (quat): [ 0.04623427  0.6337049  -0.02766827  0.77169616]
Current EE Pos: [ 0.3052867  -0.01293794  0.19279729], Rot (quat): [ 0.04616569  0.6315175  -0.0270337   0.77351377]
Current EE Pos: [ 0.30743602 -0.01304278  0.19325118], Rot (quat): [ 0.04976981  0.62753154 -0.02764025  0.77650702]
Current EE Pos: [ 0.3077408  -0.01422178  0.19432981], Rot (quat): [ 0.05010294  0.62563408 -0.02980534  0.77793531]
Current EE Pos: [ 0.31127994 -0.01583738  0.19591731], Rot (quat): [ 0.04893072  0.61818258 -0.03263594  0.78383097]
Current EE Pos: [ 0.31128036 -0.01625761  0.19558532], Rot (quat): [ 0.05015879  0.61880141 -0.03242991  0.78327339]
Current EE Pos: [ 0.31355321 -0.01659433  0.1961144 ], Rot (quat): [ 0.04886891  0.61455247 -0.03375025  0.78663715]
Current EE Pos: [ 0.31324928 -0.01630974  0.19626114], Rot (quat): [ 0.0499208   0.61537694 -0.03295397  0.78596003]
Current EE Pos: [ 0.31638918 -0.01595725  0.19599535], Rot (quat): [ 0.05089548  0.61014617 -0.03379228  0.78992999]
Current EE Pos: [ 0.31511542 -0.01577356  0.19606858], Rot (quat): [ 0.05178639  0.61290743 -0.03308989  0.7877612 ]
Current EE Pos: [ 0.31603637 -0.01593673  0.19514935], Rot (quat): [ 0.05057642  0.61125325 -0.03355343  0.78910434]
Current EE Pos: [ 0.31632139 -0.01552799  0.19524143], Rot (quat): [ 0.05170811  0.61207639 -0.03336178  0.78840076]
Current EE Pos: [ 0.3166362  -0.01597161  0.19477769], Rot (quat): [ 0.05116025  0.61097602 -0.03400363  0.78926211]
Current EE Pos: [ 0.31564039 -0.01510788  0.19501054], Rot (quat): [ 0.05395016  0.6124197  -0.03467292  0.78792721]
Current EE Pos: [ 0.31628613 -0.01588468  0.19425141], Rot (quat): [ 0.05186678  0.61145879 -0.03415158  0.78883563]
Current EE Pos: [ 0.31615195 -0.01540645  0.19468272], Rot (quat): [ 0.05596353  0.61164559 -0.03635837  0.78831201]
Current EE Pos: [ 0.31769441 -0.01460743  0.19435222], Rot (quat): [ 0.05127121  0.60916187 -0.03409338  0.79065208]
Current EE Pos: [ 0.31688013 -0.01344156  0.19423513], Rot (quat): [ 0.05653161  0.61052036 -0.03626923  0.7891474 ]
Current EE Pos: [ 0.31837355 -0.01354928  0.19533051], Rot (quat): [ 0.05093786  0.60638459 -0.03543459  0.79274678]
Current EE Pos: [ 0.31675788 -0.0121381   0.19565947], Rot (quat): [ 0.05852672  0.6090544  -0.03403029  0.7902337 ]
Current EE Pos: [ 0.3179986  -0.01153471  0.19550469], Rot (quat): [ 0.05473614  0.60649527 -0.03302631  0.7925129 ]
Current EE Pos: [ 0.31551966 -0.00975245  0.19547482], Rot (quat): [ 0.05876763  0.60972768 -0.03059493  0.78983699]
Current EE Pos: [ 0.31738006 -0.00995943  0.19589116], Rot (quat): [ 0.05606937  0.60573246 -0.03064416  0.79309858]
Current EE Pos: [ 0.31674333 -0.00814436  0.19638472], Rot (quat): [ 0.05854193  0.60809782 -0.02778552  0.7912129 ]
Current EE Pos: [ 0.31821997 -0.00918451  0.19659268], Rot (quat): [ 0.05640351  0.60426507 -0.02931911  0.79424351]
Current EE Pos: [ 0.31753686 -0.00709251  0.19673373], Rot (quat): [ 0.05674551  0.60628258 -0.02576884  0.79280347]
Current EE Pos: [ 0.31878631 -0.00925115  0.19710926], Rot (quat): [ 0.05627136  0.6027392  -0.0294263   0.7954075 ]
Current EE Pos: [ 0.31775259 -0.00768438  0.19653836], Rot (quat): [ 0.05619784  0.60558041 -0.02623674  0.7933636 ]
Current EE Pos: [ 0.31858959 -0.00894843  0.19650992], Rot (quat): [ 0.05523139  0.60338026 -0.02874787  0.79501907]
Current EE Pos: [ 0.31807442 -0.0063611   0.1962173 ], Rot (quat): [ 0.0555963   0.60506775 -0.02591124  0.79380771]
Current EE Pos: [ 0.31834665 -0.00798862  0.19628832], Rot (quat): [ 0.05280776  0.60401426 -0.02679063  0.79477064]
Current EE Pos: [ 0.31848399 -0.00507781  0.19588176], Rot (quat): [ 0.05283144  0.60467188 -0.02529956  0.79431775]
Current EE Pos: [ 0.31908923 -0.00730725  0.19670538], Rot (quat): [ 0.05306375  0.6031654  -0.02791674  0.79535928]
[INFO] [1770560049.024192360] [a1x_serl_node]: Published EEF command: pos=[0.3149, -0.0045, 0.1974], quat=[0.052, 0.609, -0.027, 0.791]
  0%|          | 70/1000000 [00:18<42:52:49,  6.48it/s][INFO] [1770560049.176499506] [a1x_serl_node]: Published EEF command: pos=[0.3168, -0.0043, 0.1978], quat=[0.053, 0.610, -0.029, 0.790]
  0%|          | 71/1000000 [00:18<42:42:08,  6.50it/s][INFO] [1770560049.330806109] [a1x_serl_node]: Published EEF command: pos=[0.3175, -0.0040, 0.1934], quat=[0.054, 0.612, -0.026, 0.789]
  0%|          | 72/1000000 [00:18<42:42:58,  6.50it/s][INFO] [1770560049.483685778] [a1x_serl_node]: Published EEF command: pos=[0.3138, -0.0055, 0.1959], quat=[0.054, 0.610, -0.030, 0.790]
  0%|          | 73/1000000 [00:18<42:41:00,  6.51it/s][INFO] [1770560049.638829614] [a1x_serl_node]: Published EEF command: pos=[0.3151, -0.0047, 0.1972], quat=[0.052, 0.614, -0.027, 0.787]
  0%|          | 74/1000000 [00:18<42:47:57,  6.49it/s][INFO] [1770560049.791500122] [a1x_serl_node]: Published EEF command: pos=[0.3169, -0.0095, 0.1974], quat=[0.053, 0.615, -0.030, 0.786]
  0%|          | 75/1000000 [00:18<42:40:00,  6.51it/s][INFO] [1770560049.944440401] [a1x_serl_node]: Published EEF command: pos=[0.3162, -0.0029, 0.1959], quat=[0.052, 0.612, -0.027, 0.789]
  0%|          | 76/1000000 [00:19<42:37:59,  6.52it/s][INFO] [1770560050.097457580] [a1x_serl_node]: Published EEF command: pos=[0.3143, -0.0070, 0.1984], quat=[0.052, 0.619, -0.028, 0.783]
  0%|          | 77/1000000 [00:19<42:34:30,  6.52it/s][INFO] [1770560050.251308852] [a1x_serl_node]: Published EEF command: pos=[0.3166, -0.0032, 0.1957], quat=[0.051, 0.611, -0.030, 0.789]
  0%|          | 78/1000000 [00:19<42:36:24,  6.52it/s]EEF delta: pos=[2.48402357e-05 3.19007738e-03 2.58151582e-03], rot=[-0.00476595 -0.01051207 -0.00216809], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 53, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00130961  0.00225502  0.00413425], rot=[ 0.00810309  0.00186624 -0.00088866], gripper: 0.854 -> 0.857 (85.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.9mm
Step done: False, reward: False, path length: 54, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00191198  0.00577822 -0.00108835], rot=[0.00782121 0.0069729  0.00272697], gripper: 0.854 -> 0.861 (86.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.2mm
Step done: False, reward: False, path length: 55, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00432591  0.00350031 -0.00036874], rot=[0.00679601 0.00098939 0.00452072], gripper: 0.854 -> 0.852 (85.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.6mm
Step done: False, reward: False, path length: 56, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00035717  0.00328268 -0.00047589], rot=[0.00486716 0.00094039 0.00295135], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.3mm
Step done: False, reward: False, path length: 57, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00265994 0.00235761 0.00233404], rot=[0.00406291 0.0034663  0.00499239], gripper: 0.854 -> 0.859 (85.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.3mm
Step done: False, reward: False, path length: 58, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[4.80528281e-04 4.32105735e-05 4.68388898e-05], rot=[ 0.00089987  0.00141482 -0.00114745], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.5mm
Step done: False, reward: False, path length: 59, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00131123 0.00189433 0.00025346], rot=[-0.00016418 -0.00061409  0.00477457], gripper: 0.854 -> 0.850 (85.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.3mm
Step done: False, reward: False, path length: 60, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00121213 -0.00094741  0.00073699], rot=[ 3.05057038e-04 -3.37725505e-05 -1.64150400e-03], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.7mm
Step done: False, reward: False, path length: 61, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00153466  0.0028557  -0.00076922], rot=[-0.00072592  0.00593903  0.00451758], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.3mm
Step done: False, reward: False, path length: 62, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00197182 -0.00018062 -0.00024373], rot=[-0.00031053  0.00505074  0.0012981 ], gripper: 0.854 -> 0.857 (85.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 63, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00081984  0.00520635 -0.00098065], rot=[0.00031734 0.00502907 0.00477604], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 64, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0002254   0.00040784  0.00151468], rot=[-0.00124273  0.00982545  0.0072678 ], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.6mm
Step done: False, reward: False, path length: 65, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00071229  0.00497308 -0.00105711], rot=[-0.00434937  0.00526526  0.00751348], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.1mm
Step done: False, reward: False, path length: 66, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.0016066  0.00092879 0.00206598], rot=[-0.0005132   0.00169849 -0.00429955], gripper: 0.854 -> 0.861 (86.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.8mm
Step done: False, reward: False, path length: 67, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0027025   0.00532402  0.0003657 ], rot=[-0.00312431  0.00645458  0.00103198], gripper: 0.854 -> 0.857 (85.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.0mm
Step done: False, reward: False, path length: 68, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00083894  0.00313083  0.00281262], rot=[ 0.00227335  0.00609078 -0.0055445 ], gripper: 0.854 -> 0.860 (86.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.3mm
Step done: False, reward: False, path length: 69, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00295277 -0.00029684  0.00109736], rot=[-0.00029008  0.01049854  0.00093962], gripper: 0.854 -> 0.858 (85.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.2mm
Step done: False, reward: False, path length: 70, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0011821   0.00155012  0.00060585], rot=[-0.00226888  0.01469477  0.00225097], gripper: 0.854 -> 0.862 (86.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 71, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00078096  0.00064238 -0.00295195], rot=[0.0041735  0.01366245 0.00215698], gripper: 0.854 -> 0.858 (85.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 72, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00304529 -0.00045519 -0.00016995], rot=[ 9.87814274e-05  8.25001486e-03 -4.29460127e-03], gripper: 0.854 -> 0.860 (86.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 73, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-1.24086975e-03  7.65568111e-05  2.18912610e-03], rot=[0.00059075 0.01338849 0.00183073], gripper: 0.854 -> 0.863 (86.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 74, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00074095 -0.00396859  0.00242832], rot=[-0.00035373  0.01729703 -0.00078225], gripper: 0.854 -> 0.861 (86.1mm)
⏱️  ✓ 执行耗时=99ms, 误差=4.7mm
Step done: False, reward: False, path length: 75, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00135292 0.00196574 0.00130497], rot=[0.00097013 0.00154821 0.00119846], gripper: 0.854 -> 0.857 (85.7mm)
⏱️  ✓ 执行耗时=99ms, 误差=2.7mm
Step done: False, reward: False, path length: 76, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00018496 -0.00088953  0.00376207], rot=[0.00287966 0.01622863 0.0009517 ], gripper: 0.854 -> 0.858 (85.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 77, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00088779 0.00099895 0.00038936], rot=[-0.00502677  0.00398907 -0.00146627], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.4mm
Step done: False, reward: False, path length: 78, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560050.406550929] [a1x_serl_node]: Published EEF command: pos=[0.3136, -0.0055, 0.1951], quat=[0.050, 0.614, -0.028, 0.787]
  0%|          | 79/1000000 [00:19<42:51:02,  6.48it/s][INFO] [1770560050.559872659] [a1x_serl_node]: Published EEF command: pos=[0.3148, -0.0036, 0.1974], quat=[0.051, 0.609, -0.029, 0.791]
  0%|          | 80/1000000 [00:19<42:41:03,  6.51it/s][INFO] [1770560050.715616747] [a1x_serl_node]: Published EEF command: pos=[0.3156, -0.0044, 0.1977], quat=[0.053, 0.615, -0.029, 0.786]
  0%|          | 81/1000000 [00:19<42:51:21,  6.48it/s][INFO] [1770560050.869259629] [a1x_serl_node]: Published EEF command: pos=[0.3161, -0.0008, 0.1975], quat=[0.051, 0.613, -0.029, 0.788]
  0%|          | 82/1000000 [00:20<42:51:58,  6.48it/s][INFO] [1770560051.024686226] [a1x_serl_node]: Published EEF command: pos=[0.3132, 0.0002, 0.1989], quat=[0.052, 0.615, -0.028, 0.786]
  0%|          | 83/1000000 [00:20<42:58:09,  6.46it/s][INFO] [1770560051.184191627] [a1x_serl_node]: Published EEF command: pos=[0.3175, 0.0004, 0.2023], quat=[0.056, 0.614, -0.029, 0.787]
  0%|          | 84/1000000 [00:20<43:21:28,  6.41it/s][INFO] [1770560051.341563772] [a1x_serl_node]: Published EEF command: pos=[0.3115, 0.0024, 0.1990], quat=[0.054, 0.617, -0.028, 0.785]
  0%|          | 85/1000000 [00:20<43:24:40,  6.40it/s][INFO] [1770560051.496496648] [a1x_serl_node]: Published EEF command: pos=[0.3142, 0.0013, 0.2019], quat=[0.055, 0.617, -0.027, 0.785]
  0%|          | 86/1000000 [00:20<43:18:21,  6.41it/s][INFO] [1770560051.656188679] [a1x_serl_node]: Published EEF command: pos=[0.3111, 0.0002, 0.1980], quat=[0.053, 0.620, -0.026, 0.783]
  0%|          | 87/1000000 [00:20<43:42:02,  6.36it/s][INFO] [1770560051.812407020] [a1x_serl_node]: Published EEF command: pos=[0.3127, 0.0030, 0.1994], quat=[0.054, 0.618, -0.025, 0.784]
  0%|          | 88/1000000 [00:21<43:30:14,  6.38it/s][INFO] [1770560051.965094738] [a1x_serl_node]: Published EEF command: pos=[0.3112, 0.0028, 0.1936], quat=[0.052, 0.619, -0.026, 0.783]
  0%|          | 89/1000000 [00:21<43:10:55,  6.43it/s][INFO] [1770560052.118442199] [a1x_serl_node]: Published EEF command: pos=[0.3118, 0.0032, 0.1951], quat=[0.052, 0.615, -0.026, 0.787]
  0%|          | 90/1000000 [00:21<43:02:22,  6.45it/s][INFO] [1770560052.272245832] [a1x_serl_node]: Published EEF command: pos=[0.3117, 0.0036, 0.1954], quat=[0.053, 0.618, -0.025, 0.784]
  0%|          | 91/1000000 [00:21<42:55:13,  6.47it/s][INFO] [1770560052.426051615] [a1x_serl_node]: Published EEF command: pos=[0.3144, 0.0006, 0.1948], quat=[0.051, 0.611, -0.027, 0.789]
  0%|          | 92/1000000 [00:21<42:50:10,  6.48it/s][INFO] [1770560052.578081891] [a1x_serl_node]: Published EEF command: pos=[0.3108, 0.0030, 0.1940], quat=[0.053, 0.617, -0.024, 0.785]
  0%|          | 93/1000000 [00:21<42:40:44,  6.51it/s][INFO] [1770560052.731812264] [a1x_serl_node]: Published EEF command: pos=[0.3103, 0.0023, 0.1954], quat=[0.050, 0.615, -0.026, 0.787]
  0%|          | 94/1000000 [00:21<42:40:25,  6.51it/s][INFO] [1770560052.883971051] [a1x_serl_node]: Published EEF command: pos=[0.3114, 0.0027, 0.1954], quat=[0.053, 0.619, -0.023, 0.783]
  0%|          | 95/1000000 [00:22<42:34:55,  6.52it/s][INFO] [1770560053.037696974] [a1x_serl_node]: Published EEF command: pos=[0.3127, 0.0014, 0.1944], quat=[0.050, 0.615, -0.025, 0.787]
  0%|          | 96/1000000 [00:22<42:34:43,  6.52it/s][INFO] [1770560053.188843788] [a1x_serl_node]: Published EEF command: pos=[0.3136, -0.0006, 0.1917], quat=[0.054, 0.621, -0.023, 0.781]
  0%|          | 97/1000000 [00:22<42:26:14,  6.54it/s][INFO] [1770560053.343015932] [a1x_serl_node]: Published EEF command: pos=[0.3091, 0.0015, 0.1955], quat=[0.052, 0.618, -0.025, 0.784]
  0%|          | 98/1000000 [00:22<42:31:16,  6.53it/s][INFO] [1770560053.496855745] [a1x_serl_node]: Published EEF command: pos=[0.3109, 0.0028, 0.1926], quat=[0.050, 0.617, -0.023, 0.785]
  0%|          | 99/1000000 [00:22<42:35:29,  6.52it/s][INFO] [1770560053.649339534] [a1x_serl_node]: Published EEF command: pos=[0.3136, 0.0020, 0.1911], quat=[0.052, 0.621, -0.023, 0.782]
  0%|          | 100/1000000 [00:22<42:37:25,  6.52it/s][INFO] [1770560053.856705785] [a1x_serl_node]: Published EEF command: pos=[0.3094, 0.0043, 0.1940], quat=[0.053, 0.620, -0.023, 0.782]
  0%|          | 101/1000000 [00:23<47:00:33,  5.91it/s][INFO] [1770560054.011356951] [a1x_serl_node]: Published EEF command: pos=[0.3095, 0.0040, 0.1920], quat=[0.053, 0.620, -0.024, 0.783]
  0%|          | 102/1000000 [00:23<45:47:18,  6.07it/s][INFO] [1770560054.164163771] [a1x_serl_node]: Published EEF command: pos=[0.3077, 0.0044, 0.1922], quat=[0.050, 0.618, -0.024, 0.785]
  0%|          | 103/1000000 [00:23<44:47:28,  6.20it/s][INFO] [1770560054.319343449] [a1x_serl_node]: Published EEF command: pos=[0.3100, 0.0069, 0.1924], quat=[0.053, 0.624, -0.022, 0.780]
  0%|          | 104/1000000 [00:23<44:19:16,  6.27it/s]EEF delta: pos=[5.06289769e-04 4.67457576e-05 4.39184369e-04], rot=[-0.00358443 -0.0046729   0.00068934], gripper: 0.854 -> 0.851 (85.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.7mm
Step done: False, reward: False, path length: 79, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00144093  0.00110957  0.00176995], rot=[-0.00073767  0.00132909 -0.00025047], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 80, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00124912 0.00063147 0.00242225], rot=[ 0.00317983  0.00613855 -0.00643631], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.8mm
Step done: False, reward: False, path length: 81, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00037448 0.00382656 0.00139292], rot=[0.00162931 0.01021137 0.00068193], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 82, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00109771  0.00458454  0.0028849 ], rot=[0.00147965 0.0064313  0.00101405], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.5mm
Step done: False, reward: False, path length: 83, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00193702 0.00330114 0.00612049], rot=[ 0.0072687   0.01108869 -0.00656302], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.2mm
Step done: False, reward: False, path length: 84, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00198146  0.00424856  0.00264545], rot=[ 0.00361619  0.00892177 -0.00095798], gripper: 0.854 -> 0.859 (85.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 85, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00075267  0.00219258  0.00388781], rot=[0.00278472 0.01493566 0.00248144], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.5mm
Step done: False, reward: False, path length: 86, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-1.64965552e-03  4.85428609e-05  9.02659260e-04], rot=[-0.00079242  0.0135817   0.00381631], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.9mm
Step done: False, reward: False, path length: 87, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00045825 0.00252993 0.00162419], rot=[0.00159398 0.00663217 0.00347993], gripper: 0.854 -> 0.852 (85.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 88, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00083642  0.00218081 -0.00311766], rot=[-0.00048109  0.00930246  0.0021631 ], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 89, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-3.50992195e-05  2.04939814e-03 -1.72798662e-03], rot=[-0.00224421 -0.00377851  0.00159595], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 90, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00022278  0.00212123 -0.00066378], rot=[ 0.00206521  0.00431268 -0.00072284], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.2mm
Step done: False, reward: False, path length: 91, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00176182 -0.00142617 -0.00130888], rot=[-0.00458726 -0.00824441 -0.00195185], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 92, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00150939  0.00087202 -0.00190117], rot=[ 0.00171341  0.00456406 -0.00197451], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 93, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00362394  0.00088715 -0.00103062], rot=[-0.00166949  0.00751724  0.00071313], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 94, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00110809  0.00085876 -0.00031637], rot=[0.00406678 0.01090999 0.00209362], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.4mm
Step done: False, reward: False, path length: 95, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00026065 -0.00058449 -0.00129293], rot=[-0.00115294  0.00272171  0.00104634], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.4mm
Step done: False, reward: False, path length: 96, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00146341 -0.00297383 -0.00331918], rot=[ 0.00540143  0.01372325 -0.00101011], gripper: 0.854 -> 0.858 (85.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.7mm
Step done: False, reward: False, path length: 97, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0038549  -0.00042653  0.00036013], rot=[ 0.00198923  0.00969995 -0.0014097 ], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 98, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00197232  0.00080545 -0.00163343], rot=[-0.00437409  0.00297628  0.00167842], gripper: 0.854 -> 0.854 (85.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 99, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00207822  0.00066217 -0.00288742], rot=[0.00276211 0.00926395 0.00301645], gripper: 0.854 -> 0.851 (85.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.6mm
Step done: False, reward: False, path length: 100, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00301201  0.00235127  0.00019762], rot=[ 0.00239526  0.01062913 -0.00262542], gripper: 0.854 -> 0.857 (85.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 101, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00306916  0.00194778 -0.00060973], rot=[ 0.00101609  0.00626212 -0.00450668], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 102, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00350209  0.00170717 -0.0004128 ], rot=[-0.00377164 -0.00271772  0.0013368 ], gripper: 0.854 -> 0.850 (85.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 103, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00109544  0.00409672 -0.00020256], rot=[0.00214874 0.01351118 0.00176407], gripper: 0.854 -> 0.855 (85.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.2mm
Step done: False, reward: False, path length: 104, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560054.473589194] [a1x_serl_node]: Published EEF command: pos=[0.3141, 0.0014, 0.1966], quat=[0.051, 0.619, -0.026, 0.783]
  0%|          | 105/1000000 [00:23<43:51:16,  6.33it/s][INFO] [1770560054.625924142] [a1x_serl_node]: Published EEF command: pos=[0.3096, 0.0063, 0.1930], quat=[0.053, 0.622, -0.023, 0.781]
  0%|          | 106/1000000 [00:23<43:20:47,  6.41it/s][INFO] [1770560054.778711012] [a1x_serl_node]: Published EEF command: pos=[0.3121, 0.0058, 0.1923], quat=[0.050, 0.620, -0.024, 0.783]
  0%|          | 107/1000000 [00:23<43:04:21,  6.45it/s][INFO] [1770560054.932816537] [a1x_serl_node]: Published EEF command: pos=[0.3096, 0.0055, 0.1919], quat=[0.054, 0.620, -0.024, 0.782]
  0%|          | 108/1000000 [00:24<43:01:38,  6.46it/s][INFO] [1770560055.086745631] [a1x_serl_node]: Published EEF command: pos=[0.3120, 0.0048, 0.1919], quat=[0.049, 0.617, -0.025, 0.785]
  0%|          | 109/1000000 [00:24<42:57:04,  6.47it/s][INFO] [1770560055.240828865] [a1x_serl_node]: Published EEF command: pos=[0.3107, 0.0046, 0.1911], quat=[0.053, 0.618, -0.025, 0.784]
  0%|          | 110/1000000 [00:24<42:52:16,  6.48it/s][INFO] [1770560055.393632826] [a1x_serl_node]: Published EEF command: pos=[0.3142, 0.0048, 0.1926], quat=[0.052, 0.616, -0.026, 0.785]
  0%|          | 111/1000000 [00:24<42:45:05,  6.50it/s][INFO] [1770560055.546664377] [a1x_serl_node]: Published EEF command: pos=[0.3099, 0.0050, 0.1923], quat=[0.051, 0.621, -0.022, 0.782]
  0%|          | 112/1000000 [00:24<42:42:38,  6.50it/s][INFO] [1770560055.700079489] [a1x_serl_node]: Published EEF command: pos=[0.3133, 0.0059, 0.1899], quat=[0.050, 0.615, -0.025, 0.787]
  0%|          | 113/1000000 [00:24<42:58:18,  6.46it/s][INFO] [1770560055.858143167] [a1x_serl_node]: Published EEF command: pos=[0.3118, 0.0053, 0.1917], quat=[0.048, 0.623, -0.019, 0.780]
  0%|          | 114/1000000 [00:25<42:55:15,  6.47it/s][INFO] [1770560056.009317742] [a1x_serl_node]: Published EEF command: pos=[0.3104, 0.0076, 0.1888], quat=[0.052, 0.617, -0.022, 0.785]
  0%|          | 115/1000000 [00:25<42:48:11,  6.49it/s][INFO] [1770560056.164818512] [a1x_serl_node]: Published EEF command: pos=[0.3131, 0.0060, 0.1904], quat=[0.047, 0.620, -0.019, 0.783]
  0%|          | 116/1000000 [00:25<42:47:37,  6.49it/s][INFO] [1770560056.321114864] [a1x_serl_node]: Published EEF command: pos=[0.3118, 0.0078, 0.1899], quat=[0.048, 0.620, -0.020, 0.783]
  0%|          | 117/1000000 [00:25<42:55:40,  6.47it/s][INFO] [1770560056.473335333] [a1x_serl_node]: Published EEF command: pos=[0.3103, 0.0061, 0.1887], quat=[0.045, 0.616, -0.018, 0.786]
  0%|          | 118/1000000 [00:25<42:45:12,  6.50it/s][INFO] [1770560056.627293817] [a1x_serl_node]: Published EEF command: pos=[0.3131, 0.0053, 0.1893], quat=[0.048, 0.615, -0.020, 0.787]
  0%|          | 119/1000000 [00:25<42:45:46,  6.49it/s][INFO] [1770560056.779968798] [a1x_serl_node]: Published EEF command: pos=[0.3146, 0.0067, 0.1909], quat=[0.042, 0.615, -0.017, 0.787]
  0%|          | 120/1000000 [00:25<42:39:11,  6.51it/s][INFO] [1770560056.936147800] [a1x_serl_node]: Published EEF command: pos=[0.3130, 0.0050, 0.1885], quat=[0.047, 0.614, -0.020, 0.787]
  0%|          | 121/1000000 [00:26<42:49:28,  6.49it/s][INFO] [1770560057.088301538] [a1x_serl_node]: Published EEF command: pos=[0.3140, 0.0052, 0.1902], quat=[0.040, 0.615, -0.015, 0.787]
  0%|          | 122/1000000 [00:26<42:42:05,  6.50it/s][INFO] [1770560057.241033079] [a1x_serl_node]: Published EEF command: pos=[0.3130, 0.0015, 0.1890], quat=[0.044, 0.616, -0.020, 0.787]
  0%|          | 123/1000000 [00:26<42:35:41,  6.52it/s][INFO] [1770560057.395468766] [a1x_serl_node]: Published EEF command: pos=[0.3125, 0.0049, 0.1906], quat=[0.041, 0.614, -0.017, 0.788]
  0%|          | 124/1000000 [00:26<42:41:33,  6.51it/s][INFO] [1770560057.547681705] [a1x_serl_node]: Published EEF command: pos=[0.3156, 0.0067, 0.1895], quat=[0.040, 0.612, -0.020, 0.789]
  0%|          | 125/1000000 [00:26<42:35:16,  6.52it/s][INFO] [1770560057.699465753] [a1x_serl_node]: Published EEF command: pos=[0.3117, 0.0059, 0.1878], quat=[0.040, 0.613, -0.020, 0.789]
  0%|          | 126/1000000 [00:26<42:28:57,  6.54it/s][INFO] [1770560057.852034243] [a1x_serl_node]: Published EEF command: pos=[0.3161, 0.0087, 0.1889], quat=[0.041, 0.610, -0.021, 0.791]
  0%|          | 127/1000000 [00:27<42:25:25,  6.55it/s][INFO] [1770560058.039174679] [a1x_serl_node]: Published EEF command: pos=[0.3156, 0.0053, 0.1906], quat=[0.040, 0.611, -0.020, 0.790]
  0%|          | 128/1000000 [00:27<45:17:20,  6.13it/s][INFO] [1770560058.193307185] [a1x_serl_node]: Published EEF command: pos=[0.3179, 0.0073, 0.1877], quat=[0.038, 0.611, -0.020, 0.790]
  0%|          | 129/1000000 [00:27<44:32:18,  6.24it/s][INFO] [1770560058.347956152] [a1x_serl_node]: Published EEF command: pos=[0.3171, 0.0107, 0.1906], quat=[0.040, 0.612, -0.017, 0.790]
  0%|          | 130/1000000 [00:27<44:02:28,  6.31it/s]EEF delta: pos=[ 0.00366068 -0.00171686  0.00375876], rot=[-0.00229818  0.00279386 -0.00370337], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.5mm
Step done: False, reward: False, path length: 105, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00037003  0.00180472  0.00068112], rot=[-0.00067765  0.00513188 -0.00236836], gripper: 0.854 -> 0.857 (85.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 106, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00037095  0.00350148 -0.00171836], rot=[-0.00050461  0.00684732  0.00204018], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 107, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00146612  0.00181822 -0.00128149], rot=[ 0.00248707  0.0056939  -0.00360563], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 108, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00039351  0.00147424 -0.0013006 ], rot=[-0.00210604 -0.00119501 -0.00250933], gripper: 0.854 -> 0.856 (85.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 109, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00041682  0.00091867 -0.00160114], rot=[-0.00168702  0.00097969 -0.00370052], gripper: 0.854 -> 0.853 (85.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.9mm
Step done: False, reward: False, path length: 110, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00189607  0.00128686 -0.00025665], rot=[ 0.0012156   0.00274074 -0.003388  ], gripper: 0.854 -> 0.849 (84.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.3mm
Step done: False, reward: False, path length: 111, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00196019  0.00152404 -0.00058137], rot=[0.00175955 0.01040118 0.00403879], gripper: 0.854 -> 0.852 (85.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 112, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00029257  0.0025312  -0.00283399], rot=[-0.00182654  0.00025661  0.00377796], gripper: 0.854 -> 0.850 (85.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 113, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00017643  0.00109567 -0.00030987], rot=[0.00141442 0.01318996 0.00917256], gripper: 0.854 -> 0.851 (85.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.2mm
Step done: False, reward: False, path length: 114, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00326574  0.00320591 -0.00339431], rot=[0.00530334 0.01024931 0.00172051], gripper: 0.854 -> 0.846 (84.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.7mm
Step done: False, reward: False, path length: 115, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00168433  0.00073298 -0.00079439], rot=[-0.00127859  0.00392221  0.00462704], gripper: 0.853 -> 0.842 (84.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 116, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00092948  0.00221401 -0.0008087 ], rot=[-0.00084539  0.01068099  0.00655321], gripper: 0.848 -> 0.840 (84.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 117, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00147     0.00030984 -0.00209094], rot=[-0.00350096 -0.00517393  0.00421013], gripper: 0.845 -> 0.836 (83.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 118, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00064878 -0.00088806 -0.00100795], rot=[-1.10464590e-03 -4.53262404e-03 -6.49099238e-05], gripper: 0.844 -> 0.839 (83.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.5mm
Step done: False, reward: False, path length: 119, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00211771 0.00051851 0.00039598], rot=[-0.00339045 -0.00278351  0.00459634], gripper: 0.844 -> 0.839 (83.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.2mm
Step done: False, reward: False, path length: 120, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00036513 -0.00094237 -0.00223686], rot=[ 0.00100043  0.00054687 -0.00102763], gripper: 0.844 -> 0.830 (83.0mm)
⏱️  ✓ 执行耗时=99ms, 误差=2.4mm
Step done: False, reward: False, path length: 121, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00025147 -0.00116379 -0.00085799], rot=[-0.00158587  0.00358962  0.00779976], gripper: 0.835 -> 0.824 (82.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.5mm
Step done: False, reward: False, path length: 122, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00111973 -0.00430984 -0.00194777], rot=[-0.00311213  0.00731879  0.00212141], gripper: 0.829 -> 0.818 (81.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.8mm
Step done: False, reward: False, path length: 123, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00146963 -0.00063605 -0.00014264], rot=[-0.00100115  0.00284048  0.00179584], gripper: 0.822 -> 0.811 (81.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.6mm
Step done: False, reward: False, path length: 124, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.0016968   0.00252887 -0.000703  ], rot=[-0.00439335 -0.00262586  0.00460726], gripper: 0.813 -> 0.810 (81.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 125, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0023254   0.00141278 -0.0022679 ], rot=[-0.00405639 -0.00151708 -0.000497  ], gripper: 0.813 -> 0.816 (81.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.5mm
Step done: False, reward: False, path length: 126, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00061833  0.00357139 -0.00140846], rot=[-0.00174822 -0.00169897 -0.00239183], gripper: 0.813 -> 0.817 (81.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 127, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 9.99265350e-04 -7.08563020e-05  1.14551885e-03], rot=[-0.00078214 -0.0023746   0.0013977 ], gripper: 0.813 -> 0.811 (81.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.5mm
Step done: False, reward: False, path length: 128, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00128923  0.00046051 -0.0023057 ], rot=[-0.00294553  0.00581291  0.00337709], gripper: 0.813 -> 0.812 (81.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 129, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.0009776  0.00505549 0.00062896], rot=[0.00422169 0.00608336 0.00515492], gripper: 0.813 -> 0.809 (80.9mm)
⏱️  ✓ 执行耗时=99ms, 误差=5.2mm
Step done: False, reward: False, path length: 130, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560058.501832287] [a1x_serl_node]: Published EEF command: pos=[0.3176, 0.0074, 0.1872], quat=[0.035, 0.616, -0.018, 0.786]
  0%|          | 131/1000000 [00:27<43:40:51,  6.36it/s][INFO] [1770560058.655815503] [a1x_serl_node]: Published EEF command: pos=[0.3188, 0.0085, 0.1877], quat=[0.040, 0.608, -0.015, 0.793]
  0%|          | 132/1000000 [00:27<43:22:14,  6.40it/s][INFO] [1770560058.808392863] [a1x_serl_node]: Published EEF command: pos=[0.3193, 0.0043, 0.1830], quat=[0.034, 0.610, -0.016, 0.791]
  0%|          | 133/1000000 [00:28<43:06:18,  6.44it/s][INFO] [1770560058.960999734] [a1x_serl_node]: Published EEF command: pos=[0.3203, 0.0065, 0.1886], quat=[0.038, 0.607, -0.014, 0.794]
  0%|          | 134/1000000 [00:28<43:02:45,  6.45it/s][INFO] [1770560059.116004113] [a1x_serl_node]: Published EEF command: pos=[0.3224, 0.0066, 0.1873], quat=[0.036, 0.611, -0.015, 0.791]
  0%|          | 135/1000000 [00:28<42:53:15,  6.48it/s][INFO] [1770560059.269451667] [a1x_serl_node]: Published EEF command: pos=[0.3193, 0.0039, 0.1899], quat=[0.037, 0.609, -0.013, 0.792]
  0%|          | 136/1000000 [00:28<42:53:24,  6.48it/s][INFO] [1770560059.428389630] [a1x_serl_node]: Published EEF command: pos=[0.3198, 0.0024, 0.1881], quat=[0.034, 0.610, -0.014, 0.792]
  0%|          | 137/1000000 [00:28<43:10:34,  6.43it/s][INFO] [1770560059.588267875] [a1x_serl_node]: Published EEF command: pos=[0.3169, 0.0041, 0.1909], quat=[0.036, 0.612, -0.012, 0.790]
  0%|          | 138/1000000 [00:28<43:33:52,  6.38it/s][INFO] [1770560059.741710089] [a1x_serl_node]: Published EEF command: pos=[0.3170, -0.0018, 0.1865], quat=[0.032, 0.609, -0.016, 0.792]
  0%|          | 139/1000000 [00:28<43:15:11,  6.42it/s][INFO] [1770560059.894750311] [a1x_serl_node]: Published EEF command: pos=[0.3180, 0.0017, 0.1898], quat=[0.036, 0.613, -0.012, 0.789]
  0%|          | 140/1000000 [00:29<43:04:20,  6.45it/s]Current EE Pos: [ 0.31782975 -0.00421868  0.19625456], Rot (quat): [ 0.0525994   0.60508108 -0.02714492  0.79396054]
Current EE Pos: [ 0.31793726 -0.00580206  0.19719292], Rot (quat): [ 0.05462088  0.6042712  -0.02886521  0.79438005]
Current EE Pos: [ 0.31675658 -0.00466708  0.19635125], Rot (quat): [ 0.05269501  0.60657355 -0.02755244  0.7928005 ]
Current EE Pos: [ 0.31684419 -0.00505064  0.1960902 ], Rot (quat): [ 0.05288768  0.60662509 -0.02814717  0.79272734]
Current EE Pos: [ 0.3163301  -0.00479175  0.19505741], Rot (quat): [ 0.05297261  0.60832699 -0.02793617  0.79142387]
Current EE Pos: [ 0.3161861  -0.00557213  0.19494548], Rot (quat): [ 0.05346364  0.60833664 -0.02933327  0.79133288]
Current EE Pos: [ 0.31485233 -0.00486786  0.19462544], Rot (quat): [ 0.05237164  0.61111077 -0.02793317  0.78931652]
Current EE Pos: [ 0.31452616 -0.00608729  0.19458989], Rot (quat): [ 0.0518719   0.6122413  -0.02834547  0.78845826]
Current EE Pos: [ 0.31573915 -0.00421718  0.19529468], Rot (quat): [ 0.05241073  0.60979778 -0.02754718  0.79034229]
Current EE Pos: [ 0.31304495 -0.00551304  0.1946869 ], Rot (quat): [ 0.0510801   0.61573869 -0.02726752  0.78582007]
Current EE Pos: [ 0.31625953 -0.00471692  0.1956788 ], Rot (quat): [ 0.05108983  0.60874026 -0.02905381  0.79118961]
Current EE Pos: [ 0.31431219 -0.00505389  0.19524477], Rot (quat): [ 0.04957669  0.61283597 -0.02743779  0.78817599]
Current EE Pos: [ 0.31568769 -0.00463566  0.19611941], Rot (quat): [ 0.05067665  0.60865032 -0.02943264  0.79127138]
Current EE Pos: [ 0.31425007 -0.00440612  0.19604665], Rot (quat): [ 0.05172538  0.61237739 -0.02851588  0.78835605]
Current EE Pos: [ 0.31554273 -0.00286578  0.19619196], Rot (quat): [ 0.0511606   0.6094995  -0.02809209  0.79063505]
Current EE Pos: [ 0.31343986 -0.00189802  0.19635698], Rot (quat): [ 0.05278884  0.6133722  -0.02841563  0.78751535]
Current EE Pos: [ 0.31496646 -0.0008485   0.19799661], Rot (quat): [ 0.05518944  0.6107821  -0.02843482  0.78936101]
Current EE Pos: [3.12703326e-01 1.52312423e-04 1.97130450e-01], Rot (quat): [ 0.05432313  0.6141415  -0.02729595  0.78685078]
Current EE Pos: [0.31223002 0.00048832 0.19774048], Rot (quat): [ 0.05464422  0.61533247 -0.0271111   0.78590391]
Current EE Pos: [0.31208465 0.00062212 0.19668642], Rot (quat): [ 0.05307666  0.61554589 -0.0262054   0.78587493]
Current EE Pos: [0.31184037 0.00117689 0.19685034], Rot (quat): [ 0.05370846  0.61625426 -0.0256191   0.78529596]
Current EE Pos: [0.31192716 0.00148592 0.19604018], Rot (quat): [ 0.05187461  0.6162145  -0.02489344  0.78547376]
Current EE Pos: [0.31267215 0.00203748 0.19608561], Rot (quat): [ 0.05228267  0.61480082 -0.02506857  0.78654818]
Current EE Pos: [0.31233652 0.00214889 0.19589512], Rot (quat): [ 0.0516696   0.61551732 -0.02397319  0.78606232]
Current EE Pos: [0.31394881 0.00139057 0.19647651], Rot (quat): [ 0.05145328  0.61197971 -0.02586075  0.78877412]
Current EE Pos: [0.31254195 0.00188251 0.19572664], Rot (quat): [ 0.05211533  0.61490452 -0.0244011   0.78649921]
Current EE Pos: [0.31298277 0.00194828 0.19565034], Rot (quat): [ 0.05113055  0.61381018 -0.02462047  0.78741131]
Current EE Pos: [0.31210966 0.00232649 0.19497574], Rot (quat): [ 0.05191917  0.61595902 -0.023498    0.78571415]
Current EE Pos: [0.31296428 0.00197059 0.1951072 ], Rot (quat): [ 0.05051111  0.61398874 -0.02438845  0.78731929]
Current EE Pos: [0.31284728 0.00199188 0.19426205], Rot (quat): [ 0.0525567   0.6158176  -0.02228136  0.78581806]
Current EE Pos: [0.31150895 0.00137712 0.19395474], Rot (quat): [ 0.05179994  0.61696682 -0.02482733  0.78489   ]
Current EE Pos: [0.31237551 0.00199373 0.19375934], Rot (quat): [ 0.05090937  0.61597114 -0.0228675   0.78578933]
Current EE Pos: [0.31258457 0.00208176 0.19263152], Rot (quat): [ 0.05146803  0.61739532 -0.02288485  0.78463389]
Current EE Pos: [0.31118036 0.00272581 0.19261371], Rot (quat): [ 0.05230777  0.61866422 -0.02310005  0.78357187]
Current EE Pos: [0.31110309 0.00281396 0.19265253], Rot (quat): [ 0.05247383  0.6182388  -0.02340157  0.78388753]
Current EE Pos: [0.31045613 0.00316465 0.19280941], Rot (quat): [ 0.05073828  0.61799569 -0.02358843  0.78418782]
Current EE Pos: [0.30996857 0.00448591 0.19236205], Rot (quat): [ 0.05219817  0.61968956 -0.02201515  0.78279981]
Current EE Pos: [0.31172289 0.00230418 0.1940092 ], Rot (quat): [ 0.05058881  0.61682993 -0.0249075   0.78507403]
Current EE Pos: [0.31107634 0.00365172 0.1931353 ], Rot (quat): [ 0.05162503  0.61786084 -0.02316713  0.78424876]
Current EE Pos: [0.31155927 0.00334291 0.19315272], Rot (quat): [ 0.04938245  0.61751209 -0.02326303  0.78466492]
Current EE Pos: [0.31113163 0.00369618 0.19265874], Rot (quat): [ 0.05207909  0.61819244 -0.02328591  0.78395385]
Current EE Pos: [0.31233927 0.00350012 0.19290544], Rot (quat): [ 0.05010252  0.6154556  -0.02456258  0.78619389]
Current EE Pos: [0.31189461 0.00349891 0.19283919], Rot (quat): [ 0.05179473  0.61647976 -0.02412082  0.78529497]
Current EE Pos: [0.31299518 0.00339146 0.19268957], Rot (quat): [ 0.05209262  0.61465515 -0.02560541  0.78665734]
Current EE Pos: [0.31159408 0.00416124 0.19202703], Rot (quat): [ 0.05077951  0.61764373 -0.02273318  0.78448765]
Current EE Pos: [0.31369991 0.00437307 0.19215365], Rot (quat): [ 0.05067776  0.61313955 -0.02436005  0.78797096]
Current EE Pos: [0.31141933 0.00530929 0.19121705], Rot (quat): [ 0.04879085  0.61869385 -0.02020663  0.78385526]
Current EE Pos: [0.31270104 0.00556375 0.19070983], Rot (quat): [ 0.05071947  0.61588426 -0.02183447  0.78589908]
Current EE Pos: [0.31177079 0.00582062 0.19080804], Rot (quat): [ 0.04720253  0.61824243 -0.01916345  0.78433474]
Current EE Pos: [0.31250116 0.00613828 0.19034614], Rot (quat): [ 0.04820068  0.61664455 -0.01977091  0.78551595]
Current EE Pos: [0.31248022 0.00616035 0.1904715 ], Rot (quat): [ 0.0445557   0.61634988 -0.01823509  0.78599943]
Current EE Pos: [0.31339746 0.0059045  0.1906994 ], Rot (quat): [ 0.046563    0.6142821  -0.01974469  0.78746399]
Current EE Pos: [0.3137718  0.00635453 0.19104504], Rot (quat): [ 0.04277705  0.61369136 -0.01737754  0.78819481]
Current EE Pos: [0.31415436 0.00582425 0.19099773], Rot (quat): [ 0.04568905  0.61268979 -0.01945117  0.78876193]
Current EE Pos: [0.31400077 0.0055113  0.19070507], Rot (quat): [ 0.04152031  0.613085   -0.01701435  0.78874163]
Current EE Pos: [0.31393512 0.00415306 0.19022116], Rot (quat): [ 0.04339708  0.61345324 -0.02005554  0.78828268]
Current EE Pos: [0.31399836 0.00448755 0.19009491], Rot (quat): [ 0.04156901  0.61326113 -0.0184467   0.78856993]
Current EE Pos: [0.31546255 0.005174   0.19030168], Rot (quat): [ 0.04052429  0.61096946 -0.01932737  0.79038001]
Current EE Pos: [0.3146266  0.00536705 0.18946946], Rot (quat): [ 0.04027209  0.61186385 -0.02022065  0.78967836]
Current EE Pos: [0.3165815  0.00687617 0.18999872], Rot (quat): [ 0.04046403  0.60876174 -0.02043257  0.79205702]
Current EE Pos: [0.31610828 0.00563306 0.18993595], Rot (quat): [ 0.03961288  0.60943441 -0.01985916  0.7915972 ]
Current EE Pos: [0.31661129 0.00627201 0.18961125], Rot (quat): [ 0.03926557  0.60873729 -0.02055651  0.79213292]
Current EE Pos: [0.31647385 0.00844146 0.18981009], Rot (quat): [ 0.03971321  0.60959989 -0.01715955  0.79152788]
Current EE Pos: [0.31633008 0.00815281 0.18870253], Rot (quat): [ 0.03577736  0.61057016 -0.01707107  0.79096943]
Current EE Pos: [0.31835651 0.00863706 0.1891893 ], Rot (quat): [ 0.03917215  0.60721093 -0.01556613  0.79342178]
Current EE Pos: [0.32000393 0.00723285 0.18848663], Rot (quat): [ 0.03423762  0.60561633 -0.01545687  0.79486963]
Current EE Pos: [0.31983524 0.0073547  0.18830479], Rot (quat): [ 0.03766369  0.6062398  -0.01549333  0.79423845]
Current EE Pos: [0.32040335 0.00718744 0.18795802], Rot (quat): [ 0.03554947  0.60606937 -0.01493406  0.79447664]
Current EE Pos: [0.31939198 0.00616009 0.18804127], Rot (quat): [ 0.03673944  0.60754203 -0.01342686  0.79332378]
Current EE Pos: [0.31962587 0.00453991 0.18773536], Rot (quat): [ 0.03446006  0.60753891 -0.01472506  0.79340541]
Current EE Pos: [0.31774343 0.00465146 0.18766558], Rot (quat): [ 0.03579512  0.61094995 -0.01371646  0.79074062]
[INFO] [1770560060.049573360] [a1x_serl_node]: Published EEF command: pos=[0.3189, -0.0005, 0.1854], quat=[0.029, 0.610, -0.014, 0.792]
  0%|          | 141/1000000 [00:29<43:04:23,  6.45it/s][INFO] [1770560060.203919367] [a1x_serl_node]: Published EEF command: pos=[0.3182, 0.0016, 0.1828], quat=[0.035, 0.612, -0.013, 0.790]
  0%|          | 142/1000000 [00:29<42:57:37,  6.46it/s][INFO] [1770560060.357147111] [a1x_serl_node]: Published EEF command: pos=[0.3173, 0.0030, 0.1895], quat=[0.029, 0.610, -0.013, 0.792]
  0%|          | 143/1000000 [00:29<42:54:59,  6.47it/s][INFO] [1770560060.512054920] [a1x_serl_node]: Published EEF command: pos=[0.3185, 0.0077, 0.1860], quat=[0.033, 0.612, -0.011, 0.790]
  0%|          | 144/1000000 [00:29<42:54:33,  6.47it/s][INFO] [1770560060.664907692] [a1x_serl_node]: Published EEF command: pos=[0.3202, 0.0045, 0.1864], quat=[0.028, 0.609, -0.011, 0.792]
  0%|          | 145/1000000 [00:29<42:55:31,  6.47it/s][INFO] [1770560060.822583311] [a1x_serl_node]: Published EEF command: pos=[0.3181, 0.0069, 0.1865], quat=[0.031, 0.611, -0.009, 0.791]
  0%|          | 146/1000000 [00:30<42:58:35,  6.46it/s][INFO] [1770560060.974951171] [a1x_serl_node]: Published EEF command: pos=[0.3182, 0.0060, 0.1868], quat=[0.027, 0.605, -0.008, 0.796]
  0%|          | 147/1000000 [00:30<42:47:49,  6.49it/s][INFO] [1770560061.127244302] [a1x_serl_node]: Published EEF command: pos=[0.3188, 0.0085, 0.1872], quat=[0.030, 0.608, -0.009, 0.793]
  0%|          | 148/1000000 [00:30<42:39:30,  6.51it/s][INFO] [1770560061.281594910] [a1x_serl_node]: Published EEF command: pos=[0.3228, 0.0099, 0.1892], quat=[0.024, 0.606, -0.006, 0.795]
  0%|          | 149/1000000 [00:30<42:42:17,  6.50it/s][INFO] [1770560061.435900237] [a1x_serl_node]: Published EEF command: pos=[0.3181, 0.0075, 0.1880], quat=[0.024, 0.609, -0.009, 0.793]
  0%|          | 150/1000000 [00:30<42:45:31,  6.50it/s][INFO] [1770560061.588302708] [a1x_serl_node]: Published EEF command: pos=[0.3194, 0.0080, 0.1922], quat=[0.022, 0.605, -0.004, 0.796]
  0%|          | 151/1000000 [00:30<42:39:37,  6.51it/s][INFO] [1770560061.740365329] [a1x_serl_node]: Published EEF command: pos=[0.3164, 0.0106, 0.1907], quat=[0.022, 0.607, -0.007, 0.794]
  0%|          | 152/1000000 [00:30<42:29:42,  6.54it/s][INFO] [1770560061.898320818] [a1x_serl_node]: Published EEF command: pos=[0.3198, 0.0085, 0.1893], quat=[0.021, 0.605, -0.004, 0.796]
  0%|          | 153/1000000 [00:31<42:54:46,  6.47it/s][INFO] [1770560062.051122741] [a1x_serl_node]: Published EEF command: pos=[0.3162, 0.0112, 0.1892], quat=[0.021, 0.608, -0.007, 0.794]
  0%|          | 154/1000000 [00:31<42:46:02,  6.49it/s][INFO] [1770560062.215412162] [a1x_serl_node]: Published EEF command: pos=[0.3189, 0.0112, 0.1905], quat=[0.018, 0.608, -0.002, 0.794]
  0%|          | 155/1000000 [00:31<43:36:33,  6.37it/s][INFO] [1770560062.367731863] [a1x_serl_node]: Published EEF command: pos=[0.3163, 0.0092, 0.1916], quat=[0.021, 0.605, -0.006, 0.796]
  0%|          | 156/1000000 [00:31<43:14:30,  6.42it/s]EEF delta: pos=[ 0.00094724  0.0011741  -0.00243109], rot=[-0.00226055  0.01904344  0.00987549], gripper: 0.813 -> 0.807 (80.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.9mm
Step done: False, reward: False, path length: 131, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 2.37606862e-03  9.33286501e-05 -2.15246901e-03], rot=[ 0.00210487 -0.00333643  0.00345037], gripper: 0.813 -> 0.803 (80.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.2mm
Step done: False, reward: False, path length: 132, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00295998 -0.00383174 -0.00570184], rot=[-0.00126264 -0.00096302  0.00410429], gripper: 0.806 -> 0.788 (78.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.5mm
Step done: False, reward: False, path length: 133, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00194443 -0.00211458 -0.00055376], rot=[-0.00035706 -0.0004644   0.00314705], gripper: 0.788 -> 0.776 (77.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.9mm
Step done: False, reward: False, path length: 134, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00240584 -0.00068158 -0.00115217], rot=[ 0.00274637  0.01238027 -0.0016416 ], gripper: 0.776 -> 0.757 (75.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 135, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00055019 -0.00348222  0.00163932], rot=[0.00339138 0.00626155 0.0049671 ], gripper: 0.756 -> 0.748 (74.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 136, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00062571 -0.00474476  0.00017504], rot=[-0.00139826  0.00957623  0.00331655], gripper: 0.748 -> 0.740 (74.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.8mm
Step done: False, reward: False, path length: 137, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00247727 -0.00202173  0.00289765], rot=[0.0006383  0.0119813  0.00255711], gripper: 0.738 -> 0.732 (73.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.3mm
Step done: False, reward: False, path length: 138, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00264436 -0.00629707 -0.00121166], rot=[-0.00473198  0.00450895  0.0005373 ], gripper: 0.733 -> 0.725 (72.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.9mm
Step done: False, reward: False, path length: 139, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00025186 -0.00293683  0.00208969], rot=[0.00281229 0.0053205  0.00338467], gripper: 0.722 -> 0.718 (71.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.6mm
Step done: False, reward: False, path length: 140, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00037857 -0.00187584 -0.00205656], rot=[-0.00153824  0.00346231  0.00645251], gripper: 0.717 -> 0.716 (71.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.8mm
Step done: False, reward: False, path length: 141, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00154946 -0.00066913 -0.00434169], rot=[-0.00101603 -0.00256399  0.00120867], gripper: 0.717 -> 0.721 (72.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.7mm
Step done: False, reward: False, path length: 142, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00169143  0.00162514  0.0019904 ], rot=[7.53179338e-05 5.73338568e-03 3.67262168e-03], gripper: 0.717 -> 0.721 (72.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 143, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00031845  0.00559312 -0.0002342 ], rot=[0.00099486 0.00467284 0.00381699], gripper: 0.717 -> 0.721 (72.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.6mm
Step done: False, reward: False, path length: 144, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00138812  0.0025104  -0.00038474], rot=[0.00050947 0.00227444 0.00497   ], gripper: 0.717 -> 0.725 (72.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.9mm
Step done: False, reward: False, path length: 145, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00069999  0.00246927  0.00011077], rot=[0.00065176 0.00472828 0.00541222], gripper: 0.718 -> 0.722 (72.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 146, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00113848  0.00243501  0.00028593], rot=[ 0.00176342 -0.0069647   0.00592083], gripper: 0.718 -> 0.725 (72.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 147, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00049779  0.00361538  0.00060795], rot=[ 0.00025162 -0.00060655  0.00247255], gripper: 0.718 -> 0.720 (72.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 148, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00268142 0.00486484 0.00208346], rot=[-0.00311158  0.00094381  0.0061663 ], gripper: 0.718 -> 0.721 (72.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.9mm
Step done: False, reward: False, path length: 149, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00156619  0.00114541  0.00087688], rot=[-0.00713787  0.00616253  0.00588224], gripper: 0.718 -> 0.722 (72.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.1mm
Step done: False, reward: False, path length: 150, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0020805   0.00027184  0.00408227], rot=[-0.0018478   0.00205272  0.00669233], gripper: 0.718 -> 0.716 (71.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.6mm
Step done: False, reward: False, path length: 151, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00349372  0.0032237   0.0036473 ], rot=[-0.00203824  0.00195552  0.00362826], gripper: 0.717 -> 0.717 (71.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.0mm
Step done: False, reward: False, path length: 152, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00010947 0.0010144  0.00141814], rot=[ 0.00085851 -0.00101581  0.00511062], gripper: 0.717 -> 0.716 (71.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.7mm
Step done: False, reward: False, path length: 153, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00211998  0.00250003  0.00042876], rot=[-0.00096413  0.00416316  0.00219756], gripper: 0.717 -> 0.717 (71.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.3mm
Step done: False, reward: False, path length: 154, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00050841  0.0022821   0.00130486], rot=[-0.00114418  0.01140393  0.00523603], gripper: 0.717 -> 0.717 (71.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 155, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-2.19655060e-03 -5.63752837e-06  2.72975489e-03], rot=[-0.00042821 -0.00249043  0.00235498], gripper: 0.717 -> 0.715 (71.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.5mm
Step done: False, reward: False, path length: 156, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560062.520582076] [a1x_serl_node]: Published EEF command: pos=[0.3179, 0.0078, 0.1899], quat=[0.016, 0.604, -0.003, 0.797]
  0%|          | 157/1000000 [00:31<42:57:29,  6.47it/s][INFO] [1770560062.676227479] [a1x_serl_node]: Published EEF command: pos=[0.3167, 0.0060, 0.1913], quat=[0.018, 0.606, -0.005, 0.795]
  0%|          | 158/1000000 [00:31<43:09:12,  6.44it/s][INFO] [1770560062.830128015] [a1x_serl_node]: Published EEF command: pos=[0.3204, 0.0098, 0.1919], quat=[0.013, 0.604, -0.002, 0.797]
  0%|          | 159/1000000 [00:32<43:01:16,  6.46it/s][INFO] [1770560062.985519797] [a1x_serl_node]: Published EEF command: pos=[0.3184, 0.0077, 0.1875], quat=[0.015, 0.606, -0.005, 0.796]
  0%|          | 160/1000000 [00:32<42:59:22,  6.46it/s][INFO] [1770560063.138292570] [a1x_serl_node]: Published EEF command: pos=[0.3183, 0.0124, 0.1893], quat=[0.010, 0.601, -0.002, 0.799]
  0%|          | 161/1000000 [00:32<42:49:44,  6.48it/s][INFO] [1770560063.292325597] [a1x_serl_node]: Published EEF command: pos=[0.3222, 0.0094, 0.1918], quat=[0.012, 0.601, -0.002, 0.799]
  0%|          | 162/1000000 [00:32<42:48:58,  6.49it/s][INFO] [1770560063.446417724] [a1x_serl_node]: Published EEF command: pos=[0.3198, 0.0129, 0.1925], quat=[0.002, 0.602, 0.000, 0.799]
  0%|          | 163/1000000 [00:32<42:50:09,  6.48it/s][INFO] [1770560063.599570739] [a1x_serl_node]: Published EEF command: pos=[0.3185, 0.0148, 0.1927], quat=[0.003, 0.603, 0.002, 0.798]
  0%|          | 164/1000000 [00:32<42:55:59,  6.47it/s][INFO] [1770560063.755073831] [a1x_serl_node]: Published EEF command: pos=[0.3154, 0.0132, 0.1970], quat=[-0.002, 0.599, 0.001, 0.801]
  0%|          | 165/1000000 [00:32<42:52:33,  6.48it/s][INFO] [1770560063.912450039] [a1x_serl_node]: Published EEF command: pos=[0.3170, 0.0116, 0.1923], quat=[-0.002, 0.604, 0.004, 0.797]
  0%|          | 166/1000000 [00:33<43:03:10,  6.45it/s][INFO] [1770560064.066101096] [a1x_serl_node]: Published EEF command: pos=[0.3167, 0.0155, 0.1943], quat=[-0.008, 0.599, 0.004, 0.800]
  0%|          | 167/1000000 [00:33<42:57:43,  6.46it/s][INFO] [1770560064.261237431] [a1x_serl_node]: Published EEF command: pos=[0.3146, 0.0168, 0.1941], quat=[-0.011, 0.596, 0.005, 0.803]
  0%|          | 168/1000000 [00:33<46:17:17,  6.00it/s][INFO] [1770560064.414423856] [a1x_serl_node]: Published EEF command: pos=[0.3144, 0.0150, 0.1967], quat=[-0.015, 0.596, 0.007, 0.802]
  0%|          | 169/1000000 [00:33<45:11:00,  6.15it/s][INFO] [1770560064.569216366] [a1x_serl_node]: Published EEF command: pos=[0.3167, 0.0149, 0.1997], quat=[-0.017, 0.595, 0.008, 0.804]
  0%|          | 170/1000000 [00:33<44:31:47,  6.24it/s][INFO] [1770560064.722174900] [a1x_serl_node]: Published EEF command: pos=[0.3127, 0.0181, 0.1967], quat=[-0.021, 0.597, 0.010, 0.802]
  0%|          | 171/1000000 [00:33<43:56:27,  6.32it/s][INFO] [1770560064.876646710] [a1x_serl_node]: Published EEF command: pos=[0.3151, 0.0157, 0.2002], quat=[-0.020, 0.592, 0.007, 0.806]
  0%|          | 172/1000000 [00:34<43:34:47,  6.37it/s][INFO] [1770560065.031199429] [a1x_serl_node]: Published EEF command: pos=[0.3143, 0.0190, 0.1979], quat=[-0.026, 0.596, 0.013, 0.802]
  0%|          | 173/1000000 [00:34<43:24:21,  6.40it/s][INFO] [1770560065.183784042] [a1x_serl_node]: Published EEF command: pos=[0.3156, 0.0176, 0.1997], quat=[-0.024, 0.591, 0.009, 0.806]
  0%|          | 174/1000000 [00:34<43:05:44,  6.44it/s][INFO] [1770560065.341216992] [a1x_serl_node]: Published EEF command: pos=[0.3141, 0.0257, 0.2020], quat=[-0.031, 0.589, 0.012, 0.807]
  0%|          | 175/1000000 [00:34<56:20:16,  4.93it/s][INFO] [1770560065.650601022] [a1x_serl_node]: Published EEF command: pos=[0.3156, 0.0310, 0.2007], quat=[-0.036, 0.587, 0.015, 0.809]
  0%|          | 176/1000000 [00:34<61:46:32,  4.50it/s][INFO] [1770560065.921684224] [a1x_serl_node]: Published EEF command: pos=[0.3165, 0.0316, 0.2025], quat=[-0.037, 0.585, 0.017, 0.810]
  0%|          | 177/1000000 [00:35<56:12:48,  4.94it/s][INFO] [1770560066.079688215] [a1x_serl_node]: Published EEF command: pos=[0.3164, 0.0307, 0.2022], quat=[-0.034, 0.588, 0.015, 0.808]
  0%|          | 178/1000000 [00:35<52:30:40,  5.29it/s][INFO] [1770560066.236627073] [a1x_serl_node]: Published EEF command: pos=[0.3159, 0.0320, 0.2033], quat=[-0.036, 0.586, 0.017, 0.809]
  0%|          | 179/1000000 [00:35<49:52:04,  5.57it/s][INFO] [1770560066.391187423] [a1x_serl_node]: Published EEF command: pos=[0.3157, 0.0302, 0.2009], quat=[-0.038, 0.586, 0.017, 0.809]
  0%|          | 180/1000000 [00:35<47:45:03,  5.82it/s][INFO] [1770560066.545989524] [a1x_serl_node]: Published EEF command: pos=[0.3186, 0.0299, 0.1985], quat=[-0.036, 0.587, 0.020, 0.809]
  0%|          | 181/1000000 [00:35<46:27:22,  5.98it/s][INFO] [1770560066.706627845] [a1x_serl_node]: Published EEF command: pos=[0.3159, 0.0266, 0.2014], quat=[-0.037, 0.586, 0.017, 0.810]
  0%|          | 182/1000000 [00:35<45:44:38,  6.07it/s]EEF delta: pos=[-0.00086645 -0.00191297  0.00138714], rot=[-0.00491743 -0.00198866  0.00258301], gripper: 0.717 -> 0.715 (71.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 157, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00104769 -0.0033616   0.00149363], rot=[-0.00238697  0.00280168  0.00381332], gripper: 0.717 -> 0.713 (71.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 158, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00185921 0.0008815  0.00181123], rot=[-0.00414163  0.00188789  0.00761808], gripper: 0.712 -> 0.706 (70.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 159, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00089408  0.00025897 -0.00289756], rot=[-0.00550087  0.00326427  0.00490799], gripper: 0.703 -> 0.698 (69.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 160, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00061285  0.00366051 -0.00156029], rot=[-0.00498123 -0.0022367   0.00673604], gripper: 0.701 -> 0.690 (69.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.0mm
Step done: False, reward: False, path length: 161, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00335059 0.00075584 0.00135299], rot=[-0.00038025 -0.00353061  0.00500337], gripper: 0.686 -> 0.675 (67.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 162, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[7.23662088e-05 2.80873361e-03 2.08900846e-03], rot=[-0.01094066  0.00371424  0.01578   ], gripper: 0.673 -> 0.670 (67.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.5mm
Step done: False, reward: False, path length: 163, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00116112  0.00475749  0.00210579], rot=[-0.00772012  0.00696324  0.01599949], gripper: 0.670 -> 0.673 (67.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.3mm
Step done: False, reward: False, path length: 164, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00464161  0.0018559   0.00596697], rot=[-0.00726411 -0.00032672  0.00958222], gripper: 0.670 -> 0.670 (67.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.8mm
Step done: False, reward: False, path length: 165, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00180738 -0.00099341  0.00145735], rot=[-0.00704453  0.00683955  0.01143193], gripper: 0.670 -> 0.670 (67.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 166, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00115678  0.00281258  0.00172219], rot=[-0.00824034 -0.00101911  0.0125401 ], gripper: 0.670 -> 0.670 (67.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.5mm
Step done: False, reward: False, path length: 167, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00305426  0.00420299  0.00138613], rot=[-0.00765229 -0.00804385  0.01214593], gripper: 0.670 -> 0.672 (67.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 168, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00321473  0.00076418  0.00340462], rot=[-0.00893347 -0.00506194  0.01404234], gripper: 0.670 -> 0.673 (67.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.7mm
Step done: False, reward: False, path length: 169, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-6.56293472e-04  2.51187012e-05  5.36550302e-03], rot=[-0.00782446 -0.00441543  0.01325025], gripper: 0.670 -> 0.670 (67.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 170, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00390147  0.00307207  0.0015166 ], rot=[-0.00596157  0.00168974  0.01299722], gripper: 0.670 -> 0.669 (66.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 171, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00178184  0.00024856  0.00359304], rot=[-0.00583213 -0.00519514  0.00485627], gripper: 0.670 -> 0.671 (67.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.0mm
Step done: False, reward: False, path length: 172, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00119424  0.00237632  0.00137088], rot=[-0.00590187  0.00249357  0.01390933], gripper: 0.670 -> 0.669 (66.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 173, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00073046  0.0019546   0.00129605], rot=[-0.00452608 -0.00281859  0.00575607], gripper: 0.670 -> 0.669 (66.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 174, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00196031  0.00815041  0.00369922], rot=[-0.00954117 -0.00698074  0.009895  ], gripper: 0.670 -> 0.672 (67.2mm)
⏱️  ✓ 执行耗时=253ms, 误差=8.0mm
Step done: False, reward: False, path length: 175, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00017074  0.0107409   0.00050052], rot=[-0.00623731 -0.00446794  0.01717592], gripper: 0.670 -> 0.674 (67.4mm)
⏱️  ✓ 执行耗时=214ms, 误差=7.9mm
Step done: False, reward: False, path length: 176, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-2.67801806e-05  5.94047550e-03  1.02605065e-03], rot=[-0.00418278 -0.00077747  0.01044196], gripper: 0.670 -> 0.670 (67.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.0mm
Step done: False, reward: False, path length: 177, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[2.81224493e-06 3.53559060e-03 7.00801611e-04], rot=[-0.00013397  0.00628686 -0.00123638], gripper: 0.670 -> 0.670 (67.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.6mm
Step done: False, reward: False, path length: 178, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00069776  0.00274187  0.00138125], rot=[-0.00224714  0.00515207  0.00218542], gripper: 0.670 -> 0.663 (66.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 179, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00065776  0.00119582 -0.00085327], rot=[-0.00687229  0.00358022  0.00717778], gripper: 0.661 -> 0.655 (65.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.6mm
Step done: False, reward: False, path length: 180, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00207676  0.00011109 -0.00315364], rot=[0.00130228 0.00600112 0.00404648], gripper: 0.653 -> 0.650 (65.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 181, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00053089 -0.00319527 -0.00019521], rot=[-0.00061029  0.00327711 -0.00102173], gripper: 0.648 -> 0.643 (64.3mm)
⏱️  ✓ 执行耗时=99ms, 误差=3.2mm
Step done: False, reward: False, path length: 182, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560066.866069851] [a1x_serl_node]: Published EEF command: pos=[0.3176, 0.0244, 0.1969], quat=[-0.034, 0.584, 0.018, 0.811]
  0%|          | 183/1000000 [00:36<45:18:05,  6.13it/s][INFO] [1770560067.019812759] [a1x_serl_node]: Published EEF command: pos=[0.3168, 0.0255, 0.1975], quat=[-0.038, 0.585, 0.017, 0.810]
  0%|          | 184/1000000 [00:36<44:33:47,  6.23it/s][INFO] [1770560067.173987518] [a1x_serl_node]: Published EEF command: pos=[0.3182, 0.0258, 0.1985], quat=[-0.034, 0.585, 0.017, 0.810]
  0%|          | 185/1000000 [00:36<44:01:25,  6.31it/s][INFO] [1770560067.329567302] [a1x_serl_node]: Published EEF command: pos=[0.3185, 0.0256, 0.1965], quat=[-0.039, 0.588, 0.018, 0.808]
  0%|          | 186/1000000 [00:36<43:51:47,  6.33it/s][INFO] [1770560067.486356500] [a1x_serl_node]: Published EEF command: pos=[0.3173, 0.0305, 0.2001], quat=[-0.033, 0.584, 0.018, 0.811]
  0%|          | 187/1000000 [00:36<43:47:09,  6.34it/s][INFO] [1770560067.644893024] [a1x_serl_node]: Published EEF command: pos=[0.3165, 0.0260, 0.1937], quat=[-0.037, 0.588, 0.019, 0.808]
  0%|          | 188/1000000 [00:36<43:45:02,  6.35it/s][INFO] [1770560067.801242650] [a1x_serl_node]: Published EEF command: pos=[0.3160, 0.0250, 0.1977], quat=[-0.035, 0.586, 0.020, 0.809]
  0%|          | 189/1000000 [00:37<43:42:32,  6.35it/s][INFO] [1770560067.955970631] [a1x_serl_node]: Published EEF command: pos=[0.3176, 0.0243, 0.1916], quat=[-0.037, 0.589, 0.019, 0.807]
  0%|          | 190/1000000 [00:37<43:26:24,  6.39it/s][INFO] [1770560068.109995660] [a1x_serl_node]: Published EEF command: pos=[0.3180, 0.0257, 0.1935], quat=[-0.030, 0.592, 0.020, 0.805]
  0%|          | 191/1000000 [00:37<43:15:32,  6.42it/s][INFO] [1770560068.266019846] [a1x_serl_node]: Published EEF command: pos=[0.3191, 0.0228, 0.1917], quat=[-0.036, 0.590, 0.020, 0.806]
  0%|          | 192/1000000 [00:37<43:19:22,  6.41it/s][INFO] [1770560068.456567748] [a1x_serl_node]: Published EEF command: pos=[0.3177, 0.0272, 0.1935], quat=[-0.032, 0.596, 0.025, 0.802]
  0%|          | 193/1000000 [00:37<46:08:07,  6.02it/s][INFO] [1770560068.612035432] [a1x_serl_node]: Published EEF command: pos=[0.3193, 0.0221, 0.1919], quat=[-0.034, 0.589, 0.019, 0.807]
  0%|          | 194/1000000 [00:37<45:15:03,  6.14it/s][INFO] [1770560068.766935084] [a1x_serl_node]: Published EEF command: pos=[0.3141, 0.0227, 0.1891], quat=[-0.034, 0.596, 0.026, 0.802]
  0%|          | 195/1000000 [00:37<44:34:21,  6.23it/s][INFO] [1770560068.920046860] [a1x_serl_node]: Published EEF command: pos=[0.3185, 0.0230, 0.1925], quat=[-0.032, 0.591, 0.018, 0.806]
  0%|          | 196/1000000 [00:38<43:58:19,  6.32it/s][INFO] [1770560069.073570477] [a1x_serl_node]: Published EEF command: pos=[0.3177, 0.0221, 0.1911], quat=[-0.034, 0.594, 0.024, 0.803]
  0%|          | 197/1000000 [00:38<43:36:59,  6.37it/s][INFO] [1770560069.229193182] [a1x_serl_node]: Published EEF command: pos=[0.3160, 0.0211, 0.1921], quat=[-0.026, 0.593, 0.016, 0.805]
  0%|          | 198/1000000 [00:38<43:28:09,  6.39it/s][INFO] [1770560069.383761893] [a1x_serl_node]: Published EEF command: pos=[0.3175, 0.0254, 0.1766], quat=[-0.034, 0.595, 0.025, 0.803]
  0%|          | 199/1000000 [00:38<73:46:42,  3.76it/s][INFO] [1770560069.905911803] [a1x_serl_node]: Published EEF command: pos=[0.3172, 0.0240, 0.1823], quat=[-0.027, 0.599, 0.025, 0.800]
  0%|          | 200/1000000 [00:39<64:39:55,  4.29it/s][INFO] [1770560070.063261084] [a1x_serl_node]: Published EEF command: pos=[0.3179, 0.0243, 0.1839], quat=[-0.029, 0.597, 0.024, 0.801]
  0%|          | 201/1000000 [00:39<58:21:39,  4.76it/s][INFO] [1770560070.217350514] [a1x_serl_node]: Published EEF command: pos=[0.3185, 0.0267, 0.1824], quat=[-0.022, 0.599, 0.025, 0.800]
  0%|          | 202/1000000 [00:39<53:41:42,  5.17it/s][INFO] [1770560070.372010916] [a1x_serl_node]: Published EEF command: pos=[0.3179, 0.0273, 0.1839], quat=[-0.024, 0.602, 0.025, 0.798]
  0%|          | 203/1000000 [00:39<50:31:20,  5.50it/s][INFO] [1770560070.527067139] [a1x_serl_node]: Published EEF command: pos=[0.3170, 0.0314, 0.1811], quat=[-0.015, 0.603, 0.028, 0.797]
  0%|          | 204/1000000 [00:39<48:16:03,  5.75it/s][INFO] [1770560070.680122076] [a1x_serl_node]: Published EEF command: pos=[0.3180, 0.0296, 0.1833], quat=[-0.016, 0.601, 0.025, 0.798]
  0%|          | 205/1000000 [00:39<46:29:44,  5.97it/s][INFO] [1770560070.833661094] [a1x_serl_node]: Published EEF command: pos=[0.3136, 0.0331, 0.1823], quat=[-0.011, 0.604, 0.030, 0.797]
  0%|          | 206/1000000 [00:40<45:20:28,  6.13it/s][INFO] [1770560070.985959488] [a1x_serl_node]: Published EEF command: pos=[0.3161, 0.0318, 0.1840], quat=[-0.015, 0.599, 0.028, 0.800]
  0%|          | 207/1000000 [00:40<44:27:24,  6.25it/s][INFO] [1770560071.146240469] [a1x_serl_node]: Published EEF command: pos=[0.3169, 0.0364, 0.1887], quat=[-0.005, 0.600, 0.022, 0.800]
  0%|          | 208/1000000 [00:40<57:11:05,  4.86it/s]EEF delta: pos=[ 5.54826111e-05 -6.08847197e-03 -3.87285231e-03], rot=[-0.00069041  0.00046114 -0.00454807], gripper: 0.646 -> 0.640 (64.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.2mm
Step done: False, reward: False, path length: 183, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00035972 -0.00378931 -0.00351868], rot=[-0.00336033  0.00115643  0.00093214], gripper: 0.640 -> 0.632 (63.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 184, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00039651 -0.00110249 -0.00183668], rot=[ 0.00199163  0.00750852 -0.00168228], gripper: 0.630 -> 0.624 (62.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.2mm
Step done: False, reward: False, path length: 185, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00059219 -0.0014048  -0.00341679], rot=[-0.00065738  0.01087283  0.00512894], gripper: 0.622 -> 0.615 (61.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 186, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00115706  0.00391336  0.00052535], rot=[ 0.00420004  0.0044784  -0.00026129], gripper: 0.615 -> 0.603 (60.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 187, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0010575  -0.0008961  -0.00501569], rot=[0.00227813 0.00886856 0.00309234], gripper: 0.605 -> 0.596 (59.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 188, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00172382 -0.00340116 -0.00128477], rot=[-0.00169615  0.00622564  0.00475257], gripper: 0.598 -> 0.584 (58.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.0mm
Step done: False, reward: False, path length: 189, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00044254 -0.00293159 -0.00612881], rot=[0.00035438 0.00784027 0.00114046], gripper: 0.583 -> 0.570 (57.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.8mm
Step done: False, reward: False, path length: 190, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.0009979  -0.00147003 -0.00432386], rot=[ 0.01096275  0.01621583 -0.00556512], gripper: 0.569 -> 0.560 (56.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.7mm
Step done: False, reward: False, path length: 191, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00117077 -0.0032399  -0.00420021], rot=[ 0.00337186  0.00989706 -0.00032495], gripper: 0.559 -> 0.551 (55.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 192, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 8.73620156e-05  9.24623630e-04 -1.30975968e-03], rot=[0.00534693 0.01884214 0.00595455], gripper: 0.551 -> 0.545 (54.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.6mm
Step done: False, reward: False, path length: 193, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.0006021  -0.00253889 -0.0020104 ], rot=[ 0.00230439  0.0034856  -0.00059097], gripper: 0.544 -> 0.533 (53.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.3mm
Step done: False, reward: False, path length: 194, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00258233 -0.00422539 -0.00362747], rot=[-0.00084086  0.0096937   0.00623769], gripper: 0.529 -> 0.516 (51.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.1mm
Step done: False, reward: False, path length: 195, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00010732 -0.00225238 -0.00075283], rot=[ 0.000698    0.00719623 -0.00620612], gripper: 0.510 -> 0.501 (50.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.4mm
Step done: False, reward: False, path length: 196, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00179365 -0.00288978 -0.00022545], rot=[-0.00083571  0.00159696 -0.00149222], gripper: 0.494 -> 0.487 (48.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.4mm
Step done: False, reward: False, path length: 197, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00183858 -0.00277873  0.00012282], rot=[ 0.00662908  0.00738934 -0.01129271], gripper: 0.478 -> 0.474 (47.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.3mm
Step done: False, reward: False, path length: 198, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00012839  0.00186216 -0.01490017], rot=[0.00058222 0.00775475 0.00246822], gripper: 0.464 -> 0.467 (46.7mm)
⏱️  ✓ 执行耗时=465ms, 误差=7.9mm
Step done: False, reward: False, path length: 199, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00254982 -0.00086834 -0.00161353], rot=[ 0.01061029  0.01078529 -0.00750256], gripper: 0.457 -> 0.453 (45.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 200, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00177896 -0.00082303  0.00075585], rot=[ 0.00597329  0.00482527 -0.00763424], gripper: 0.445 -> 0.433 (43.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.1mm
Step done: False, reward: False, path length: 201, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00055996  0.00194248 -0.0001006 ], rot=[ 0.01122703  0.00649137 -0.00830681], gripper: 0.422 -> 0.412 (41.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 202, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00124884  0.00295297  0.00126691], rot=[ 0.00949334  0.01365913 -0.00362941], gripper: 0.402 -> 0.404 (40.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.4mm
Step done: False, reward: False, path length: 203, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00196901  0.00607909 -0.00138915], rot=[ 0.01672158  0.01456713 -0.00641874], gripper: 0.399 -> 0.400 (40.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.5mm
Step done: False, reward: False, path length: 204, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.000145   0.00401581 0.00141686], rot=[ 0.011528    0.00545522 -0.01050777], gripper: 0.393 -> 0.393 (39.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.3mm
Step done: False, reward: False, path length: 205, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00397234  0.00563044  0.0008484 ], rot=[ 0.01138504  0.00901561 -0.00299049], gripper: 0.385 -> 0.387 (38.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.9mm
Step done: False, reward: False, path length: 206, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00156616  0.00480571  0.00235531], rot=[ 0.00440906 -0.00152667 -0.00065514], gripper: 0.375 -> 0.371 (37.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.6mm
Step done: False, reward: False, path length: 207, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00109972 0.0065148  0.00736323], rot=[ 0.00127809 -0.00626843 -0.01915217], gripper: 0.359 -> 0.359 (35.9mm)
⏱️  ✓ 执行耗时=255ms, 误差=7.8mm
Step done: False, reward: False, path length: 208, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560071.454393359] [a1x_serl_node]: Published EEF command: pos=[0.3174, 0.0362, 0.1882], quat=[-0.002, 0.596, 0.022, 0.803]
  0%|          | 209/1000000 [00:40<53:02:04,  5.24it/s][INFO] [1770560071.611672200] [a1x_serl_node]: Published EEF command: pos=[0.3181, 0.0423, 0.1863], quat=[-0.004, 0.601, 0.028, 0.799]
  0%|          | 210/1000000 [00:40<62:05:36,  4.47it/s][INFO] [1770560071.909261605] [a1x_serl_node]: Published EEF command: pos=[0.3188, 0.0432, 0.1897], quat=[-0.002, 0.596, 0.028, 0.803]
  0%|          | 211/1000000 [00:41<56:19:55,  4.93it/s]Current EE Pos: [0.31847466 0.00134859 0.18741395], Rot (quat): [ 0.03189483  0.60829374 -0.01554915  0.79291845]
Current EE Pos: [0.31661004 0.00228175 0.18715192], Rot (quat): [ 0.03536964  0.61265475 -0.0135701   0.7894422 ]
Current EE Pos: [0.31901236 0.00139548 0.1874872 ], Rot (quat): [ 0.03017416  0.60738812 -0.01464728  0.79369682]
Current EE Pos: [0.31877011 0.00208235 0.18619527], Rot (quat): [ 0.03349044  0.60962874 -0.01315544  0.79187002]
Current EE Pos: [0.3188367  0.00202682 0.18676912], Rot (quat): [ 0.02939024  0.60844443 -0.01328507  0.79294079]
Current EE Pos: [0.31875652 0.00443537 0.18640849], Rot (quat): [ 0.03259634  0.60904473 -0.01172535  0.79237902]
Current EE Pos: [0.31933453 0.00356532 0.18655441], Rot (quat): [ 0.02811631  0.60776487 -0.01086698  0.79354473]
Current EE Pos: [0.31926575 0.00491483 0.18661784], Rot (quat): [ 0.03064918  0.6079927  -0.00991889  0.7932888 ]
Current EE Pos: [0.32016524 0.00499617 0.18714475], Rot (quat): [ 0.02665046  0.6055681  -0.0077823   0.79530904]
Current EE Pos: [0.31971462 0.00635244 0.18709835], Rot (quat): [ 0.02914741  0.60643273 -0.00902952  0.79454908]
Current EE Pos: [0.32147537 0.00773359 0.18811741], Rot (quat): [ 0.02446818  0.60368515 -0.00652842  0.79682051]
Current EE Pos: [0.31986162 0.00736636 0.18710031], Rot (quat): [ 0.02419074  0.6065278  -0.00778296  0.79465606]
Current EE Pos: [0.31972519 0.00749986 0.18791702], Rot (quat): [ 0.02214491  0.60553507 -0.0060864   0.79548717]
Current EE Pos: [0.31831216 0.00874263 0.18877642], Rot (quat): [ 0.0225094   0.60628738 -0.00733697  0.79489314]
Current EE Pos: [0.31943119 0.00892215 0.18916368], Rot (quat): [ 0.02042786  0.60355301 -0.00389794  0.79705161]
Current EE Pos: [0.31847942 0.00919923 0.18889685], Rot (quat): [ 0.02159377  0.60562772 -0.0064753   0.79542872]
Current EE Pos: [0.31871867 0.00974291 0.18850987], Rot (quat): [ 0.01907961  0.60510031 -0.00277591  0.79591575]
Current EE Pos: [0.3177967  0.00940784 0.18980789], Rot (quat): [ 0.02047743  0.604705   -0.0056832   0.79616596]
Current EE Pos: [0.31859032 0.00895694 0.19009775], Rot (quat): [ 0.01682828  0.6028335  -0.00377286  0.79768061]
Current EE Pos: [0.31751736 0.00744216 0.19035208], Rot (quat): [ 0.01819373  0.60440294 -0.00570227  0.7964506 ]
Current EE Pos: [0.31889822 0.00872243 0.19082177], Rot (quat): [ 0.01403632  0.60161432 -0.00342061  0.79865605]
Current EE Pos: [0.31885631 0.0086105  0.1904873 ], Rot (quat): [ 0.01386029  0.60244791 -0.00400462  0.7980278 ]
Current EE Pos: [0.31972257 0.01013871 0.1904311 ], Rot (quat): [ 0.01108053  0.60014489 -0.00277053  0.79980976]
Current EE Pos: [0.31968474 0.00999663 0.19063091], Rot (quat): [ 0.01136971  0.60042388 -0.0019932   0.7995986 ]
Current EE Pos: [0.32002922 0.01130166 0.19099523], Rot (quat): [ 4.09030144e-03  5.99308169e-01 -5.93918577e-04  8.00507736e-01]
Current EE Pos: [0.31876915 0.01258516 0.19082177], Rot (quat): [0.00411    0.60156162 0.00130658 0.79881476]
Current EE Pos: [0.31783699 0.01267357 0.19255509], Rot (quat): [-0.00120374  0.59974589  0.00136893  0.80018844]
Current EE Pos: [0.31763003 0.01257929 0.1927465 ], Rot (quat): [-0.004267    0.59969732  0.00204969  0.80021292]
Current EE Pos: [0.31766251 0.01423415 0.19327986], Rot (quat): [-0.00755876  0.59860833  0.003803    0.80099717]
Current EE Pos: [0.31735689 0.01487787 0.19431005], Rot (quat): [-0.01012549  0.59666984  0.00464046  0.80240953]
Current EE Pos: [0.31655979 0.01506484 0.19514992], Rot (quat): [-0.01441963  0.59650532  0.00628914  0.80245494]
Current EE Pos: [0.31687174 0.01549077 0.19657442], Rot (quat): [-0.01665407  0.5943157   0.00722387  0.80402693]
Current EE Pos: [0.3155035  0.01662689 0.19657386], Rot (quat): [-0.01971819  0.59534283  0.0090492   0.80317882]
Current EE Pos: [0.31635953 0.01569277 0.19836262], Rot (quat): [-0.02014618  0.59190276  0.0076766   0.80572099]
Current EE Pos: [0.31601725 0.01751782 0.19832509], Rot (quat): [-0.02410003  0.59205319  0.01105142  0.80546264]
Current EE Pos: [0.31573006 0.02030098 0.20021918], Rot (quat): [-0.0283393   0.58916914  0.01039158  0.80744574]
Current EE Pos: [0.31648293 0.02567594 0.2015061 ], Rot (quat): [-0.03240094  0.58529849  0.01409148  0.81004771]
Current EE Pos: [0.31635535 0.02720345 0.20150994], Rot (quat): [-0.0338855   0.58519685  0.01562858  0.8100322 ]
Current EE Pos: [0.31664197 0.02929717 0.20188441], Rot (quat): [-0.03481371  0.58413215  0.01705509  0.81073224]
Current EE Pos: [0.31638532 0.02901858 0.20171632], Rot (quat): [-0.03361672  0.58478979  0.01644974  0.81032106]
Current EE Pos: [0.3164785  0.02980018 0.20170134], Rot (quat): [-0.03513957  0.58428985  0.01782496  0.81058797]
Current EE Pos: [0.31641222 0.0297755  0.20158481], Rot (quat): [-0.03715146  0.58426207  0.01773121  0.81052033]
Current EE Pos: [0.31758215 0.03043921 0.20080995], Rot (quat): [-0.03540739  0.58346203  0.01970722  0.81112884]
Current EE Pos: [0.31714604 0.02928758 0.20104935], Rot (quat): [-0.03661786  0.58404773  0.01726021  0.81070924]
Current EE Pos: [0.31864301 0.02688307 0.20030788], Rot (quat): [-0.03495322  0.58170433  0.01742116  0.81246221]
Current EE Pos: [0.3178678  0.02700053 0.1998961 ], Rot (quat): [-0.03725406  0.5835132   0.01609803  0.81108898]
Current EE Pos: [0.31843743 0.02661017 0.19961845], Rot (quat): [-0.03436918  0.5825205   0.01725105  0.8119058 ]
Current EE Pos: [0.31755143 0.02685168 0.1987429 ], Rot (quat): [-0.03735729  0.58475592  0.01682144  0.81017405]
Current EE Pos: [0.31776597 0.02836071 0.19897594], Rot (quat): [-0.03321511  0.58388862  0.01799907  0.81095429]
Current EE Pos: [0.31718616 0.02719588 0.19769474], Rot (quat): [-0.03705757  0.5859182   0.01803532  0.80932152]
Current EE Pos: [0.31705034 0.02719929 0.1977856 ], Rot (quat): [-0.03580637  0.58574657  0.01869189  0.80948717]
Current EE Pos: [0.31793006 0.02601508 0.19589268], Rot (quat): [-0.03733151  0.58609256  0.01876882  0.80916599]
Current EE Pos: [0.31762671 0.02628494 0.19483495], Rot (quat): [-0.03280059  0.58832703  0.02034415  0.80770139]
Current EE Pos: [0.31867283 0.02468533 0.1938903 ], Rot (quat): [-0.03501191  0.58757342  0.01887054  0.80819276]
Current EE Pos: [0.31667186 0.02693159 0.1926832 ], Rot (quat): [-0.03225066  0.59210045  0.02329014  0.80488168]
Current EE Pos: [0.31864519 0.02526882 0.19328502], Rot (quat): [-0.03439045  0.58777993  0.02005427  0.80804076]
Current EE Pos: [0.31592934 0.0250057  0.19131079], Rot (quat): [-0.03363571  0.59368045  0.02480025  0.80361503]
Current EE Pos: [0.31787311 0.02392408 0.19198112], Rot (quat): [-0.03258706  0.58977447  0.01868246  0.80669395]
Current EE Pos: [0.31732279 0.02354263 0.1915215 ], Rot (quat): [-0.03325738  0.59184491  0.02360623  0.80501943]
Current EE Pos: [0.31971787 0.02490289 0.18389028], Rot (quat): [-0.03328093  0.59447551  0.024482    0.8030516 ]
Current EE Pos: [0.31969273 0.02510616 0.18317703], Rot (quat): [-0.03369378  0.59539789  0.02484382  0.80233962]
Current EE Pos: [0.31905285 0.02478321 0.1825165 ], Rot (quat): [-0.02903254  0.59688848  0.02486764  0.80141303]
Current EE Pos: [0.31915399 0.02431822 0.18261077], Rot (quat): [-0.02933634  0.59661445  0.0237658   0.80163942]
Current EE Pos: [0.31894868 0.02532393 0.18252357], Rot (quat): [-0.02336016  0.59712315  0.02539383  0.80140714]
Current EE Pos: [0.31784049 0.02557376 0.18193048], Rot (quat): [-0.02430228  0.59930186  0.0257682   0.79973913]
Current EE Pos: [0.31754258 0.02745097 0.18143138], Rot (quat): [-0.01618328  0.60008456  0.02768373  0.79929358]
Current EE Pos: [0.31769735 0.02703898 0.1816714 ], Rot (quat): [-0.01705     0.59994875  0.02654037  0.79941629]
Current EE Pos: [0.31579863 0.02993042 0.18136312], Rot (quat): [-0.01103901  0.60222151  0.02942215  0.7977103 ]
Current EE Pos: [0.3160532  0.03188536 0.18407073], Rot (quat): [-0.00512499  0.59976377  0.02353158  0.79981462]
Current EE Pos: [0.31622599 0.03233382 0.18529329], Rot (quat): [-0.0051641   0.59823179  0.0237439   0.80095461]
Current EE Pos: [0.31695307 0.03626672 0.18639539], Rot (quat): [-0.00207643  0.59667536  0.02549805  0.80207485]
[INFO] [1770560072.061471609] [a1x_serl_node]: Published EEF command: pos=[0.3191, 0.0437, 0.1860], quat=[-0.003, 0.597, 0.031, 0.802]
  0%|          | 212/1000000 [00:41<52:13:03,  5.32it/s][INFO] [1770560072.218521399] [a1x_serl_node]: Published EEF command: pos=[0.3163, 0.0461, 0.1847], quat=[-0.002, 0.598, 0.031, 0.801]
  0%|          | 213/1000000 [00:41<49:31:38,  5.61it/s][INFO] [1770560072.372350379] [a1x_serl_node]: Published EEF command: pos=[0.3208, 0.0490, 0.1876], quat=[-0.004, 0.593, 0.031, 0.804]
  0%|          | 214/1000000 [00:41<47:30:37,  5.85it/s][INFO] [1770560072.527104292] [a1x_serl_node]: Published EEF command: pos=[0.3185, 0.0472, 0.1864], quat=[-0.004, 0.593, 0.032, 0.805]
  0%|          | 215/1000000 [00:41<46:07:05,  6.02it/s][INFO] [1770560072.682040795] [a1x_serl_node]: Published EEF command: pos=[0.3210, 0.0470, 0.1864], quat=[-0.005, 0.592, 0.036, 0.805]
  0%|          | 216/1000000 [00:41<45:12:36,  6.14it/s][INFO] [1770560072.835906255] [a1x_serl_node]: Published EEF command: pos=[0.3217, 0.0450, 0.1848], quat=[-0.004, 0.593, 0.035, 0.805]
  0%|          | 217/1000000 [00:42<44:30:16,  6.24it/s][INFO] [1770560072.990765759] [a1x_serl_node]: Published EEF command: pos=[0.3220, 0.0477, 0.1855], quat=[-0.004, 0.592, 0.037, 0.805]
  0%|          | 218/1000000 [00:42<44:03:47,  6.30it/s][INFO] [1770560073.150805929] [a1x_serl_node]: Published EEF command: pos=[0.3230, 0.0472, 0.1871], quat=[-0.002, 0.592, 0.035, 0.805]
  0%|          | 219/1000000 [00:42<44:04:16,  6.30it/s][INFO] [1770560073.304313178] [a1x_serl_node]: Published EEF command: pos=[0.3182, 0.0478, 0.1875], quat=[-0.005, 0.591, 0.037, 0.805]
  0%|          | 220/1000000 [00:42<43:41:36,  6.36it/s][INFO] [1770560073.456798884] [a1x_serl_node]: Published EEF command: pos=[0.3237, 0.0465, 0.1862], quat=[-0.001, 0.593, 0.035, 0.804]
  0%|          | 221/1000000 [00:42<43:15:53,  6.42it/s][INFO] [1770560073.609046989] [a1x_serl_node]: Published EEF command: pos=[0.3239, 0.0435, 0.1837], quat=[-0.002, 0.590, 0.038, 0.807]
  0%|          | 222/1000000 [00:42<42:59:53,  6.46it/s][INFO] [1770560073.764770985] [a1x_serl_node]: Published EEF command: pos=[0.3237, 0.0495, 0.1903], quat=[0.003, 0.589, 0.031, 0.807]
  0%|          | 223/1000000 [00:42<43:11:24,  6.43it/s][INFO] [1770560073.919527418] [a1x_serl_node]: Published EEF command: pos=[0.3271, 0.0436, 0.1850], quat=[-0.004, 0.584, 0.039, 0.811]
  0%|          | 224/1000000 [00:43<43:00:59,  6.46it/s][INFO] [1770560074.071690963] [a1x_serl_node]: Published EEF command: pos=[0.3242, 0.0472, 0.1867], quat=[0.005, 0.590, 0.031, 0.807]
  0%|          | 225/1000000 [00:43<42:48:24,  6.49it/s][INFO] [1770560074.226246176] [a1x_serl_node]: Published EEF command: pos=[0.3282, 0.0463, 0.1866], quat=[0.001, 0.584, 0.038, 0.811]
  0%|          | 226/1000000 [00:43<42:56:57,  6.47it/s][INFO] [1770560074.385058313] [a1x_serl_node]: Published EEF command: pos=[0.3292, 0.0528, 0.1897], quat=[0.007, 0.588, 0.030, 0.808]
  0%|          | 227/1000000 [00:43<43:11:03,  6.43it/s][INFO] [1770560074.542247145] [a1x_serl_node]: Published EEF command: pos=[0.3288, 0.0475, 0.1878], quat=[0.001, 0.583, 0.038, 0.811]
  0%|          | 228/1000000 [00:43<43:17:16,  6.42it/s][INFO] [1770560074.695956015] [a1x_serl_node]: Published EEF command: pos=[0.3293, 0.0469, 0.1881], quat=[0.009, 0.586, 0.029, 0.810]
  0%|          | 229/1000000 [00:43<43:05:37,  6.44it/s][INFO] [1770560074.849232493] [a1x_serl_node]: Published EEF command: pos=[0.3292, 0.0493, 0.1889], quat=[0.001, 0.582, 0.037, 0.812]
  0%|          | 230/1000000 [00:44<42:59:52,  6.46it/s][INFO] [1770560075.003363535] [a1x_serl_node]: Published EEF command: pos=[0.3305, 0.0488, 0.1895], quat=[0.008, 0.582, 0.028, 0.813]
  0%|          | 231/1000000 [00:44<42:54:23,  6.47it/s][INFO] [1770560075.155753831] [a1x_serl_node]: Published EEF command: pos=[0.3313, 0.0463, 0.1889], quat=[0.002, 0.580, 0.034, 0.814]
  0%|          | 232/1000000 [00:44<42:43:13,  6.50it/s][INFO] [1770560075.308294157] [a1x_serl_node]: Published EEF command: pos=[0.3314, 0.0483, 0.1920], quat=[0.009, 0.583, 0.027, 0.812]
  0%|          | 233/1000000 [00:44<42:37:34,  6.52it/s][INFO] [1770560075.462397189] [a1x_serl_node]: Published EEF command: pos=[0.3322, 0.0489, 0.1903], quat=[0.001, 0.581, 0.032, 0.813]
  0%|          | 234/1000000 [00:44<42:40:39,  6.51it/s]EEF delta: pos=[0.00133747 0.00428602 0.00408025], rot=[ 0.00392273 -0.01030988 -0.00731624], gripper: 0.346 -> 0.345 (34.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.1mm
Step done: False, reward: False, path length: 209, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00183485 0.01000317 0.00101213], rot=[0.00739719 0.00681592 0.0055476 ], gripper: 0.339 -> 0.347 (34.7mm)
⏱️  ✓ 执行耗时=243ms, 误差=7.9mm
Step done: False, reward: False, path length: 210, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00188198 0.00691855 0.00328966], rot=[ 0.00215515 -0.00266522  0.00394328], gripper: 0.339 -> 0.350 (35.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.9mm
Step done: False, reward: False, path length: 211, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00207359 0.00526883 0.00014335], rot=[ 0.00406444 -0.00116647  0.00365161], gripper: 0.339 -> 0.349 (34.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.7mm
Step done: False, reward: False, path length: 212, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0017892   0.00597521 -0.00239386], rot=[0.00362843 0.00846217 0.00343351], gripper: 0.339 -> 0.347 (34.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.7mm
Step done: False, reward: False, path length: 213, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.0019409  0.00751388 0.00078138], rot=[-0.00176991 -0.00204543  0.00221157], gripper: 0.339 -> 0.346 (34.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=7.8mm
Step done: False, reward: False, path length: 214, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.0005899  0.00423225 0.00030997], rot=[-0.00391714 -0.00778031  0.00317655], gripper: 0.339 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.3mm
Step done: False, reward: False, path length: 215, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00109377  0.00157674 -0.00080682], rot=[0.00286327 0.0010797  0.00709026], gripper: 0.338 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.1mm
Step done: False, reward: False, path length: 216, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00189126 -0.00049655 -0.00233551], rot=[0.00212127 0.00334191 0.00421786], gripper: 0.338 -> 0.344 (34.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 217, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00138558  0.00104147 -0.00187866], rot=[0.00330646 0.00444476 0.00277979], gripper: 0.338 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 218, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00215437 0.00113873 0.00028366], rot=[ 0.00532757  0.00521665 -0.00259257], gripper: 0.338 -> 0.339 (33.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 219, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.0026326   0.00116478  0.00071222], rot=[0.00037839 0.00284275 0.00249208], gripper: 0.338 -> 0.341 (34.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 220, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00224577 -0.00010945 -0.00047557], rot=[ 0.0011125   0.00744732 -0.00312091], gripper: 0.338 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.3mm
Step done: False, reward: False, path length: 221, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00272166 -0.00349422 -0.0029446 ], rot=[ 5.09504089e-03 -2.69995071e-05 -1.10907224e-03], gripper: 0.338 -> 0.342 (34.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.3mm
Step done: False, reward: False, path length: 222, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00223093 0.00334267 0.00415884], rot=[ 0.00314852 -0.00200574 -0.01155918], gripper: 0.338 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.8mm
Step done: False, reward: False, path length: 223, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00385697 -0.00235408 -0.00117648], rot=[-0.00089163 -0.00903135  0.00454907], gripper: 0.338 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.7mm
Step done: False, reward: False, path length: 224, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00134784  0.00029188 -0.00062907], rot=[ 0.0019736   0.00266662 -0.00242413], gripper: 0.338 -> 0.341 (34.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.5mm
Step done: False, reward: False, path length: 225, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 1.99437910e-03 -1.77512993e-06 -1.45267742e-03], rot=[ 0.00664863  0.00677516 -0.00463382], gripper: 0.338 -> 0.338 (33.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 226, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00526066 0.00623227 0.00247709], rot=[ 0.00143239  0.00316442 -0.00446017], gripper: 0.338 -> 0.342 (34.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=8.0mm
Step done: False, reward: False, path length: 227, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00213984 0.0008915  0.00038841], rot=[ 0.00210515  0.00289989 -0.00139523], gripper: 0.338 -> 0.338 (33.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.4mm
Step done: False, reward: False, path length: 228, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[2.86061922e-03 4.54811379e-06 8.83337227e-04], rot=[-1.77070615e-05  2.36505014e-03 -6.02524262e-03], gripper: 0.338 -> 0.344 (34.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 229, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00069544 0.00123862 0.00089796], rot=[-0.00250582  0.00503052 -0.00452624], gripper: 0.338 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.7mm
Step done: False, reward: False, path length: 230, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00320144 0.00212186 0.0025006 ], rot=[-0.00260432 -0.00414184 -0.00592703], gripper: 0.338 -> 0.344 (34.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.6mm
Step done: False, reward: False, path length: 231, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00239828 -0.00080922  0.00112741], rot=[-2.80462974e-03 -2.28348654e-05 -4.47806297e-03], gripper: 0.338 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.8mm
Step done: False, reward: False, path length: 232, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.0026654  0.0014871  0.00415196], rot=[ 2.10227736e-05  4.26210323e-03 -7.67197507e-03], gripper: 0.338 -> 0.344 (34.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 233, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00210407 0.00246953 0.00168369], rot=[-0.00345212  0.00731176 -0.0039029 ], gripper: 0.338 -> 0.341 (34.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 234, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560075.615409807] [a1x_serl_node]: Published EEF command: pos=[0.3309, 0.0462, 0.1892], quat=[0.010, 0.584, 0.023, 0.811]
  0%|          | 235/1000000 [00:44<42:34:21,  6.52it/s][INFO] [1770560075.771515135] [a1x_serl_node]: Published EEF command: pos=[0.3303, 0.0470, 0.1890], quat=[0.003, 0.579, 0.030, 0.815]
  0%|          | 236/1000000 [00:44<42:51:46,  6.48it/s][INFO] [1770560075.924513293] [a1x_serl_node]: Published EEF command: pos=[0.3331, 0.0442, 0.1893], quat=[0.011, 0.583, 0.023, 0.812]
  0%|          | 237/1000000 [00:45<42:42:48,  6.50it/s][INFO] [1770560076.077272920] [a1x_serl_node]: Published EEF command: pos=[0.3304, 0.0442, 0.1914], quat=[0.007, 0.581, 0.030, 0.813]
  0%|          | 238/1000000 [00:45<42:47:55,  6.49it/s][INFO] [1770560076.232691077] [a1x_serl_node]: Published EEF command: pos=[0.3285, 0.0446, 0.1878], quat=[0.013, 0.583, 0.022, 0.812]
  0%|          | 239/1000000 [00:45<42:51:44,  6.48it/s][INFO] [1770560076.387710902] [a1x_serl_node]: Published EEF command: pos=[0.3290, 0.0478, 0.1904], quat=[0.009, 0.584, 0.028, 0.811]
  0%|          | 240/1000000 [00:45<42:48:14,  6.49it/s][INFO] [1770560076.542740967] [a1x_serl_node]: Published EEF command: pos=[0.3313, 0.0432, 0.1922], quat=[0.014, 0.581, 0.019, 0.813]
  0%|          | 241/1000000 [00:45<42:52:47,  6.48it/s][INFO] [1770560076.697956543] [a1x_serl_node]: Published EEF command: pos=[0.3288, 0.0442, 0.1923], quat=[0.013, 0.584, 0.025, 0.811]
  0%|          | 242/1000000 [00:45<42:57:35,  6.46it/s][INFO] [1770560076.851243582] [a1x_serl_node]: Published EEF command: pos=[0.3322, 0.0456, 0.1915], quat=[0.015, 0.580, 0.018, 0.814]
  0%|          | 243/1000000 [00:46<42:53:53,  6.47it/s][INFO] [1770560077.006367268] [a1x_serl_node]: Published EEF command: pos=[0.3311, 0.0441, 0.1909], quat=[0.012, 0.583, 0.022, 0.812]
  0%|          | 244/1000000 [00:46<42:53:56,  6.47it/s][INFO] [1770560077.158304303] [a1x_serl_node]: Published EEF command: pos=[0.3319, 0.0425, 0.1911], quat=[0.017, 0.580, 0.017, 0.814]
  0%|          | 245/1000000 [00:46<42:46:44,  6.49it/s][INFO] [1770560077.313990400] [a1x_serl_node]: Published EEF command: pos=[0.3318, 0.0454, 0.1924], quat=[0.012, 0.583, 0.020, 0.812]
  0%|          | 246/1000000 [00:46<42:49:52,  6.48it/s][INFO] [1770560077.468347553] [a1x_serl_node]: Published EEF command: pos=[0.3293, 0.0393, 0.1921], quat=[0.018, 0.581, 0.013, 0.813]
  0%|          | 247/1000000 [00:46<42:51:48,  6.48it/s][INFO] [1770560077.621265752] [a1x_serl_node]: Published EEF command: pos=[0.3308, 0.0419, 0.1901], quat=[0.012, 0.582, 0.021, 0.813]
  0%|          | 248/1000000 [00:46<42:42:59,  6.50it/s][INFO] [1770560077.773099697] [a1x_serl_node]: Published EEF command: pos=[0.3303, 0.0391, 0.1914], quat=[0.017, 0.584, 0.013, 0.812]
  0%|          | 249/1000000 [00:46<42:32:41,  6.53it/s][INFO] [1770560077.929377076] [a1x_serl_node]: Published EEF command: pos=[0.3338, 0.0425, 0.1913], quat=[0.013, 0.580, 0.016, 0.814]
  0%|          | 250/1000000 [00:47<42:46:24,  6.49it/s][INFO] [1770560078.082873077] [a1x_serl_node]: Published EEF command: pos=[0.3288, 0.0388, 0.1911], quat=[0.014, 0.581, 0.011, 0.814]
  0%|          | 251/1000000 [00:47<42:48:03,  6.49it/s][INFO] [1770560078.237642232] [a1x_serl_node]: Published EEF command: pos=[0.3336, 0.0424, 0.1922], quat=[0.013, 0.580, 0.013, 0.814]
  0%|          | 252/1000000 [00:47<42:47:43,  6.49it/s][INFO] [1770560078.392092726] [a1x_serl_node]: Published EEF command: pos=[0.3303, 0.0365, 0.1894], quat=[0.014, 0.581, 0.012, 0.813]
  0%|          | 253/1000000 [00:47<42:50:50,  6.48it/s][INFO] [1770560078.545411396] [a1x_serl_node]: Published EEF command: pos=[0.3297, 0.0408, 0.1904], quat=[0.013, 0.581, 0.010, 0.813]
  0%|          | 254/1000000 [00:47<42:45:04,  6.50it/s][INFO] [1770560078.699242447] [a1x_serl_node]: Published EEF command: pos=[0.3330, 0.0388, 0.1913], quat=[0.012, 0.581, 0.010, 0.814]
  0%|          | 255/1000000 [00:47<42:43:40,  6.50it/s][INFO] [1770560078.861132477] [a1x_serl_node]: Published EEF command: pos=[0.3335, 0.0364, 0.1904], quat=[0.012, 0.579, 0.008, 0.815]
  0%|          | 256/1000000 [00:48<43:25:41,  6.39it/s][INFO] [1770560079.014982938] [a1x_serl_node]: Published EEF command: pos=[0.3310, 0.0388, 0.1900], quat=[0.013, 0.581, 0.008, 0.814]
  0%|          | 257/1000000 [00:48<43:12:29,  6.43it/s][INFO] [1770560079.167273075] [a1x_serl_node]: Published EEF command: pos=[0.3308, 0.0362, 0.1901], quat=[0.013, 0.581, 0.005, 0.814]
  0%|          | 258/1000000 [00:48<42:56:42,  6.47it/s][INFO] [1770560079.321175727] [a1x_serl_node]: Published EEF command: pos=[0.3312, 0.0359, 0.1912], quat=[0.013, 0.578, 0.006, 0.816]
  0%|          | 259/1000000 [00:48<42:56:02,  6.47it/s][INFO] [1770560079.476319414] [a1x_serl_node]: Published EEF command: pos=[0.3325, 0.0398, 0.1917], quat=[0.013, 0.578, 0.005, 0.816]
  0%|          | 260/1000000 [00:48<42:55:16,  6.47it/s]EEF delta: pos=[ 1.49838801e-03 -4.92328545e-05  3.72557784e-04], rot=[-0.00226515  0.00940379 -0.00793846], gripper: 0.338 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.5mm
Step done: False, reward: False, path length: 235, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00047596 -0.00035149 -0.00012736], rot=[-0.00043182  0.00262414 -0.00623473], gripper: 0.338 -> 0.336 (33.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.6mm
Step done: False, reward: False, path length: 236, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00296288 -0.00188541  0.00062195], rot=[ 0.00217707  0.0086346  -0.0088149 ], gripper: 0.338 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.6mm
Step done: False, reward: False, path length: 237, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00047695 -0.00190867  0.00258102], rot=[ 0.00654326  0.00771123 -0.00443231], gripper: 0.338 -> 0.339 (33.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.2mm
Step done: False, reward: False, path length: 238, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00198924 -0.00031174 -0.00059358], rot=[ 0.0024045   0.00852756 -0.00840513], gripper: 0.338 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.1mm
Step done: False, reward: False, path length: 239, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00120077  0.00295316  0.00127849], rot=[ 0.00150932  0.01065775 -0.00492203], gripper: 0.338 -> 0.335 (33.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.4mm
Step done: False, reward: False, path length: 240, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00113161 -0.00067277  0.00385224], rot=[-0.00039419  0.00264526 -0.00839214], gripper: 0.338 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 241, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00026956 -0.00048665  0.00438107], rot=[ 0.00240841  0.00558102 -0.00608105], gripper: 0.338 -> 0.333 (33.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.4mm
Step done: False, reward: False, path length: 242, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00203532 0.00228233 0.00218006], rot=[ 0.0007946   0.00033979 -0.00617354], gripper: 0.337 -> 0.335 (33.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 243, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00202381 0.00039052 0.00123972], rot=[-0.00489838  0.00212989 -0.00322148], gripper: 0.337 -> 0.332 (33.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.4mm
Step done: False, reward: False, path length: 244, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00097222 -0.00126069  0.00070047], rot=[ 0.0022359   0.00441664 -0.00509417], gripper: 0.337 -> 0.335 (33.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.7mm
Step done: False, reward: False, path length: 245, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00147652 0.00141576 0.00211129], rot=[-0.00297551  0.00587056 -0.00360357], gripper: 0.337 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.9mm
Step done: False, reward: False, path length: 246, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00227445 -0.003388    0.00153502], rot=[-0.00206017  0.01002723 -0.00695694], gripper: 0.337 -> 0.332 (33.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.4mm
Step done: False, reward: False, path length: 247, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 3.19890212e-04 -1.33805932e-03 -2.61432579e-05], rot=[-0.00044121  0.00535769  0.00203256], gripper: 0.337 -> 0.333 (33.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.4mm
Step done: False, reward: False, path length: 248, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00022225 -0.00141247  0.00134274], rot=[-0.00057824  0.00917468 -0.00023578], gripper: 0.337 -> 0.333 (33.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.0mm
Step done: False, reward: False, path length: 249, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00369424 0.00063835 0.00145108], rot=[-0.00187979 -0.00031844 -0.00302736], gripper: 0.337 -> 0.336 (33.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.0mm
Step done: False, reward: False, path length: 250, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00117816 -0.00151732  0.00148448], rot=[-0.00629363  0.00081577 -0.00202323], gripper: 0.337 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.4mm
Step done: False, reward: False, path length: 251, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00270434 0.00181985 0.00192195], rot=[-0.0050032   0.00386696 -0.00151313], gripper: 0.337 -> 0.336 (33.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.8mm
Step done: False, reward: False, path length: 252, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00018184 -0.00267759 -0.00061053], rot=[-0.0011965   0.00348894  0.00109688], gripper: 0.337 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.8mm
Step done: False, reward: False, path length: 253, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00200872  0.00020342 -0.00043495], rot=[-0.00230875  0.01036011 -0.00509828], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.1mm
Step done: False, reward: False, path length: 254, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00156791 -0.00104963  0.00093368], rot=[-0.00568217  0.00778608 -0.00241978], gripper: 0.337 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.1mm
Step done: False, reward: False, path length: 255, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00252123 -0.00335543  0.00054753], rot=[-0.00556923  0.00065518 -0.00424082], gripper: 0.337 -> 0.336 (33.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.2mm
Step done: False, reward: False, path length: 256, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00014119 -0.00026996  0.00023374], rot=[-0.00226198  0.00543134 -0.0059604 ], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.4mm
Step done: False, reward: False, path length: 257, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-1.62403192e-03 -1.40934193e-03 -4.44857869e-05], rot=[-0.00345017  0.00987815 -0.00597942], gripper: 0.337 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.2mm
Step done: False, reward: False, path length: 258, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00063287 -0.00169278  0.00134062], rot=[-0.00419916 -0.00027632 -0.0038902 ], gripper: 0.337 -> 0.341 (34.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.3mm
Step done: False, reward: False, path length: 259, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00088758 0.00331962 0.00239393], rot=[-0.00181908 -0.00259624 -0.00394635], gripper: 0.337 -> 0.345 (34.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.2mm
Step done: False, reward: False, path length: 260, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560079.631393080] [a1x_serl_node]: Published EEF command: pos=[0.3314, 0.0399, 0.1910], quat=[0.012, 0.583, 0.003, 0.813]
  0%|          | 261/1000000 [00:48<42:57:20,  6.46it/s][INFO] [1770560079.786170266] [a1x_serl_node]: Published EEF command: pos=[0.3314, 0.0400, 0.1905], quat=[0.010, 0.579, 0.004, 0.815]
  0%|          | 262/1000000 [00:48<42:57:53,  6.46it/s][INFO] [1770560079.939457016] [a1x_serl_node]: Published EEF command: pos=[0.3321, 0.0352, 0.1900], quat=[0.014, 0.581, 0.003, 0.814]
  0%|          | 263/1000000 [00:49<42:51:06,  6.48it/s][INFO] [1770560080.094359752] [a1x_serl_node]: Published EEF command: pos=[0.3325, 0.0338, 0.1928], quat=[0.009, 0.575, 0.002, 0.818]
  0%|          | 264/1000000 [00:49<42:53:02,  6.48it/s][INFO] [1770560080.246760150] [a1x_serl_node]: Published EEF command: pos=[0.3308, 0.0370, 0.1898], quat=[0.011, 0.581, 0.001, 0.814]
  0%|          | 265/1000000 [00:49<42:43:07,  6.50it/s][INFO] [1770560080.398204684] [a1x_serl_node]: Published EEF command: pos=[0.3329, 0.0361, 0.1900], quat=[0.008, 0.578, 0.000, 0.816]
  0%|          | 266/1000000 [00:49<42:30:52,  6.53it/s][INFO] [1770560080.552632498] [a1x_serl_node]: Published EEF command: pos=[0.3316, 0.0413, 0.1913], quat=[0.010, 0.579, 0.001, 0.815]
  0%|          | 267/1000000 [00:49<42:37:52,  6.51it/s][INFO] [1770560080.707653095] [a1x_serl_node]: Published EEF command: pos=[0.3342, 0.0396, 0.1924], quat=[0.005, 0.580, 0.001, 0.815]
  0%|          | 268/1000000 [00:49<42:45:04,  6.50it/s][INFO] [1770560080.862139999] [a1x_serl_node]: Published EEF command: pos=[0.3308, 0.0445, 0.1912], quat=[0.008, 0.583, 0.002, 0.812]
  0%|          | 269/1000000 [00:50<42:47:29,  6.49it/s][INFO] [1770560081.013951465] [a1x_serl_node]: Published EEF command: pos=[0.3304, 0.0442, 0.1908], quat=[0.002, 0.579, 0.000, 0.816]
  0%|          | 270/1000000 [00:50<42:42:26,  6.50it/s][INFO] [1770560081.168811741] [a1x_serl_node]: Published EEF command: pos=[0.3294, 0.0445, 0.1893], quat=[0.005, 0.582, 0.002, 0.813]
  0%|          | 271/1000000 [00:50<42:43:42,  6.50it/s][INFO] [1770560081.327857932] [a1x_serl_node]: Published EEF command: pos=[0.3317, 0.0449, 0.1910], quat=[0.002, 0.580, 0.001, 0.815]
  0%|          | 272/1000000 [00:50<43:12:38,  6.43it/s][INFO] [1770560081.488632688] [a1x_serl_node]: Published EEF command: pos=[0.3311, 0.0431, 0.1941], quat=[0.003, 0.582, 0.006, 0.813]
  0%|          | 273/1000000 [00:50<43:36:42,  6.37it/s][INFO] [1770560081.646686455] [a1x_serl_node]: Published EEF command: pos=[0.3330, 0.0437, 0.1904], quat=[0.004, 0.580, -0.001, 0.815]
  0%|          | 274/1000000 [00:50<43:48:50,  6.34it/s][INFO] [1770560081.808142704] [a1x_serl_node]: Published EEF command: pos=[0.3316, 0.0450, 0.1917], quat=[0.002, 0.583, 0.004, 0.813]
  0%|          | 275/1000000 [00:51<43:57:33,  6.32it/s][INFO] [1770560081.963346712] [a1x_serl_node]: Published EEF command: pos=[0.3310, 0.0374, 0.1804], quat=[0.007, 0.582, -0.004, 0.813]
  0%|          | 276/1000000 [00:51<58:13:01,  4.77it/s][INFO] [1770560082.293250953] [a1x_serl_node]: Published EEF command: pos=[0.3314, 0.0421, 0.1862], quat=[0.010, 0.583, -0.008, 0.812]
  0%|          | 277/1000000 [00:51<53:43:19,  5.17it/s][INFO] [1770560082.451030530] [a1x_serl_node]: Published EEF command: pos=[0.3322, 0.0403, 0.1871], quat=[0.007, 0.581, -0.007, 0.814]
  0%|          | 278/1000000 [00:51<50:51:16,  5.46it/s][INFO] [1770560082.609105418] [a1x_serl_node]: Published EEF command: pos=[0.3379, 0.0414, 0.1873], quat=[0.011, 0.581, -0.009, 0.814]
  0%|          | 279/1000000 [00:51<48:40:52,  5.70it/s][INFO] [1770560082.764828277] [a1x_serl_node]: Published EEF command: pos=[0.3317, 0.0399, 0.1854], quat=[0.005, 0.580, -0.009, 0.815]
  0%|          | 280/1000000 [00:51<47:04:29,  5.90it/s][INFO] [1770560082.921240949] [a1x_serl_node]: Published EEF command: pos=[0.3335, 0.0407, 0.1887], quat=[0.008, 0.578, -0.009, 0.816]
  0%|          | 281/1000000 [00:52<45:56:57,  6.04it/s][INFO] [1770560083.073235446] [a1x_serl_node]: Published EEF command: pos=[0.3353, 0.0425, 0.1860], quat=[0.003, 0.580, -0.008, 0.814]
  0%|          | 282/1000000 [00:52<44:53:55,  6.19it/s][INFO] [1770560083.228674025] [a1x_serl_node]: Published EEF command: pos=[0.3344, 0.0413, 0.1886], quat=[0.005, 0.579, -0.010, 0.815]
  0%|          | 283/1000000 [00:52<44:19:16,  6.27it/s][INFO] [1770560083.385908340] [a1x_serl_node]: Published EEF command: pos=[0.3341, 0.0442, 0.1871], quat=[0.003, 0.580, -0.009, 0.815]
  0%|          | 284/1000000 [00:52<44:06:14,  6.30it/s]Current EE Pos: [0.31707004 0.03844284 0.18589755], Rot (quat): [-0.00345181  0.59721838  0.02824978  0.8015736 ]
Current EE Pos: [0.31808701 0.04007494 0.18710122], Rot (quat): [-0.00232351  0.59473986  0.02863584  0.80340468]
Current EE Pos: [0.31882823 0.041448   0.18680447], Rot (quat): [-0.00275798  0.59393996  0.03076134  0.80391633]
Current EE Pos: [0.31792374 0.04300139 0.1861397 ], Rot (quat): [-0.00128973  0.59563078  0.03155668  0.80263721]
Current EE Pos: [0.31987828 0.04540611 0.18719752], Rot (quat): [-0.00384961  0.59179345  0.03193646  0.80544755]
Current EE Pos: [0.31980235 0.04550735 0.18711671], Rot (quat): [-0.00411201  0.59155316  0.03282865  0.80558689]
Current EE Pos: [0.32061393 0.04665788 0.187412  ], Rot (quat): [-0.00458231  0.59006325  0.03509081  0.80658106]
Current EE Pos: [0.32080255 0.04605793 0.18679953], Rot (quat): [-0.00457381  0.59047268  0.03465375  0.80630032]
Current EE Pos: [0.32080038 0.04667696 0.18676023], Rot (quat): [-0.00419382  0.59029097  0.03583672  0.80638372]
Current EE Pos: [0.32142635 0.04663815 0.18669264], Rot (quat): [-0.00202328  0.59000831  0.0357845   0.80660125]
Current EE Pos: [0.32121255 0.04698702 0.18660852], Rot (quat): [-0.00415498  0.59006471  0.03658633  0.80651585]
Current EE Pos: [0.32148333 0.04618006 0.18615966], Rot (quat): [-0.00128874  0.59028505  0.03505539  0.80643228]
Current EE Pos: [0.32319768 0.04593848 0.18614119], Rot (quat): [-0.00171606  0.58789209  0.0370254   0.80808976]
Current EE Pos: [0.32281725 0.04688483 0.18732108], Rot (quat): [0.00343993 0.58850225 0.03165362 0.80786838]
Current EE Pos: [0.32619796 0.0462898  0.18807356], Rot (quat): [-0.00338857  0.58156573  0.03813398  0.81259807]
Current EE Pos: [0.32396762 0.04657112 0.1871948 ], Rot (quat): [0.00491797 0.58720006 0.0313748  0.8088186 ]
Current EE Pos: [0.32661393 0.04660408 0.18742375], Rot (quat): [1.44381333e-04 5.81988159e-01 3.81814887e-02 8.12300398e-01]
Current EE Pos: [0.32638942 0.04690129 0.1872192 ], Rot (quat): [0.00671209 0.58462961 0.03125436 0.8106703 ]
Current EE Pos: [0.32846118 0.04808682 0.18795865], Rot (quat): [7.82333162e-04 5.80179079e-01 3.92321023e-02 8.13543156e-01]
Current EE Pos: [0.32729582 0.04669453 0.1870181 ], Rot (quat): [0.00699718 0.5837685  0.0311256  0.81129315]
Current EE Pos: [0.32895037 0.04707924 0.18776114], Rot (quat): [0.00144566 0.58011627 0.03660355 0.81370953]
Current EE Pos: [0.32877248 0.04685215 0.18781759], Rot (quat): [0.0065613  0.58080705 0.02981772 0.81346852]
Current EE Pos: [0.33013732 0.0464599  0.18858089], Rot (quat): [0.001229   0.57783022 0.03443528 0.8154293 ]
Current EE Pos: [0.3293668  0.04621091 0.18884237], Rot (quat): [0.00807183 0.58034234 0.02722307 0.81387746]
Current EE Pos: [0.33076704 0.04734559 0.18912045], Rot (quat): [0.00146421 0.57785358 0.03218211 0.81550439]
Current EE Pos: [0.33009525 0.04610973 0.18865388], Rot (quat): [0.00791397 0.57984401 0.02624697 0.81426616]
Current EE Pos: [0.33088819 0.04612925 0.18886283], Rot (quat): [0.0029234  0.57774193 0.0296086  0.81567705]
Current EE Pos: [0.33045865 0.04489283 0.18842892], Rot (quat): [0.00953622 0.57967714 0.0244838  0.8144225 ]
Current EE Pos: [0.33019975 0.04484938 0.18908388], Rot (quat): [0.00657844 0.57935327 0.02920132 0.81452673]
Current EE Pos: [0.33019606 0.04390145 0.18834594], Rot (quat): [0.01125733 0.58013278 0.02284235 0.81412373]
Current EE Pos: [0.32910026 0.04465106 0.18795372], Rot (quat): [0.01006263 0.58217222 0.02668581 0.81256515]
Current EE Pos: [0.330147   0.04329199 0.18929582], Rot (quat): [0.01261788 0.58016147 0.02021786 0.81415275]
Current EE Pos: [0.3290774  0.04371773 0.18969598], Rot (quat): [0.01305748 0.58180956 0.0244175  0.81285357]
Current EE Pos: [0.33095887 0.04374201 0.19037231], Rot (quat): [0.01473539 0.57827674 0.01845608 0.81549878]
Current EE Pos: [0.3302897  0.04398962 0.19024257], Rot (quat): [0.01172139 0.58021669 0.02219235 0.81407536]
Current EE Pos: [0.33162331 0.04272574 0.19058084], Rot (quat): [0.01654054 0.57741872 0.01690506 0.81610554]
Current EE Pos: [0.33046051 0.04326553 0.1901641 ], Rot (quat): [0.01246183 0.57987612 0.01998707 0.81436411]
Current EE Pos: [0.33006746 0.04054314 0.19006012], Rot (quat): [0.01696196 0.58021912 0.01382628 0.81416638]
Current EE Pos: [0.33005821 0.04188829 0.18981607], Rot (quat): [0.01332521 0.58032932 0.01790524 0.81407599]
Current EE Pos: [0.32997732 0.04034497 0.18958408], Rot (quat): [0.01629405 0.58079003 0.01384854 0.81377249]
Current EE Pos: [0.3308527  0.04060779 0.19025464], Rot (quat): [0.01410166 0.57853432 0.01484312 0.81540105]
Current EE Pos: [0.33013303 0.03918685 0.18996379], Rot (quat): [0.0144938  0.57993977 0.01174384 0.81444574]
Current EE Pos: [0.33172834 0.04060927 0.19081227], Rot (quat): [0.0128256  0.57720954 0.01310094 0.81639023]
Current EE Pos: [0.33148177 0.03988062 0.19037182], Rot (quat): [0.01321858 0.57808743 0.01270991 0.81576875]
Current EE Pos: [0.33094647 0.03980212 0.18980472], Rot (quat): [0.01334842 0.57898678 0.01162822 0.81514472]
Current EE Pos: [0.33111427 0.03911693 0.18976054], Rot (quat): [0.01253047 0.5788212  0.01110314 0.8152826 ]
Current EE Pos: [0.33242477 0.03760004 0.19019025], Rot (quat): [0.01240288 0.57710788 0.00882944 0.816526  ]
Current EE Pos: [0.33184217 0.03762154 0.18982731], Rot (quat): [0.01351039 0.57801773 0.00841439 0.81586897]
Current EE Pos: [0.33164132 0.03652206 0.18932702], Rot (quat): [0.01279167 0.57865625 0.00663123 0.81544426]
Current EE Pos: [0.33183298 0.03605727 0.18989768], Rot (quat): [0.01319716 0.57762385 0.00516804 0.81618001]
Current EE Pos: [0.33235493 0.0375898  0.1906252 ], Rot (quat): [0.01353271 0.57635996 0.00476403 0.81706999]
Current EE Pos: [0.3312781  0.03790465 0.19026286], Rot (quat): [0.01267835 0.57870385 0.00415784 0.81542862]
Current EE Pos: [0.33192242 0.03822848 0.1901128 ], Rot (quat): [0.01111625 0.57710255 0.00511499 0.81658001]
Current EE Pos: [0.33140608 0.0372099  0.18989066], Rot (quat): [0.01288574 0.57857532 0.00325018 0.81552069]
Current EE Pos: [0.33256417 0.03616635 0.19071338], Rot (quat): [0.01031364 0.5753904  0.00209237 0.81781119]
Current EE Pos: [0.33170473 0.03631633 0.19011735], Rot (quat): [0.01196035 0.57771754 0.00161268 0.81614753]
Current EE Pos: [0.3321449  0.03613157 0.18988336], Rot (quat): [0.00891302 0.57660145 0.00182137 0.81697491]
Current EE Pos: [0.33152311 0.0385658  0.19013993], Rot (quat): [0.01046975 0.57773883 0.00169486 0.81615278]
Current EE Pos: [0.33231125 0.03834825 0.19067267], Rot (quat): [0.00581993 0.57659626 0.00182426 0.81700646]
Current EE Pos: [0.33079373 0.04187991 0.19036585], Rot (quat): [0.00841133 0.57902576 0.00332012 0.8152591 ]
Current EE Pos: [0.33113078 0.04137679 0.1903226 ], Rot (quat): [0.00331105 0.57773021 0.0013163  0.81622001]
Current EE Pos: [0.33022843 0.04251334 0.18967928], Rot (quat): [0.00583762 0.57992064 0.00386103 0.81464291]
Current EE Pos: [0.33083301 0.04226699 0.19005207], Rot (quat): [0.00221621 0.57799149 0.00260318 0.81603563]
Current EE Pos: [0.32969571 0.04299263 0.190424  ], Rot (quat): [0.00261782 0.58046776 0.00590057 0.81425764]
Current EE Pos: [0.33193909 0.04239186 0.19054468], Rot (quat): [3.71337224e-03 5.76580977e-01 5.14055656e-04 8.17031409e-01]
Current EE Pos: [0.33268681 0.03917702 0.18635482], Rot (quat): [ 0.00632811  0.57920422 -0.00315475  0.81515181]
Current EE Pos: [0.33253101 0.03879754 0.18499948], Rot (quat): [ 0.00639447  0.58100431 -0.00376914  0.81386663]
Current EE Pos: [0.33203868 0.03920073 0.18497181], Rot (quat): [ 0.00974242  0.58166282 -0.00697211  0.81334177]
Current EE Pos: [0.33231653 0.03925167 0.18555183], Rot (quat): [ 0.00673713  0.58046502 -0.00563749  0.8142378 ]
Current EE Pos: [0.33471954 0.03969129 0.18671982], Rot (quat): [ 0.01100271  0.57715755 -0.00804012  0.81651911]
Current EE Pos: [0.33323151 0.03903529 0.1857463 ], Rot (quat): [ 0.00478605  0.57885599 -0.0074421   0.81538178]
Current EE Pos: [0.33362725 0.0396596  0.18657254], Rot (quat): [ 0.00746334  0.57783828 -0.00844002  0.81607352]
Current EE Pos: [0.33439992 0.04064076 0.18623541], Rot (quat): [ 0.00423383  0.57711452 -0.00772014  0.81661576]
[INFO] [1770560083.542629263] [a1x_serl_node]: Published EEF command: pos=[0.3367, 0.0454, 0.1884], quat=[0.003, 0.573, -0.011, 0.819]
  0%|          | 285/1000000 [00:52<43:58:15,  6.32it/s][INFO] [1770560083.700920712] [a1x_serl_node]: Published EEF command: pos=[0.3385, 0.0419, 0.1895], quat=[0.006, 0.576, -0.009, 0.817]
  0%|          | 286/1000000 [00:52<43:52:39,  6.33it/s]EEF delta: pos=[-0.00047023  0.00387917  0.00105324], rot=[-0.00427468  0.01232856 -0.00231521], gripper: 0.337 -> 0.344 (34.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.0mm
Step done: False, reward: False, path length: 261, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-9.20424820e-04  2.38457555e-03 -8.09436606e-05], rot=[-0.00668903  0.0071917   0.00312399], gripper: 0.337 -> 0.339 (33.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.6mm
Step done: False, reward: False, path length: 262, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00078704 -0.0026831  -0.00021324], rot=[-3.58629040e-06  5.09560481e-03 -2.92700808e-03], gripper: 0.337 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.8mm
Step done: False, reward: False, path length: 263, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00054219 -0.00438535  0.00271545], rot=[-0.00607888 -0.00486079 -0.00269721], gripper: 0.337 -> 0.338 (33.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 264, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-5.59860840e-04 -2.31856597e-04 -7.95009546e-05], rot=[-0.00669757  0.00663335 -0.00145054], gripper: 0.337 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.6mm
Step done: False, reward: False, path length: 265, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 3.41171050e-04 -6.27175905e-05 -7.46137695e-04], rot=[-0.00654394  0.00699928  0.00021279], gripper: 0.337 -> 0.342 (34.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.8mm
Step done: False, reward: False, path length: 266, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-9.07492358e-05  4.93564410e-03  1.14542304e-03], rot=[-0.00458582  0.00434425  0.00172446], gripper: 0.337 -> 0.349 (34.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.1mm
Step done: False, reward: False, path length: 267, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00209841 0.0035181  0.00246771], rot=[-0.00783572  0.00734556  0.00338793], gripper: 0.337 -> 0.344 (34.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.8mm
Step done: False, reward: False, path length: 268, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00068208  0.00593421  0.00104082], rot=[-0.00363006  0.01284974  0.00430604], gripper: 0.337 -> 0.345 (34.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.1mm
Step done: False, reward: False, path length: 269, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00195015  0.00587659  0.00015656], rot=[-0.00775796  0.00474983  0.00127905], gripper: 0.337 -> 0.346 (34.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.2mm
Step done: False, reward: False, path length: 270, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00143865  0.00266557 -0.00104617], rot=[-0.00601056  0.00685391  0.0019901 ], gripper: 0.337 -> 0.342 (34.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.2mm
Step done: False, reward: False, path length: 271, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00061854 0.00353971 0.0006961 ], rot=[-0.00239654  0.00455853  0.00115048], gripper: 0.337 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 272, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00082738 0.00056946 0.00445392], rot=[-0.00181034  0.003933    0.0077143 ], gripper: 0.337 -> 0.339 (33.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.6mm
Step done: False, reward: False, path length: 273, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00220497 0.00144411 0.000328  ], rot=[-0.00113298  0.00378652 -0.00685366], gripper: 0.337 -> 0.342 (34.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 274, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00187776 0.00205283 0.00130711], rot=[-0.00343746  0.00607464 -0.00325677], gripper: 0.337 -> 0.341 (34.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 275, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00094624 -0.00498391 -0.01019449], rot=[ 0.00016435  0.01371677 -0.01043877], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=274ms, 误差=7.8mm
Step done: False, reward: False, path length: 276, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00127384  0.00289196 -0.00015215], rot=[ 0.00071083  0.0092026  -0.01133541], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.2mm
Step done: False, reward: False, path length: 277, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00035547  0.00149912  0.00205297], rot=[-0.00300159 -0.0002234  -0.0046616 ], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 278, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00590681 0.00223743 0.00234701], rot=[-0.00025246 -0.00191884 -0.00518842], gripper: 0.337 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=6.7mm
Step done: False, reward: False, path length: 279, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00062509  0.00060933 -0.00012753], rot=[-0.00681266 -0.00163373 -0.00234999], gripper: 0.337 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.9mm
Step done: False, reward: False, path length: 280, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00119209  0.00100223  0.00197481], rot=[-0.00545494  0.00225707  0.00222094], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 281, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00201993 0.00351316 0.00028072], rot=[-0.00387717  0.00388168  0.00031795], gripper: 0.337 -> 0.341 (34.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 282, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00075691 0.00160031 0.00202325], rot=[-0.00512212  0.00323187  0.00066596], gripper: 0.337 -> 0.340 (34.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.7mm
Step done: False, reward: False, path length: 283, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00027771  0.00359006  0.00087157], rot=[-0.00310256  0.00715224 -0.00113996], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 284, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00266759 0.00482831 0.00160608], rot=[-0.00651369 -0.00970214 -0.00307562], gripper: 0.337 -> 0.342 (34.2mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.7mm
Step done: False, reward: False, path length: 285, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00456589 0.00052151 0.00279884], rot=[ 0.00136137 -0.0037782  -0.00069798], gripper: 0.337 -> 0.344 (34.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.4mm
Step done: False, reward: False, path length: 286, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560083.855988780] [a1x_serl_node]: Published EEF command: pos=[0.3370, 0.0516, 0.1891], quat=[0.002, 0.573, -0.008, 0.819]
  0%|          | 287/1000000 [00:53<55:35:35,  5.00it/s][INFO] [1770560084.152845700] [a1x_serl_node]: Published EEF command: pos=[0.3378, 0.0505, 0.1879], quat=[0.003, 0.572, -0.007, 0.820]
  0%|          | 288/1000000 [00:53<51:45:59,  5.36it/s][INFO] [1770560084.310965458] [a1x_serl_node]: Published EEF command: pos=[0.3395, 0.0445, 0.1882], quat=[0.004, 0.571, -0.008, 0.821]
  0%|          | 289/1000000 [00:53<49:30:41,  5.61it/s][INFO] [1770560084.467963623] [a1x_serl_node]: Published EEF command: pos=[0.3407, 0.0524, 0.1882], quat=[0.002, 0.571, -0.008, 0.821]
  0%|          | 290/1000000 [00:53<47:35:23,  5.84it/s][INFO] [1770560084.620710123] [a1x_serl_node]: Published EEF command: pos=[0.3391, 0.0510, 0.1889], quat=[0.002, 0.573, -0.008, 0.820]
  0%|          | 291/1000000 [00:53<46:05:51,  6.02it/s][INFO] [1770560084.780999389] [a1x_serl_node]: Published EEF command: pos=[0.3380, 0.0510, 0.1897], quat=[-0.002, 0.575, -0.005, 0.818]
  0%|          | 292/1000000 [00:53<45:34:52,  6.09it/s][INFO] [1770560084.934947953] [a1x_serl_node]: Published EEF command: pos=[0.3399, 0.0512, 0.1913], quat=[0.000, 0.576, -0.010, 0.818]
  0%|          | 293/1000000 [00:54<44:45:28,  6.20it/s][INFO] [1770560085.108122873] [a1x_serl_node]: Published EEF command: pos=[0.3367, 0.0479, 0.1878], quat=[-0.004, 0.576, -0.008, 0.818]
  0%|          | 294/1000000 [00:54<45:44:53,  6.07it/s][INFO] [1770560085.261344995] [a1x_serl_node]: Published EEF command: pos=[0.3363, 0.0490, 0.1867], quat=[0.000, 0.577, -0.011, 0.817]
  0%|          | 295/1000000 [00:54<44:45:23,  6.20it/s][INFO] [1770560085.413727104] [a1x_serl_node]: Published EEF command: pos=[0.3405, 0.0476, 0.1867], quat=[-0.002, 0.574, -0.012, 0.819]
  0%|          | 296/1000000 [00:54<44:02:01,  6.31it/s][INFO] [1770560085.569401984] [a1x_serl_node]: Published EEF command: pos=[0.3388, 0.0455, 0.1864], quat=[-0.001, 0.580, -0.010, 0.814]
  0%|          | 297/1000000 [00:54<43:50:16,  6.33it/s][INFO] [1770560085.726071548] [a1x_serl_node]: Published EEF command: pos=[0.3411, 0.0455, 0.1859], quat=[-0.002, 0.569, -0.011, 0.822]
  0%|          | 298/1000000 [00:54<43:42:08,  6.35it/s][INFO] [1770560085.881639318] [a1x_serl_node]: Published EEF command: pos=[0.3395, 0.0462, 0.1848], quat=[0.001, 0.578, -0.012, 0.816]
  0%|          | 299/1000000 [00:55<43:33:14,  6.38it/s][INFO] [1770560086.035015131] [a1x_serl_node]: Published EEF command: pos=[0.3418, 0.0378, 0.1803], quat=[0.000, 0.570, -0.011, 0.822]
  0%|          | 300/1000000 [00:55<60:14:00,  4.61it/s][INFO] [1770560086.395504950] [a1x_serl_node]: Published EEF command: pos=[0.3436, 0.0445, 0.1847], quat=[0.001, 0.572, -0.012, 0.820]
  0%|          | 301/1000000 [00:55<55:12:54,  5.03it/s][INFO] [1770560086.552782596] [a1x_serl_node]: Published EEF command: pos=[0.3449, 0.0445, 0.1847], quat=[0.000, 0.570, -0.012, 0.822]
  0%|          | 302/1000000 [00:55<51:44:58,  5.37it/s][INFO] [1770560086.704423783] [a1x_serl_node]: Published EEF command: pos=[0.3437, 0.0421, 0.1837], quat=[-0.001, 0.572, -0.011, 0.820]
  0%|          | 303/1000000 [00:55<48:49:33,  5.69it/s][INFO] [1770560086.860961927] [a1x_serl_node]: Published EEF command: pos=[0.3466, 0.0438, 0.1843], quat=[0.002, 0.571, -0.014, 0.821]
  0%|          | 304/1000000 [00:56<47:10:15,  5.89it/s][INFO] [1770560087.016293957] [a1x_serl_node]: Published EEF command: pos=[0.3461, 0.0453, 0.1856], quat=[-0.000, 0.569, -0.012, 0.822]
  0%|          | 305/1000000 [00:56<46:03:03,  6.03it/s][INFO] [1770560087.183775978] [a1x_serl_node]: Published EEF command: pos=[0.3441, 0.0413, 0.1845], quat=[0.004, 0.572, -0.016, 0.820]
  0%|          | 306/1000000 [00:56<46:12:27,  6.01it/s][INFO] [1770560087.338334525] [a1x_serl_node]: Published EEF command: pos=[0.3481, 0.0443, 0.1840], quat=[0.002, 0.568, -0.015, 0.823]
  0%|          | 307/1000000 [00:56<45:09:13,  6.15it/s][INFO] [1770560087.490970866] [a1x_serl_node]: Published EEF command: pos=[0.3426, 0.0411, 0.1809], quat=[0.004, 0.574, -0.019, 0.819]
  0%|          | 308/1000000 [00:56<44:23:36,  6.26it/s][INFO] [1770560087.645424263] [a1x_serl_node]: Published EEF command: pos=[0.3484, 0.0460, 0.1811], quat=[0.007, 0.568, -0.022, 0.823]
  0%|          | 309/1000000 [00:56<44:02:18,  6.31it/s][INFO] [1770560087.800921433] [a1x_serl_node]: Published EEF command: pos=[0.3491, 0.0430, 0.1904], quat=[0.003, 0.576, -0.017, 0.817]
  0%|          | 310/1000000 [00:57<79:05:31,  3.51it/s][INFO] [1770560088.384299634] [a1x_serl_node]: Published EEF command: pos=[0.3406, 0.0401, 0.1865], quat=[0.001, 0.580, -0.016, 0.814]
  0%|          | 311/1000000 [00:57<68:31:43,  4.05it/s][INFO] [1770560088.542369094] [a1x_serl_node]: Published EEF command: pos=[0.3472, 0.0442, 0.1869], quat=[0.003, 0.568, -0.015, 0.823]
  0%|          | 312/1000000 [00:57<61:16:17,  4.53it/s]EEF delta: pos=[0.0009392  0.00870604 0.00082779], rot=[-0.00176419  0.00198048  0.00510529], gripper: 0.337 -> 0.348 (34.8mm)
⏱️  ✓ 执行耗时=243ms, 误差=7.6mm
Step done: False, reward: False, path length: 287, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00132822  0.00409081 -0.00085457], rot=[-1.52420253e-05  2.38249358e-03  2.71367398e-03], gripper: 0.337 -> 0.350 (35.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.4mm
Step done: False, reward: False, path length: 288, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00288865 -0.00330128 -0.00066176], rot=[ 0.00111809  0.00225054 -0.00293186], gripper: 0.337 -> 0.343 (34.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.4mm
Step done: False, reward: False, path length: 289, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00325621  0.00320081 -0.00029977], rot=[-0.00288549  0.00251646 -0.00203926], gripper: 0.337 -> 0.336 (33.6mm)
⏱️  ✓ 执行耗时=99ms, 误差=4.6mm
Step done: False, reward: False, path length: 290, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00088464 0.00385349 0.00066243], rot=[-0.0027368   0.00700129  0.00040758], gripper: 0.337 -> 0.337 (33.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 291, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00139475  0.00097592  0.00157752], rot=[-0.00520486  0.01755281  0.00954808], gripper: 0.337 -> 0.338 (33.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.3mm
Step done: False, reward: False, path length: 292, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00116423 0.00168638 0.00355525], rot=[-0.00585804  0.01347895 -0.00454132], gripper: 0.337 -> 0.336 (33.6mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 293, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00084245 -0.00292187  0.00077947], rot=[-0.00845538  0.00842283 -0.00239559], gripper: 0.337 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.1mm
Step done: False, reward: False, path length: 294, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00143396 -0.00117589 -0.00148956], rot=[-0.0027297   0.01023833 -0.00489963], gripper: 0.337 -> 0.335 (33.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.4mm
Step done: False, reward: False, path length: 295, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00233708 -0.00193456 -0.00106001], rot=[-0.00425392  0.00618987 -0.00783139], gripper: 0.337 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.2mm
Step done: False, reward: False, path length: 296, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00193707 -0.00353006 -0.00015109], rot=[-0.00209954  0.01320305  0.00109959], gripper: 0.337 -> 0.334 (33.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.0mm
Step done: False, reward: False, path length: 297, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00261218 -0.00264906 -0.0012107 ], rot=[-0.00213959 -0.00637845 -0.00158535], gripper: 0.337 -> 0.338 (33.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.9mm
Step done: False, reward: False, path length: 298, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00200298 -0.00049752 -0.001393  ], rot=[ 0.00042887  0.0048688  -0.00504391], gripper: 0.337 -> 0.339 (33.9mm)
⏱️  ✓ 执行耗时=100ms, 误差=2.5mm
Step done: False, reward: False, path length: 299, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00095357 -0.00873257 -0.00671094], rot=[ 0.00185289  0.00138116 -0.00217956], gripper: 0.337 -> 0.346 (34.6mm)
⏱️  ✓ 执行耗时=304ms, 误差=7.7mm
Step done: False, reward: False, path length: 300, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 9.56499018e-04  2.78544193e-03 -7.11911125e-05], rot=[ 0.00247104  0.00819133 -0.00216829], gripper: 0.337 -> 0.350 (35.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 301, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00192332 0.00305995 0.00058268], rot=[ 0.00029511  0.00163418 -0.00094657], gripper: 0.337 -> 0.347 (34.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.7mm
Step done: False, reward: False, path length: 302, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 8.82706547e-04  7.19839300e-05 -1.36839808e-04], rot=[-0.00289058  0.00689533  0.0019831 ], gripper: 0.337 -> 0.348 (34.8mm)
⏱️  ✓ 执行耗时=100ms, 误差=0.9mm
Step done: False, reward: False, path length: 303, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00284114  0.00119676 -0.00015052], rot=[-0.00131808  0.00753187 -0.00681532], gripper: 0.337 -> 0.360 (36.0mm)
⏱️  ✓ 执行耗时=99ms, 误差=3.1mm
Step done: False, reward: False, path length: 304, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00275027 0.00257045 0.00152238], rot=[-0.00301275  0.00089857 -0.00363354], gripper: 0.337 -> 0.363 (36.3mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.1mm
Step done: False, reward: False, path length: 305, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00014604 -0.00062394  0.00077749], rot=[ 0.00030502  0.00957938 -0.0066257 ], gripper: 0.337 -> 0.361 (36.1mm)
⏱️  ✓ 执行耗时=100ms, 误差=1.0mm
Step done: False, reward: False, path length: 306, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00328116  0.00138553 -0.00042586], rot=[-0.00255071  0.00390866 -0.00586465], gripper: 0.337 -> 0.360 (36.0mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.6mm
Step done: False, reward: False, path length: 307, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-1.43122114e-03  1.07816595e-05 -2.59034429e-03], rot=[-0.00538425  0.01347304 -0.00779076], gripper: 0.337 -> 0.365 (36.5mm)
⏱️  ✓ 执行耗时=100ms, 误差=3.0mm
Step done: False, reward: False, path length: 308, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[ 0.00230421  0.00355462 -0.00296815], rot=[-0.00241718  0.00498719 -0.01631425], gripper: 0.338 -> 0.377 (37.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.2mm
Step done: False, reward: False, path length: 309, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.00434285 0.00138014 0.00802214], rot=[-0.00117282  0.0161752  -0.00319102], gripper: 0.339 -> 0.367 (36.7mm)
⏱️  ✓ 执行耗时=525ms, 误差=8.0mm
Step done: False, reward: False, path length: 310, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[-0.00402704 -0.0016491   0.00253588], rot=[-0.00367011  0.02073176  0.00155332], gripper: 0.339 -> 0.367 (36.7mm)
⏱️  ✓ 执行耗时=100ms, 误差=5.0mm
Step done: False, reward: False, path length: 311, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
EEF delta: pos=[0.0029982  0.00246023 0.00277687], rot=[ 0.00021444 -0.01008673  0.00018509], gripper: 0.339 -> 0.364 (36.4mm)
⏱️  ✓ 执行耗时=100ms, 误差=4.8mm
Step done: False, reward: False, path length: 312, terminate: False
Action scale: [-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.]
[INFO] [1770560088.703483114] [a1x_serl_node]: Published EEF command: pos=[0.3420, 0.0394, 0.1877], quat=[0.002, 0.581, -0.016, 0.814]
  0%|          | 313/1000000 [00:57<56:18:43,  4.93it/s][INFO] [1770560088.865666307] [a1x_serl_node]: Published EEF command: pos=[0.3486, 0.0424, 0.1842], quat=[0.001, 0.568, -0.016, 0.823]
  0%|          | 314/1000000 [00:58<52:43:50,  5.27it/s][INFO] [1770560089.017990877] [a1x_serl_node]: Published EEF command: pos=[0.3419, 0.0438, 0.1836], quat=[0.002, 0.582, -0.017, 0.813]
  0%|          | 315/1000000 [00:58<49:39:36,  5.59it/s][INFO] [1770560089.172051044] [a1x_serl_node]: Published EEF command: pos=[0.3484, 0.0408, 0.1860], quat=[0.000, 0.569, -0.016, 0.822]
  0%|          | 316/1000000 [00:58<47:36:10,  5.83it/s][INFO] [1770560089.326124760] [a1x_serl_node]: Published EEF command: pos=[0.3414, 0.0393, 0.1822], quat=[0.000, 0.581, -0.021, 0.813]
  0%|          | 317/1000000 [00:58<46:10:20,  6.01it/s][INFO] [1770560089.480457227] [a1x_serl_node]: Published EEF command: pos=[0.3467, 0.0461, 0.1837], quat=[-0.001, 0.568, -0.016, 0.823]
  0%|          | 318/1000000 [00:58<45:14:55,  6.14it/s][INFO] [1770560089.637564594] [a1x_serl_node]: Published EEF command: pos=[0.3422, 0.0448, 0.1845], quat=[-0.001, 0.578, -0.022, 0.816]
  0%|          | 319/1000000 [00:58<44:40:34,  6.22it/s][INFO] [1770560089.791768270] [a1x_serl_node]: Published EEF command: pos=[0.3443, 0.0470, 0.1863], quat=[-0.006, 0.571, -0.015, 0.821]
  0%|          | 320/1000000 [00:58<44:06:48,  6.29it/s][INFO] [1770560089.946111758] [a1x_serl_node]: Published EEF command: pos=[0.3419, 0.0467, 0.1844], quat=[-0.003, 0.577, -0.022, 0.816]
  0%|          | 321/1000000 [00:59<43:46:29,  6.34it/s][INFO] [1770560090.101299388] [a1x_serl_node]: Published EEF command: pos=[0.3496, 0.0466, 0.1842], quat=[-0.002, 0.569, -0.018, 0.822]
  0%|          | 322/1000000 [00:59<43:32:14,  6.38it/s][INFO] [1770560090.255809216] [a1x_serl_node]: Published EEF command: pos=[0.3462, 0.0455, 0.1864], quat=[-0.004, 0.577, -0.020, 0.816]
  0%|          | 323/1000000 [00:59<43:21:31,  6.40it/s][INFO] [1770560090.411354748] [a1x_serl_node]: Published EEF command: pos=[0.3505, 0.0512, 0.1904], quat=[-0.002, 0.567, -0.013, 0.823]
  0%|          | 324/1000000 [00:59<58:38:15,  4.74it/s][INFO] [1770560090.753531847] [a1x_serl_node]: Published EEF command: pos=[0.3447, 0.0514, 0.1920], quat=[-0.004, 0.572, -0.009, 0.820]
  0%|          | 325/1000000 [00:59<54:12:05,  5.12it/s][INFO] [1770560090.909131879] [a1x_serl_node]: Published EEF command: pos=[0.3473, 0.0509, 0.1847], quat=[-0.006, 0.571, -0.016, 0.820]
  0%|          | 326/1000000 [01:00<50:57:05,  5.45it/s][INFO] [1770560091.065937275] [a1x_serl_node]: Published EEF command: pos=[0.3454, 0.0525, 0.1870], quat=[-0.006, 0.571, -0.012, 0.821]
  0%|          | 327/1000000 [01:00<48:41:24,  5.70it/s][INFO] [1770560091.221018295] [a1x_serl_node]: Published EEF command: pos=[0.3435, 0.0467, 0.1880], quat=[-0.006, 0.576, -0.016, 0.817]
  0%|          | 328/1000000 [01:00<47:13:23,  5.88it/s][INFO] [1770560091.379535858] [a1x_serl_node]: Published EEF command: pos=[0.3429, 0.0493, 0.1882], quat=[-0.007, 0.575, -0.016, 0.818]
  0%|          | 329/1000000 [01:00<46:01:37,  6.03it/s][INFO] [1770560091.533987276] [a1x_serl_node]: Published EEF command: pos=[0.3427, 0.0479, 0.1876], quat=[-0.006, 0.578, -0.020, 0.816]
  0%|          | 330/1000000 [01:00<45:07:02,  6.15it/s][INFO] [1770560091.687090899] [a1x_serl_node]: Published EEF command: pos=[0.3428, 0.0461, 0.1811], quat=[-0.007, 0.578, -0.016, 0.816]
  0%|          | 331/1000000 [01:00<44:18:56,  6.27it/s][INFO] [1770560091.840876395] [a1x_serl_node]: Published EEF command: pos=[0.3412, 0.0467, 0.1838], quat=[-0.007, 0.578, -0.022, 0.816]
  0%|          | 332/1000000 [01:01<43:47:27,  6.34it/s][INFO] [1770560091.996666068] [a1x_serl_node]: Published EEF command: pos=[0.3465, 0.0497, 0.1845], quat=[-0.007, 0.576, -0.017, 0.817]
  0%|          | 333/1000000 [01:01<43:38:46,  6.36it/s][INFO] [1770560092.149647512] [a1x_serl_node]: Published EEF command: pos=[0.3439, 0.0501, 0.1865], quat=[-0.009, 0.580, -0.019, 0.814]
  0%|          | 334/1000000 [01:01<43:21:58,  6.40it/s][INFO] [1770560092.306299348] [a1x_serl_node]: Published EEF command: pos=[0.3480, 0.0494, 0.1862], quat=[-0.007, 0.571, -0.019, 0.820]
  0%|          | 335/1000000 [01:01<43:26:48,  6.39it/s][INFO] [1770560092.461538009] [a1x_serl_node]: Published EEF command: pos=[0.3436, 0.0538, 0.1865], quat=[-0.008, 0.580, -0.019, 0.815]
  0%|          | 336/1000000 [01:01<43:16:17,  6.42it/s][INFO] [1770560092.616526040] [a1x_serl_node]: Published EEF command: pos=[0.3485, 0.0520, 0.1873], quat=[-0.007, 0.568, -0.021, 0.823]
