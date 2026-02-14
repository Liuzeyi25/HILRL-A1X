### Todo list

/home/dungeon_master/conrft/serl_robot_infra/franka_env/envs/a1x_env.py

172 def step(self, action: np.ndarray) -> tuple: 

需要设计逻辑：action如果是delta eef，直接执行，如果是joint，也是直接执行

 [SpaceMouse] 已应用: reward=-1.0, succeed=False
last return: -1.0:   0%|                | 587/992999 [02:01<30:31:37,  9.03it/s]CURRENT JOINTS: [ 0.23021277  1.93765957 -1.61191489  0.98       -0.07744681  0.16787234
  0.        ]
Interpolating move TO [-1.53100e-02  1.82555e+00 -1.13900e+00  8.68000e-01 -5.30000e-02
 -1.03000e-01  1.00000e+02] over 20 steps (2.0s).
FINAL JOINTS: [-1.34042553e-02  1.82851064e+00 -1.14191489e+00  8.62553191e-01
 -5.27659574e-02 -1.02340426e-01  8.63544094e+01]
⚠️  Warning: Large position error detected!
   Position error: [1.90574468e-03 2.96063830e-03 2.91489362e-03 5.44680851e-03
 2.34042553e-04 6.59574468e-04 1.36455906e+01]
   Max error: 13.6456 rad
[A1XRobot] Updated IK seed to current joints: [-0.01340426  1.82851064 -1.14191489  0.86255319 -0.05276596 -0.10234043]
last return: -1.0:   0%|               | 588/992999 [02:06<400:33:33,  1.45s/it]Raw action: [ 0.00883234  0.00336304  0.00080106  0.0115923  -0.02230305 -0.00289948
 -0.2       ], Scaled action: [ 0.00883234  0.00336304  0.00080106  0.0115923  -0.02230305 -0.00289948
 -0.2       ]
EEF delta: pos=[0.00883234 0.00336304 0.00080106], rot=[ 0.0115923  -0.02230305 -0.00289948], gripper: 0.864 -> 0.664 (66.4mm)
[a1x_robot] Action to be Solved - pos: [ 0.27169938 -0.00587878  0.18990541], quat[x,y,z,w]: [-0.0450192   0.69181305  0.01437031  0.72052861]
I0213 19:25:53.172770 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.172968 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.00579143  1.8420997  -1.1655214   0.8515486  -0.04134494 -0.09072389]
Best IK solution: [-0.00579143  1.8420997  -1.1655214   0.8515486  -0.04134494 -0.09072389]
🌟 IK 求解耗时: 46.27 ms
[a1x_robot] IK Solution Found - joints: [-0.00579143  1.8420997  -1.1655214   0.8515486  -0.04134494 -0.09072389], max joint diff: 0.0249 rad (1.43°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 1, terminate: False
last return: -1.0:   0%|               | 589/992999 [02:06<289:34:22,  1.05s/it]Raw action: [ 0.0086308   0.000684   -0.00030717  0.01135057 -0.02030068  0.00535007
 -0.2       ], Scaled action: [ 0.0086308   0.000684   -0.00030717  0.01135057 -0.02030068  0.00535007
 -0.2       ]
EEF delta: pos=[ 0.0086308   0.000684   -0.00030717], rot=[ 0.01135057 -0.02030068  0.00535007], gripper: 0.843 -> 0.643 (64.3mm)
[a1x_robot] Action to be Solved - pos: [ 0.27122701 -0.00872166  0.18904739], quat[x,y,z,w]: [-0.04772687  0.69197952  0.01704269  0.72013612]
I0213 19:25:53.280819 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.281017 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01647391  1.841702   -1.1603118   0.8469331  -0.04082449 -0.10900953]
Best IK solution: [-0.01647391  1.841702   -1.1603118   0.8469331  -0.04082449 -0.10900953]
🌟 IK 求解耗时: 53.27 ms
[a1x_robot] IK Solution Found - joints: [-0.01647391  1.841702   -1.1603118   0.8469331  -0.04082449 -0.10900953], max joint diff: 0.0207 rad (1.19°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 2, terminate: False
last return: -1.0:   0%|               | 590/992999 [02:06<212:33:36,  1.30it/s]Raw action: [ 0.01        0.00059255  0.00086452  0.01616061 -0.0245304  -0.00064525
 -0.2       ], Scaled action: [ 0.01        0.00059255  0.00086452  0.01616061 -0.0245304  -0.00064525
 -0.2       ]
EEF delta: pos=[0.01       0.00059255 0.00086452], rot=[ 0.01616061 -0.0245304  -0.00064525], gripper: 0.792 -> 0.592 (59.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.27248016 -0.00892313  0.19019538], quat[x,y,z,w]: [-0.04383787  0.69064189  0.01634654  0.72168193]
I0213 19:25:53.400242 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.400431 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01894127  1.8425373  -1.1677039   0.84973216 -0.03610616 -0.10495451]
Best IK solution: [-0.01894127  1.8425373  -1.1677039   0.84973216 -0.03610616 -0.10495451]
🌟 IK 求解耗时: 54.78 ms
[a1x_robot] IK Solution Found - joints: [-0.01894127  1.8425373  -1.1677039   0.84973216 -0.03610616 -0.10495451], max joint diff: 0.0286 rad (1.64°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 3, terminate: False
last return: -1.0:   0%|               | 591/992999 [02:06<157:47:08,  1.75it/s]Raw action: [ 0.00849777  0.00325649  0.00088134  0.01391901 -0.02352448  0.00463224
 -0.2       ], Scaled action: [ 0.00849777  0.00325649  0.00088134  0.01391901 -0.02352448  0.00463224
 -0.2       ]
EEF delta: pos=[0.00849777 0.00325649 0.00088134], rot=[ 0.01391901 -0.02352448  0.00463224], gripper: 0.740 -> 0.540 (54.0mm)
[a1x_robot] Action to be Solved - pos: [ 0.27137186 -0.00590984  0.19022958], quat[x,y,z,w]: [-0.0449555   0.69026756  0.01761243  0.72194148]
I0213 19:25:53.508618 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.508819 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.0078535   1.8383669  -1.1618752   0.84752935 -0.03627318 -0.09726316]
Best IK solution: [-0.0078535   1.8383669  -1.1618752   0.84752935 -0.03627318 -0.09726316]
🌟 IK 求解耗时: 50.31 ms
[a1x_robot] IK Solution Found - joints: [-0.0078535   1.8383669  -1.1618752   0.84752935 -0.03627318 -0.09726316], max joint diff: 0.0223 rad (1.28°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 4, terminate: False
last return: -1.0:   0%|               | 592/992999 [02:06<119:25:21,  2.31it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.          0.          0.         -0.00285714 -0.         -0.
  0.        ], Scaled action: [-0.          0.          0.         -0.00285714 -0.         -0.
  0.        ]
EEF delta: pos=[-0.  0.  0.], rot=[-0.00285714 -0.         -0.        ], gripper: 0.685 -> 0.685 (68.5mm)
[a1x_robot] Action to be Solved - pos: [ 0.26671499 -0.00826985  0.19017553], quat[x,y,z,w]: [-0.04798163  0.69351113  0.01186189  0.71874848]
I0213 19:25:53.617681 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.617886 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01185623  1.8285117  -1.1475883   0.85142744 -0.04906294 -0.09749223]
Best IK solution: [-0.01185623  1.8285117  -1.1475883   0.85142744 -0.04906294 -0.09749223]
🌟 IK 求解耗时: 55.16 ms
[a1x_robot] IK Solution Found - joints: [-0.01185623  1.8285117  -1.1475883   0.85142744 -0.04906294 -0.09749223], max joint diff: 0.0029 rad (0.17°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 5, terminate: False
last return: -1.0:   0%|                | 593/992999 [02:06<92:36:18,  2.98it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.          0.00285714  0.         -0.04       -0.         -0.
  0.        ], Scaled action: [-0.          0.00285714  0.         -0.04       -0.         -0.
  0.        ]
EEF delta: pos=[-0.          0.00285714  0.        ], rot=[-0.04 -0.   -0.  ], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.26999941 -0.0051359   0.19106531], quat[x,y,z,w]: [-0.06026701  0.68967986  0.00192518  0.72159952]
I0213 19:25:53.726810 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.727006 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.0121952   1.8324776  -1.1564896   0.8469412  -0.08101854 -0.07782477]
Best IK solution: [ 0.0121952   1.8324776  -1.1564896   0.8469412  -0.08101854 -0.07782477]
🌟 IK 求解耗时: 49.53 ms
[a1x_robot] IK Solution Found - joints: [ 0.0121952   1.8324776  -1.1564896   0.8469412  -0.08101854 -0.07782477], max joint diff: 0.0412 rad (2.36°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 6, terminate: False
last return: -1.0:   0%|                | 594/992999 [02:07<73:46:12,  3.74it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.    0.   -0.04  0.02 -0.    0.  ], Scaled action: [-0.    0.    0.   -0.04  0.02 -0.    0.  ]
EEF delta: pos=[-0.  0.  0.], rot=[-0.04  0.02 -0.  ], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.27204081 -0.00811745  0.19159666], quat[x,y,z,w]: [-0.05829021  0.69480485  0.00241879  0.71682816]
I0213 19:25:53.834122 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.834313 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-1.6154761e-04  1.8465862e+00 -1.1802884e+00  8.6989236e-01
 -7.7602327e-02 -8.7466583e-02]
Best IK solution: [-1.6154761e-04  1.8465862e+00 -1.1802884e+00  8.6989236e-01
 -7.7602327e-02 -8.7466583e-02]
🌟 IK 求解耗时: 51.62 ms
[a1x_robot] IK Solution Found - joints: [-1.6154761e-04  1.8465862e+00 -1.1802884e+00  8.6989236e-01
 -7.7602327e-02 -8.7466583e-02], max joint diff: 0.0408 rad (2.34°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 7, terminate: False
last return: -1.0:   0%|                | 595/992999 [02:07<60:36:40,  4.55it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.          0.          0.         -0.00285714  0.02857143 -0.
  0.        ], Scaled action: [-0.          0.          0.         -0.00285714  0.02857143 -0.
  0.        ]
EEF delta: pos=[-0.  0.  0.], rot=[-0.00285714  0.02857143 -0.        ], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.27068667 -0.0079493   0.19082663], quat[x,y,z,w]: [-0.04696464  0.69952243  0.01262317  0.71295396]
I0213 19:25:53.961418 137493517162304 logger.py:71] Updating safety params
I0213 19:25:53.961631 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01109832  1.8505856  -1.1833231   0.88196117 -0.04748698 -0.09593626]
Best IK solution: [-0.01109832  1.8505856  -1.1833231   0.88196117 -0.04748698 -0.09593626]
🌟 IK 求解耗时: 49.35 ms
[a1x_robot] IK Solution Found - joints: [-0.01109832  1.8505856  -1.1833231   0.88196117 -0.04748698 -0.09593626], max joint diff: 0.0369 rad (2.11°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 8, terminate: False
last return: -1.0:   0%|                | 596/992999 [02:07<52:54:13,  5.21it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.    0.   -0.04 -0.   -0.    0.  ], Scaled action: [-0.01  0.    0.   -0.04 -0.   -0.    0.  ]
EEF delta: pos=[-0.01  0.    0.  ], rot=[-0.04 -0.   -0.  ], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.25901421 -0.00846287  0.18982085], quat[x,y,z,w]: [-0.06640418  0.69151686 -0.00867331  0.71924939]
I0213 19:25:54.070290 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.070479 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.00944709  1.801608   -1.1054512   0.83192945 -0.10491646 -0.0745877 ]
Best IK solution: [ 0.00944709  1.801608   -1.1054512   0.83192945 -0.10491646 -0.0745877 ]
🌟 IK 求解耗时: 48.24 ms
[a1x_robot] IK Solution Found - joints: [ 0.00944709  1.801608   -1.1054512   0.83192945 -0.10491646 -0.0745877 ], max joint diff: 0.0458 rad (2.63°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 9, terminate: False
last return: -1.0:   0%|                | 597/992999 [02:07<46:05:02,  5.98it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.00857143  0.         -0.01       -0.04        0.04       -0.
  0.        ], Scaled action: [-0.00857143  0.         -0.01       -0.04        0.04       -0.
  0.        ]
EEF delta: pos=[-0.00857143  0.         -0.01      ], rot=[-0.04  0.04 -0.  ], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.2600052  -0.00826531  0.18024436], quat[x,y,z,w]: [-0.07249334  0.70747014 -0.01023905  0.70294088]
I0213 19:25:54.180557 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.180793 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.01505998  1.8408988  -1.1081953   0.8411146  -0.11720485 -0.07298258]
Best IK solution: [ 0.01505998  1.8408988  -1.1081953   0.8411146  -0.11720485 -0.07298258]
🌟 IK 求解耗时: 51.23 ms
[a1x_robot] IK Solution Found - joints: [ 0.01505998  1.8408988  -1.1081953   0.8411146  -0.11720485 -0.07298258], max joint diff: 0.0469 rad (2.69°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 10, terminate: False
last return: -1.0:   0%|                | 598/992999 [02:07<41:20:39,  6.67it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01       -0.00571429 -0.01       -0.04        0.01142857 -0.
  0.        ], Scaled action: [-0.01       -0.00571429 -0.01       -0.04        0.01142857 -0.
  0.        ]
EEF delta: pos=[-0.01       -0.00571429 -0.01      ], rot=[-0.04        0.01142857 -0.        ], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.26044856 -0.01377772  0.18205327], quat[x,y,z,w]: [-0.07237014  0.6962111  -0.01073621  0.71409902]
I0213 19:25:54.289334 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.289528 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.00654222  1.8212816  -1.0897723   0.8079694  -0.11616751 -0.0957232 ]
Best IK solution: [-0.00654222  1.8212816  -1.0897723   0.8079694  -0.11616751 -0.0957232 ]
🌟 IK 求解耗时: 47.63 ms
[a1x_robot] IK Solution Found - joints: [-0.00654222  1.8212816  -1.0897723   0.8079694  -0.11616751 -0.0957232 ], max joint diff: 0.0787 rad (4.51°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 11, terminate: False
last return: -1.0:   0%|                | 599/992999 [02:07<37:56:25,  7.27it/s] [Actor] Step 7600: actions shape = (7,)
[SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.04 -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.04 -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0.04 -0.   -0.  ], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.26122115 -0.01722984  0.18275301], quat[x,y,z,w]: [-0.06790956  0.69249866 -0.00714878  0.71818019]
I0213 19:25:54.398700 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.398946 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.02475365  1.8165884  -1.0873859   0.7979887  -0.10347404 -0.1130977 ]
Best IK solution: [-0.02475365  1.8165884  -1.0873859   0.7979887  -0.10347404 -0.1130977 ]
🌟 IK 求解耗时: 50.43 ms
[a1x_robot] IK Solution Found - joints: [-0.02475365  1.8165884  -1.0873859   0.7979887  -0.10347404 -0.1130977 ], max joint diff: 0.0888 rad (5.09°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 12, terminate: False
 [Actor] Step 7600: Transition actions shape = (7,), intervened = True
last return: -1.0:   0%|                | 600/992999 [02:07<37:16:57,  7.39it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.26051915 -0.01710459  0.18142313], quat[x,y,z,w]: [-0.05200603  0.69281241  0.00570501  0.71921749]
I0213 19:25:54.527220 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.527440 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.04074195  1.816342   -1.0816941   0.7935369  -0.0622193  -0.12376203]
Best IK solution: [-0.04074195  1.816342   -1.0816941   0.7935369  -0.0622193  -0.12376203]
🌟 IK 求解耗时: 54.76 ms
[a1x_robot] IK Solution Found - joints: [-0.04074195  1.816342   -1.0816941   0.7935369  -0.0622193  -0.12376203], max joint diff: 0.0864 rad (4.95°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 13, terminate: False
last return: -1.0:   0%|                | 601/992999 [02:07<35:07:51,  7.85it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.25832612 -0.01818765  0.17827342], quat[x,y,z,w]: [-0.05825228  0.69417577  0.00082755  0.71744406]
I0213 19:25:54.636453 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.636698 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.03881567  1.8149538  -1.0616813   0.77809316 -0.07830022 -0.12401596]
Best IK solution: [-0.03881567  1.8149538  -1.0616813   0.77809316 -0.07830022 -0.12401596]
🌟 IK 求解耗时: 56.02 ms
[a1x_robot] IK Solution Found - joints: [-0.03881567  1.8149538  -1.0616813   0.77809316 -0.07830022 -0.12401596], max joint diff: 0.0860 rad (4.93°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 14, terminate: False
last return: -1.0:   0%|                | 602/992999 [02:08<33:35:58,  8.20it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.2575523  -0.02061638  0.17506474], quat[x,y,z,w]: [-0.06507695  0.6914188  -0.00194673  0.71951459]
I0213 19:25:54.745409 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.745614 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.04325781  1.8119462  -1.0384634   0.7492518  -0.09092892 -0.13487153]
Best IK solution: [-0.04325781  1.8119462  -1.0384634   0.7492518  -0.09092892 -0.13487153]
🌟 IK 求解耗时: 55.57 ms
[a1x_robot] IK Solution Found - joints: [-0.04325781  1.8119462  -1.0384634   0.7492518  -0.09092892 -0.13487153], max joint diff: 0.0852 rad (4.88°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 15, terminate: False
last return: -1.0:   0%|                | 603/992999 [02:08<32:35:02,  8.46it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.25645882 -0.02322446  0.17311497], quat[x,y,z,w]: [-0.06434079  0.68953234 -0.00244743  0.72138715]
I0213 19:25:54.855682 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.855872 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.05411488  1.8081323  -1.0218513   0.7302833  -0.0895923  -0.14422034]
Best IK solution: [-0.05411488  1.8081323  -1.0218513   0.7302833  -0.0895923  -0.14422034]
🌟 IK 求解耗时: 51.20 ms
[a1x_robot] IK Solution Found - joints: [-0.05411488  1.8081323  -1.0218513   0.7302833  -0.0895923  -0.14422034], max joint diff: 0.0841 rad (4.82°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 16, terminate: False
last return: -1.0:   0%|                | 604/992999 [02:08<31:52:39,  8.65it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.25407691 -0.02496969  0.17091027], quat[x,y,z,w]: [-6.23554953e-02  6.90472255e-01 -2.84008939e-04  7.20666204e-01]
I0213 19:25:54.966232 137493517162304 logger.py:71] Updating safety params
I0213 19:25:54.966432 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.06393784  1.8051922  -1.0056893   0.7191632  -0.08347113 -0.15403351]
Best IK solution: [-0.06393784  1.8051922  -1.0056893   0.7191632  -0.08347113 -0.15403351]
🌟 IK 求解耗时: 50.75 ms
[a1x_robot] IK Solution Found - joints: [-0.06393784  1.8051922  -1.0056893   0.7191632  -0.08347113 -0.15403351], max joint diff: 0.0830 rad (4.76°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 17, terminate: False
last return: -1.0:   0%|                | 605/992999 [02:08<31:28:16,  8.76it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.25186925 -0.02676647  0.16869694], quat[x,y,z,w]: [-0.060431    0.69091881  0.00130699  0.72040099]
I0213 19:25:55.075307 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.075500 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.0737661   1.8022634  -0.9892854   0.7064928  -0.07814271 -0.16320209]
Best IK solution: [-0.0737661   1.8022634  -0.9892854   0.7064928  -0.07814271 -0.16320209]
🌟 IK 求解耗时: 55.81 ms
[a1x_robot] IK Solution Found - joints: [-0.0737661   1.8022634  -0.9892854   0.7064928  -0.07814271 -0.16320209], max joint diff: 0.0820 rad (4.70°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 18, terminate: False
last return: -1.0:   0%|                | 606/992999 [02:08<31:02:20,  8.88it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01 -0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01 -0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.25184576 -0.02853869  0.1667111 ], quat[x,y,z,w]: [-6.22942404e-02  6.87598426e-01  6.04200145e-04  7.23413759e-01]
I0213 19:25:55.185123 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.185317 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.07987803  1.7999697  -0.97401035  0.6835945  -0.08026473 -0.17143048]
Best IK solution: [-0.07987803  1.7999697  -0.97401035  0.6835945  -0.08026473 -0.17143048]
🌟 IK 求解耗时: 51.66 ms
[a1x_robot] IK Solution Found - joints: [-0.07987803  1.7999697  -0.97401035  0.6835945  -0.08026473 -0.17143048], max joint diff: 0.0813 rad (4.66°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 19, terminate: False
last return: -1.0:   0%|                | 607/992999 [02:08<30:50:04,  8.94it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01  0.   -0.01], rot=[-0. -0. -0.], gripper: 0.652 -> 0.652 (65.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.25022463 -0.02088657  0.1640658 ], quat[x,y,z,w]: [-6.41461879e-02  6.86591052e-01  2.40191808e-04  7.24208489e-01]
I0213 19:25:55.294790 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.295007 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.04805554  1.796194   -0.953244    0.6662457  -0.08499255 -0.14188923]
Best IK solution: [-0.04805554  1.796194   -0.953244    0.6662457  -0.08499255 -0.14188923]
🌟 IK 求解耗时: 54.43 ms
[a1x_robot] IK Solution Found - joints: [-0.04805554  1.796194   -0.953244    0.6662457  -0.08499255 -0.14188923], max joint diff: 0.0821 rad (4.70°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 20, terminate: False
last return: -1.0:   0%|                | 608/992999 [02:08<30:40:07,  8.99it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01  0.   -0.01], rot=[-0. -0. -0.], gripper: 0.651 -> 0.651 (65.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.24749139 -0.02332349  0.16129913], quat[x,y,z,w]: [-0.06376212  0.68726957  0.00116066  0.72359767]
I0213 19:25:55.405293 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.405489 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.05938035  1.7930743  -0.9329983   0.6501574  -0.08263032 -0.15385808]
Best IK solution: [-0.05938035  1.7930743  -0.9329983   0.6501574  -0.08263032 -0.15385808]
🌟 IK 求解耗时: 47.37 ms
[a1x_robot] IK Solution Found - joints: [-0.05938035  1.7930743  -0.9329983   0.6501574  -0.08263032 -0.15385808], max joint diff: 0.0815 rad (4.67°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 21, terminate: False
last return: -1.0:   0%|                | 609/992999 [02:08<30:35:54,  9.01it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01  0.   -0.01], rot=[-0. -0. -0.], gripper: 0.651 -> 0.651 (65.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.24625069 -0.02544443  0.1591537 ], quat[x,y,z,w]: [-0.06326351  0.68558518  0.00140045  0.72523702]
I0213 19:25:55.515057 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.515255 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.06913762  1.7902454  -0.9162512   0.63089156 -0.08045813 -0.16338997]
Best IK solution: [-0.06913762  1.7902454  -0.9162512   0.63089156 -0.08045813 -0.16338997]
🌟 IK 求解耗时: 55.08 ms
[a1x_robot] IK Solution Found - joints: [-0.06913762  1.7902454  -0.9162512   0.63089156 -0.08045813 -0.16338997], max joint diff: 0.0810 rad (4.64°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 22, terminate: False
last return: -1.0:   0%|                | 610/992999 [02:08<31:35:06,  8.73it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.01  0.   -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.01  0.   -0.01], rot=[-0. -0. -0.], gripper: 0.651 -> 0.651 (65.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.24478203 -0.02688726  0.15726238], quat[x,y,z,w]: [-6.27452058e-02  6.84704450e-01  1.33511238e-05  7.26114905e-01]
I0213 19:25:55.637409 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.637613 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.0751538   1.7875693  -0.901146    0.61546296 -0.08106805 -0.16684628]
Best IK solution: [-0.0751538   1.7875693  -0.901146    0.61546296 -0.08106805 -0.16684628]
🌟 IK 求解耗时: 51.65 ms
[a1x_robot] IK Solution Found - joints: [-0.0751538   1.7875693  -0.901146    0.61546296 -0.08106805 -0.16684628], max joint diff: 0.0806 rad (4.62°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 23, terminate: False
last return: -1.0:   0%|                | 611/992999 [02:09<31:07:51,  8.85it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01        0.01       -0.01       -0.03142857 -0.         -0.
  0.        ], Scaled action: [-0.01        0.01       -0.01       -0.03142857 -0.         -0.
  0.        ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.03142857 -0.         -0.        ], gripper: 0.647 -> 0.647 (64.7mm)
[a1x_robot] Action to be Solved - pos: [ 0.24339858 -0.01624414  0.15517161], quat[x,y,z,w]: [-0.07325592  0.68361233 -0.01169981  0.72606533]
I0213 19:25:55.747851 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.748042 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01708709  1.7847806  -0.88281095  0.60125273 -0.1162784  -0.10830163]
Best IK solution: [-0.01708709  1.7847806  -0.88281095  0.60125273 -0.1162784  -0.10830163]
🌟 IK 求解耗时: 55.12 ms
[a1x_robot] IK Solution Found - joints: [-0.01708709  1.7847806  -0.88281095  0.60125273 -0.1162784  -0.10830163], max joint diff: 0.0825 rad (4.73°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 24, terminate: False
last return: -1.0:   0%|                | 612/992999 [02:09<30:54:02,  8.92it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01        0.01       -0.01       -0.04        0.01714286 -0.04
  0.        ], Scaled action: [-0.01        0.01       -0.01       -0.04        0.01714286 -0.04
  0.        ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04        0.01714286 -0.04      ], gripper: 0.647 -> 0.647 (64.7mm)
[a1x_robot] Action to be Solved - pos: [ 0.2421078  -0.01551515  0.15236117], quat[x,y,z,w]: [-0.0637149   0.68894946 -0.0275278   0.72147853]
I0213 19:25:55.858108 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.858320 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.00942669  1.7945036  -0.8797392   0.60521966 -0.12738606 -0.06397179]
Best IK solution: [-0.00942669  1.7945036  -0.8797392   0.60521966 -0.12738606 -0.06397179]
🌟 IK 求解耗时: 52.81 ms
[a1x_robot] IK Solution Found - joints: [-0.00942669  1.7945036  -0.8797392   0.60521966 -0.12738606 -0.06397179], max joint diff: 0.0956 rad (5.48°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 25, terminate: False
last return: -1.0:   0%|                | 613/992999 [02:09<30:42:24,  8.98it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04 -0.04  0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04 -0.04  0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04 -0.04], gripper: 0.647 -> 0.647 (64.7mm)
[a1x_robot] Action to be Solved - pos: [ 0.24007312 -0.0159667   0.14950337], quat[x,y,z,w]: [-0.0642492   0.69617742 -0.02632901  0.7145039 ]
I0213 19:25:55.967332 137493517162304 logger.py:71] Updating safety params
I0213 19:25:55.967551 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01152176  1.8040321  -0.87687147  0.61263144 -0.12707531 -0.06725142]
Best IK solution: [-0.01152176  1.8040321  -0.87687147  0.61263144 -0.12707531 -0.06725142]
🌟 IK 求解耗时: 65.09 ms
[a1x_robot] IK Solution Found - joints: [-0.01152176  1.8040321  -0.87687147  0.61263144 -0.12707531 -0.06725142], max joint diff: 0.0957 rad (5.48°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 26, terminate: False
last return: -1.0:   0%|                | 614/992999 [02:09<30:36:25,  9.01it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01        0.01       -0.01       -0.04        0.04       -0.00857143
  0.        ], Scaled action: [-0.01        0.01       -0.01       -0.04        0.04       -0.00857143
  0.        ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04        0.04       -0.00857143], gripper: 0.647 -> 0.647 (64.7mm)
[a1x_robot] Action to be Solved - pos: [ 0.2373913  -0.01656545  0.14774837], quat[x,y,z,w]: [-0.0748825   0.69654452 -0.01550365  0.71342692]
I0213 19:25:56.078093 137493517162304 logger.py:71] Updating safety params
I0213 19:25:56.078298 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01449159  1.7990081  -0.8591284   0.59965354 -0.12633502 -0.10065352]
Best IK solution: [-0.01449159  1.7990081  -0.8591284   0.59965354 -0.12633502 -0.10065352]
🌟 IK 求解耗时: 52.72 ms
[a1x_robot] IK Solution Found - joints: [-0.01449159  1.7990081  -0.8591284   0.59965354 -0.12633502 -0.10065352], max joint diff: 0.0642 rad (3.68°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 27, terminate: False
last return: -1.0:   0%|                | 615/992999 [02:09<30:33:43,  9.02it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04 -0.    0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04 -0.    0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04 -0.  ], gripper: 0.647 -> 0.647 (64.7mm)
[a1x_robot] Action to be Solved - pos: [ 0.2356111  -0.01654453  0.1467908 ], quat[x,y,z,w]: [-0.07838416  0.69751529 -0.01784287  0.71204633]
I0213 19:25:56.188761 137493517162304 logger.py:71] Updating safety params
I0213 19:25:56.188962 137493517162304 logger.py:71] Updating optimizer params
prev_q: [-0.01077999  1.797187   -0.85010785  0.59570444 -0.13487484 -0.09851992]
Best IK solution: [-0.01077999  1.797187   -0.85010785  0.59570444 -0.13487484 -0.09851992]
🌟 IK 求解耗时: 143.03 ms
[a1x_robot] IK Solution Found - joints: [-0.01077999  1.797187   -0.85010785  0.59570444 -0.13487484 -0.09851992], max joint diff: 0.0588 rad (3.37°)
⏱️  ✓ 执行耗时=145ms, 误差=0.0mm
Step done: False, reward: False, path length: 28, terminate: False
last return: -1.0:   0%|                | 616/992999 [02:09<34:14:22,  8.05it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01        0.01       -0.01       -0.04        0.02285714  0.04
  0.        ], Scaled action: [-0.01        0.01       -0.01       -0.04        0.02285714  0.04
  0.        ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04        0.02285714  0.04      ], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.23421811 -0.01324469  0.14564081], quat[x,y,z,w]: [-0.08799808  0.69114964 -0.01688092  0.71713565]
I0213 19:25:56.343158 137493517162304 logger.py:71] Updating safety params
I0213 19:25:56.343499 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.00846529  1.7863486  -0.8261406   0.5672949  -0.14673541 -0.09562756]
Best IK solution: [ 0.00846529  1.7863486  -0.8261406   0.5672949  -0.14673541 -0.09562756]
🌟 IK 求解耗时: 55.32 ms
[a1x_robot] IK Solution Found - joints: [ 0.00846529  1.7863486  -0.8261406   0.5672949  -0.14673541 -0.09562756], max joint diff: 0.0658 rad (3.77°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 29, terminate: False
last return: -1.0:   0%|                | 617/992999 [02:09<33:00:20,  8.35it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01        0.01       -0.01       -0.04        0.01714286  0.04
  0.        ], Scaled action: [-0.01        0.01       -0.01       -0.04        0.01714286  0.04
  0.        ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04        0.01714286  0.04      ], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.23320115 -0.01085734  0.14434418], quat[x,y,z,w]: [-0.08292137  0.6903283  -0.02495566  0.71829527]
I0213 19:25:56.452272 137493517162304 logger.py:71] Updating safety params
I0213 19:25:56.452478 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.02101409  1.7855628  -0.8165602   0.5588113  -0.15184388 -0.06451267]
Best IK solution: [ 0.02101409  1.7855628  -0.8165602   0.5588113  -0.15184388 -0.06451267]
🌟 IK 求解耗时: 262.88 ms
[a1x_robot] IK Solution Found - joints: [ 0.02101409  1.7855628  -0.8165602   0.5588113  -0.15184388 -0.06451267], max joint diff: 0.0694 rad (3.98°)
⏱️  ✓ 执行耗时=264ms, 误差=0.0mm
Step done: False, reward: False, path length: 30, terminate: False
last return: -1.0:   0%|                | 618/992999 [02:09<45:43:25,  6.03it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.23024808 -0.00809808  0.14061464], quat[x,y,z,w]: [-0.09343456  0.70069248 -0.02557942  0.70685622]
I0213 19:25:56.727964 137493517162304 logger.py:71] Updating safety params
I0213 19:25:56.728169 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.04070238  1.8013583  -0.81486773  0.5742278  -0.16843575 -0.05648973]
Best IK solution: [ 0.04070238  1.8013583  -0.81486773  0.5742278  -0.16843575 -0.05648973]
🌟 IK 求解耗时: 48.08 ms
[a1x_robot] IK Solution Found - joints: [ 0.04070238  1.8013583  -0.81486773  0.5742278  -0.16843575 -0.05648973], max joint diff: 0.0609 rad (3.49°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 31, terminate: False
last return: -1.0:   0%|                | 619/992999 [02:10<41:15:51,  6.68it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.22882183 -0.0062683   0.13789703], quat[x,y,z,w]: [-0.10526252  0.69950748 -0.0209301   0.70652036]
I0213 19:25:56.838050 137493517162304 logger.py:71] Updating safety params
I0213 19:25:56.838239 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.05353975  1.8018471  -0.7972374   0.55500096 -0.17862722 -0.06727824]
Best IK solution: [ 0.05353975  1.8018471  -0.7972374   0.55500096 -0.17862722 -0.06727824]
🌟 IK 求解耗时: 50.20 ms
[a1x_robot] IK Solution Found - joints: [ 0.05353975  1.8018471  -0.7972374   0.55500096 -0.17862722 -0.06727824], max joint diff: 0.0616 rad (3.53°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 32, terminate: False
last return: -1.0:   0%|                | 620/992999 [02:10<41:02:38,  6.72it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.22588619 -0.00411034  0.13605124], quat[x,y,z,w]: [-0.11474299  0.70074277 -0.01696354  0.70392177]
I0213 19:25:56.985364 137493517162304 logger.py:71] Updating safety params
I0213 19:25:56.985556 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.06740141  1.8002102  -0.7822212   0.5478551  -0.1865179  -0.0720598 ]
Best IK solution: [ 0.06740141  1.8002102  -0.7822212   0.5478551  -0.1865179  -0.0720598 ]
🌟 IK 求解耗时: 47.71 ms
[a1x_robot] IK Solution Found - joints: [ 0.06740141  1.8002102  -0.7822212   0.5478551  -0.1865179  -0.0720598 ], max joint diff: 0.0627 rad (3.59°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 33, terminate: False
last return: -1.0:   0%|                | 621/992999 [02:10<37:49:32,  7.29it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.22481745 -0.00295565  0.13525833], quat[x,y,z,w]: [-0.11398776  0.70119812 -0.02045453  0.70349811]
I0213 19:25:57.093981 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.094167 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.0746452   1.8004198  -0.77695024  0.54560655 -0.19042972 -0.05850133]
Best IK solution: [ 0.0746452   1.8004198  -0.77695024  0.54560655 -0.19042972 -0.05850133]
🌟 IK 求解耗时: 51.54 ms
[a1x_robot] IK Solution Found - joints: [ 0.0746452   1.8004198  -0.77695024  0.54560655 -0.19042972 -0.05850133], max joint diff: 0.0632 rad (3.62°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 34, terminate: False
last return: -1.0:   0%|                | 622/992999 [02:10<35:28:29,  7.77it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.22299055 -0.00095118  0.13399645], quat[x,y,z,w]: [-0.11718321  0.70355976 -0.0243198   0.70048577]
I0213 19:25:57.204319 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.204509 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.08879421  1.8035386  -0.771983    0.5482711  -0.20028919 -0.04238258]
Best IK solution: [ 0.08879421  1.8035386  -0.771983    0.5482711  -0.20028919 -0.04238258]
🌟 IK 求解耗时: 48.76 ms
[a1x_robot] IK Solution Found - joints: [ 0.08879421  1.8035386  -0.771983    0.5482711  -0.20028919 -0.04238258], max joint diff: 0.0639 rad (3.66°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 35, terminate: False
last return: -1.0:   0%|                | 623/992999 [02:10<33:56:41,  8.12it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.01  0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [0.22118464 0.00149631 0.13160517], quat[x,y,z,w]: [-0.12663336  0.70475989 -0.0232947   0.69766385]
I0213 19:25:57.314631 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.314830 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.10599919  1.8079284  -0.76104957  0.5411284  -0.21198255 -0.03926926]
Best IK solution: [ 0.10599919  1.8079284  -0.76104957  0.5411284  -0.21198255 -0.03926926]
🌟 IK 求解耗时: 47.66 ms
[a1x_robot] IK Solution Found - joints: [ 0.10599919  1.8079284  -0.76104957  0.5411284  -0.21198255 -0.03926926], max joint diff: 0.0639 rad (3.66°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 36, terminate: False
last return: -1.0:   0%|                | 624/992999 [02:10<32:53:35,  8.38it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.    0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [0.22846574 0.00429144 0.12882386], quat[x,y,z,w]: [-0.13597906  0.70693025 -0.01931133  0.69382014]
I0213 19:25:57.424374 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.424577 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.11807469  1.8403145  -0.78983814  0.5470129  -0.21924661 -0.04517846]
Best IK solution: [ 0.11807469  1.8403145  -0.78983814  0.5470129  -0.21924661 -0.04517846]
🌟 IK 求解耗时: 48.88 ms
[a1x_robot] IK Solution Found - joints: [ 0.11807469  1.8403145  -0.78983814  0.5470129  -0.21924661 -0.04517846], max joint diff: 0.0604 rad (3.46°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 37, terminate: False
last return: -1.0:   0%|                | 625/992999 [02:10<32:08:48,  8.58it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01 -0.04  0.04  0.04  0.  ], Scaled action: [-0.    0.01 -0.01 -0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[-0.04  0.04  0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [0.22620223 0.00613251 0.12693287], quat[x,y,z,w]: [-0.1396698   0.70888089 -0.02215813  0.69100596]
I0213 19:25:57.535328 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.535584 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.13125753  1.8435725  -0.7806663   0.54472715 -0.2278561  -0.03202545]
Best IK solution: [ 0.13125753  1.8435725  -0.7806663   0.54472715 -0.2278561  -0.03202545]
🌟 IK 求解耗时: 46.90 ms
[a1x_robot] IK Solution Found - joints: [ 0.13125753  1.8435725  -0.7806663   0.54472715 -0.2278561  -0.03202545], max joint diff: 0.0606 rad (3.47°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 38, terminate: False
last return: -1.0:   0%|                | 626/992999 [02:10<31:36:20,  8.72it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01  0.04  0.04  0.04  0.  ], Scaled action: [-0.    0.01 -0.01  0.04  0.04  0.04  0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[0.04 0.04 0.04], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [0.22480138 0.00836209 0.12542982], quat[x,y,z,w]: [-0.11680071  0.71114655  0.00362074  0.69326407]
I0213 19:25:57.643931 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.644116 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.11080381  1.8381232  -0.77172005  0.5339937  -0.15953085 -0.05790669]
Best IK solution: [ 0.11080381  1.8381232  -0.77172005  0.5339937  -0.15953085 -0.05790669]
🌟 IK 求解耗时: 46.18 ms
[a1x_robot] IK Solution Found - joints: [ 0.11080381  1.8381232  -0.77172005  0.5339937  -0.15953085 -0.05790669], max joint diff: 0.0428 rad (2.45°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 39, terminate: False
last return: -1.0:   0%|                | 627/992999 [02:11<31:07:36,  8.86it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.          0.00571429 -0.00571429  0.04       -0.         -0.
  0.        ], Scaled action: [-0.          0.00571429 -0.00571429  0.04       -0.         -0.
  0.        ]
EEF delta: pos=[-0.          0.00571429 -0.00571429], rot=[ 0.04 -0.   -0.  ], gripper: 0.646 -> 0.646 (64.6mm)
[a1x_robot] Action to be Solved - pos: [0.22319265 0.00623157 0.12769265], quat[x,y,z,w]: [-0.10984984  0.70217539 -0.00976639  0.70341122]
I0213 19:25:57.753012 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.753231 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.10659821  1.815009   -0.75141424  0.5115131  -0.16932508 -0.03520976]
Best IK solution: [ 0.10659821  1.815009   -0.75141424  0.5115131  -0.16932508 -0.03520976]
🌟 IK 求解耗时: 45.71 ms
[a1x_robot] IK Solution Found - joints: [ 0.10659821  1.815009   -0.75141424  0.5115131  -0.16932508 -0.03520976], max joint diff: 0.0398 rad (2.28°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 40, terminate: False
last return: -1.0:   0%|                | 628/992999 [02:11<30:49:03,  8.94it/s]Raw action: [ 0.00804801  0.00305355  0.00313563  0.02400789 -0.03736814 -0.00365536
 -0.10848074], Scaled action: [ 0.00804801  0.00305355  0.00313563  0.02400789 -0.03736814 -0.00365536
 -0.10848074]
EEF delta: pos=[0.00804801 0.00305355 0.00313563], rot=[ 0.02400789 -0.03736814 -0.00365536], gripper: 0.646 -> 0.538 (53.8mm)
[a1x_robot] Action to be Solved - pos: [0.23208862 0.00562889 0.13454092], quat[x,y,z,w]: [-0.12055262  0.69071827 -0.0189713   0.71275201]
I0213 19:25:57.861789 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.861975 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.11294907  1.8120761  -0.7878449   0.523111   -0.19874024 -0.03389326]
Best IK solution: [ 0.11294907  1.8120761  -0.7878449   0.523111   -0.19874024 -0.03389326]
🌟 IK 求解耗时: 46.52 ms
[a1x_robot] IK Solution Found - joints: [ 0.11294907  1.8120761  -0.7878449   0.523111   -0.19874024 -0.03389326], max joint diff: 0.0196 rad (1.12°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 41, terminate: False
last return: -1.0:   0%|                | 629/992999 [02:11<30:33:20,  9.02it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.   -0.01  0.04 -0.   -0.    0.  ], Scaled action: [-0.    0.   -0.01  0.04 -0.   -0.    0.  ]
EEF delta: pos=[-0.    0.   -0.01], rot=[ 0.04 -0.   -0.  ], gripper: 0.633 -> 0.633 (63.3mm)
[a1x_robot] Action to be Solved - pos: [0.22511785 0.00547653 0.11926261], quat[x,y,z,w]: [-0.12121554  0.70497728 -0.00499072  0.69877673]
I0213 19:25:57.970735 137493517162304 logger.py:71] Updating safety params
I0213 19:25:57.970921 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.10652866  1.8445085  -0.7381407   0.47762102 -0.17845845 -0.05750953]
Best IK solution: [ 0.10652866  1.8445085  -0.7381407   0.47762102 -0.17845845 -0.05750953]
🌟 IK 求解耗时: 49.62 ms
[a1x_robot] IK Solution Found - joints: [ 0.10652866  1.8445085  -0.7381407   0.47762102 -0.17845845 -0.05750953], max joint diff: 0.0607 rad (3.48°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 42, terminate: False
last return: -1.0:   0%|                | 630/992999 [02:11<32:10:53,  8.57it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.01        0.01       -0.01       -0.         -0.         -0.01428571
  0.        ], Scaled action: [-0.01        0.01       -0.01       -0.         -0.         -0.01428571
  0.        ]
EEF delta: pos=[-0.01  0.01 -0.01], rot=[-0.         -0.         -0.01428571], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.21569969 0.01777861 0.11739291], quat[x,y,z,w]: [-0.12202878  0.70506553 -0.01679042  0.69836212]
I0213 19:25:58.102928 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.103123 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.17545938  1.8332523  -0.70512766  0.47224826 -0.19465885  0.02865397]
Best IK solution: [ 0.17545938  1.8332523  -0.70512766  0.47224826 -0.19465885  0.02865397]
🌟 IK 求解耗时: 48.38 ms
[a1x_robot] IK Solution Found - joints: [ 0.17545938  1.8332523  -0.70512766  0.47224826 -0.19465885  0.02865397], max joint diff: 0.0665 rad (3.81°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 43, terminate: False
last return: -1.0:   0%|                | 631/992999 [02:11<31:40:53,  8.70it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.          0.01       -0.01       -0.         -0.          0.01142857
  0.        ], Scaled action: [-0.          0.01       -0.01       -0.         -0.          0.01142857
  0.        ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[-0.         -0.          0.01142857], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.2250524  0.01837884 0.11665955], quat[x,y,z,w]: [-0.12080191  0.70357516 -0.00314161  0.70027068]
I0213 19:25:58.211800 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.211999 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.161697    1.8550953  -0.7371956   0.4714492  -0.17459378 -0.00391585]
Best IK solution: [ 0.161697    1.8550953  -0.7371956   0.4714492  -0.17459378 -0.00391585]
🌟 IK 求解耗时: 46.75 ms
[a1x_robot] IK Solution Found - joints: [ 0.161697    1.8550953  -0.7371956   0.4714492  -0.17459378 -0.00391585], max joint diff: 0.0471 rad (2.70°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 44, terminate: False
last return: -1.0:   0%|                | 632/992999 [02:11<31:09:29,  8.85it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.22565882 0.0184215  0.11681153], quat[x,y,z,w]: [-0.11450624  0.70264662 -0.00550947  0.70224333]
I0213 19:25:58.322047 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.322235 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.1590469   1.854274   -0.7376808   0.46877214 -0.1693052   0.00541876]
Best IK solution: [ 0.1590469   1.854274   -0.7376808   0.46877214 -0.1693052   0.00541876]
🌟 IK 求解耗时: 49.73 ms
[a1x_robot] IK Solution Found - joints: [ 0.1590469   1.854274   -0.7376808   0.46877214 -0.1693052   0.00541876], max joint diff: 0.0448 rad (2.57°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 45, terminate: False
last return: -1.0:   0%|                | 633/992999 [02:11<30:56:34,  8.91it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.22564741 0.01773106 0.11565298], quat[x,y,z,w]: [-0.11586169  0.70147856 -0.00646873  0.70317996]
I0213 19:25:58.430864 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.431073 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.15763465  1.8553913  -0.7310113   0.4579635  -0.17284723  0.00319339]
Best IK solution: [ 0.15763465  1.8553913  -0.7310113   0.4579635  -0.17284723  0.00319339]
🌟 IK 求解耗时: 50.88 ms
[a1x_robot] IK Solution Found - joints: [ 0.15763465  1.8553913  -0.7310113   0.4579635  -0.17284723  0.00319339], max joint diff: 0.0448 rad (2.57°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 46, terminate: False
last return: -1.0:   0%|                | 634/992999 [02:11<30:41:46,  8.98it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.22571837 0.02012076 0.1138673 ], quat[x,y,z,w]: [-0.11532781  0.69965802 -0.00806644  0.70506247]
I0213 19:25:58.540431 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.540636 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.16887254  1.8590375  -0.72414106  0.44454202 -0.1746903   0.0172041 ]
Best IK solution: [ 0.16887254  1.8590375  -0.72414106  0.44454202 -0.1746903   0.0172041 ]
🌟 IK 求解耗时: 50.91 ms
[a1x_robot] IK Solution Found - joints: [ 0.16887254  1.8590375  -0.72414106  0.44454202 -0.1746903   0.0172041 ], max joint diff: 0.0446 rad (2.56°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 47, terminate: False
last return: -1.0:   0%|                | 635/992999 [02:11<30:32:15,  9.03it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ], Scaled action: [-0.    0.01 -0.01 -0.   -0.   -0.    0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[-0. -0. -0.], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.22640986 0.02462813 0.11174818], quat[x,y,z,w]: [-0.11318106  0.6977147  -0.00860773  0.70732605]
I0213 19:25:58.649916 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.650107 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18709733  1.865978   -0.720583    0.43135473 -0.17287764  0.03903089]
Best IK solution: [ 0.18709733  1.865978   -0.720583    0.43135473 -0.17287764  0.03903089]
🌟 IK 求解耗时: 47.27 ms
[a1x_robot] IK Solution Found - joints: [ 0.18709733  1.865978   -0.720583    0.43135473 -0.17287764  0.03903089], max joint diff: 0.0444 rad (2.54°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 48, terminate: False
last return: -1.0:   0%|                | 636/992999 [02:12<30:24:28,  9.07it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01  0.02 -0.   -0.    0.  ], Scaled action: [-0.    0.01 -0.01  0.02 -0.   -0.    0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[ 0.02 -0.   -0.  ], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.22726918 0.02708837 0.10968412], quat[x,y,z,w]: [-0.10533917  0.69617496 -0.00152479  0.71009982]
I0213 19:25:58.758844 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.759033 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18782946  1.8696444  -0.714289    0.4130642  -0.15253948  0.04026256]
Best IK solution: [ 0.18782946  1.8696444  -0.714289    0.4130642  -0.15253948  0.04026256]
🌟 IK 求解耗时: 48.26 ms
[a1x_robot] IK Solution Found - joints: [ 0.18782946  1.8696444  -0.714289    0.4130642  -0.15253948  0.04026256], max joint diff: 0.0491 rad (2.81°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 49, terminate: False
last return: -1.0:   0%|                | 637/992999 [02:12<30:20:30,  9.09it/s][SpaceMouse] Intervention detected: using expert action, 0.0
Raw action: [-0.    0.01 -0.01  0.04 -0.   -0.    0.  ], Scaled action: [-0.    0.01 -0.01  0.04 -0.   -0.    0.  ]
EEF delta: pos=[-0.    0.01 -0.01], rot=[ 0.04 -0.   -0.  ], gripper: 0.608 -> 0.608 (60.8mm)
[a1x_robot] Action to be Solved - pos: [0.22809897 0.02879897 0.10744597], quat[x,y,z,w]: [-0.09740893  0.69411393  0.00440454  0.71323065]
I0213 19:25:58.868212 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.868404 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18608356  1.8730055  -0.7055893   0.39138156 -0.13384417  0.04084446]
Best IK solution: [ 0.18608356  1.8730055  -0.7055893   0.39138156 -0.13384417  0.04084446]
🌟 IK 求解耗时: 50.25 ms
[a1x_robot] IK Solution Found - joints: [ 0.18608356  1.8730055  -0.7055893   0.39138156 -0.13384417  0.04084446], max joint diff: 0.0539 rad (3.09°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 50, terminate: False
last return: -1.0:   0%|                | 638/992999 [02:12<30:14:37,  9.11it/s]Raw action: [ 0.01        0.00330767 -0.00529616  0.02175616 -0.01725901  0.01222276
 -0.07631578], Scaled action: [ 0.01        0.00330767 -0.00529616  0.02175616 -0.01725901  0.01222276
 -0.07631578]
EEF delta: pos=[ 0.01        0.00330767 -0.00529616], rot=[ 0.02175616 -0.01725901  0.01222276], gripper: 0.608 -> 0.532 (53.2mm)
[a1x_robot] Action to be Solved - pos: [0.23813396 0.02501397 0.11018435], quat[x,y,z,w]: [-1.06141766e-01  6.86580711e-01  2.18653134e-04  7.19264072e-01]
I0213 19:25:58.977043 137493517162304 logger.py:71] Updating safety params
I0213 19:25:58.977221 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.17128372  1.8826869  -0.73850244  0.3946469  -0.15366162  0.01779857]
Best IK solution: [ 0.17128372  1.8826869  -0.73850244  0.3946469  -0.15366162  0.01779857]
🌟 IK 求解耗时: 50.44 ms
[a1x_robot] IK Solution Found - joints: [ 0.17128372  1.8826869  -0.73850244  0.3946469  -0.15366162  0.01779857], max joint diff: 0.0402 rad (2.31°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 51, terminate: False
last return: -1.0:   0%|                | 639/992999 [02:12<30:09:50,  9.14it/s]Raw action: [ 0.00850416  0.00476706 -0.00019699  0.01836629 -0.02736071  0.0049532
 -0.06609821], Scaled action: [ 0.00850416  0.00476706 -0.00019699  0.01836629 -0.02736071  0.0049532
 -0.06609821]
EEF delta: pos=[ 0.00850416  0.00476706 -0.00019699], rot=[ 0.01836629 -0.02736071  0.0049532 ], gripper: 0.600 -> 0.533 (53.3mm)
[a1x_robot] Action to be Solved - pos: [0.23639596 0.02879757 0.11337666], quat[x,y,z,w]: [-0.10490009  0.68359183 -0.00238258  0.72228284]
I0213 19:25:59.085803 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.085989 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18974921  1.8690002  -0.7380059   0.4028215  -0.15719831  0.04125205]
Best IK solution: [ 0.18974921  1.8690002  -0.7380059   0.4028215  -0.15719831  0.04125205]
🌟 IK 求解耗时: 51.19 ms
[a1x_robot] IK Solution Found - joints: [ 0.18974921  1.8690002  -0.7380059   0.4028215  -0.15719831  0.04125205], max joint diff: 0.0242 rad (1.39°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 52, terminate: False
last return: -1.0:   0%|                | 640/992999 [02:12<37:43:41,  7.31it/s]Raw action: [ 0.01        0.00129084  0.00046984  0.01318542 -0.038171    0.00173826
 -0.07195544], Scaled action: [ 0.01        0.00129084  0.00046984  0.01318542 -0.038171    0.00173826
 -0.07195544]
EEF delta: pos=[0.01       0.00129084 0.00046984], rot=[ 0.01318542 -0.038171    0.00173826], gripper: 0.578 -> 0.506 (50.6mm)
[a1x_robot] Action to be Solved - pos: [0.24027534 0.02957782 0.11103835], quat[x,y,z,w]: [-0.09638754  0.67571979  0.00567317  0.73080779]
I0213 19:25:59.287790 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.288034 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.1808466   1.8721801  -0.72719383  0.36179516 -0.13599692  0.03161119]
Best IK solution: [ 0.1808466   1.8721801  -0.72719383  0.36179516 -0.13599692  0.03161119]
🌟 IK 求解耗时: 49.33 ms
[a1x_robot] IK Solution Found - joints: [ 0.1808466   1.8721801  -0.72719383  0.36179516 -0.13599692  0.03161119], max joint diff: 0.0350 rad (2.01°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 53, terminate: False
last return: -1.0:   0%|                | 641/992999 [02:12<35:26:35,  7.78it/s]Raw action: [ 0.00720268 -0.00026953  0.00258177  0.00747526 -0.02020862  0.00276121
 -0.08858001], Scaled action: [ 0.00720268 -0.00026953  0.00258177  0.00747526 -0.02020862  0.00276121
 -0.08858001]
EEF delta: pos=[ 0.00720268 -0.00026953  0.00258177], rot=[ 0.00747526 -0.02020862  0.00276121], gripper: 0.561 -> 0.472 (47.2mm)
[a1x_robot] Action to be Solved - pos: [0.23888044 0.02797141 0.112481  ], quat[x,y,z,w]: [-0.09782426  0.68119564  0.00717838  0.72550078]
I0213 19:25:59.396427 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.396672 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.17439944  1.8709835  -0.7366559   0.3865637  -0.13385637  0.02204411]
Best IK solution: [ 0.17439944  1.8709835  -0.7366559   0.3865637  -0.13385637  0.02204411]
🌟 IK 求解耗时: 46.44 ms
[a1x_robot] IK Solution Found - joints: [ 0.17439944  1.8709835  -0.7366559   0.3865637  -0.13385637  0.02204411], max joint diff: 0.0209 rad (1.20°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 54, terminate: False
last return: -1.0:   0%|                | 642/992999 [02:12<33:49:21,  8.15it/s]Raw action: [ 0.00673936 -0.00129402  0.00186631  0.00667567 -0.01022541  0.00778757
 -0.06400689], Scaled action: [ 0.00673936 -0.00129402  0.00186631  0.00667567 -0.01022541  0.00778757
 -0.06400689]
EEF delta: pos=[ 0.00673936 -0.00129402  0.00186631], rot=[ 0.00667567 -0.01022541  0.00778757], gripper: 0.533 -> 0.469 (46.9mm)
[a1x_robot] Action to be Solved - pos: [0.24071161 0.02637013 0.11310959], quat[x,y,z,w]: [-0.10242775  0.68229188  0.00419079  0.72385688]
I0213 19:25:59.506376 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.506570 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.17086592  1.8761882  -0.7487641   0.39769995 -0.1439738   0.01622585]
Best IK solution: [ 0.17086592  1.8761882  -0.7487641   0.39769995 -0.1439738   0.01622585]
🌟 IK 求解耗时: 51.68 ms
[a1x_robot] IK Solution Found - joints: [ 0.17086592  1.8761882  -0.7487641   0.39769995 -0.1439738   0.01622585], max joint diff: 0.0234 rad (1.34°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 55, terminate: False
last return: -1.0:   0%|                | 643/992999 [02:12<32:43:34,  8.42it/s]Raw action: [ 0.00712534  0.00364301  0.00439642  0.00842945 -0.02720885  0.01047965
 -0.06543357], Scaled action: [ 0.00712534  0.00364301  0.00439642  0.00842945 -0.02720885  0.01047965
 -0.06543357]
EEF delta: pos=[0.00712534 0.00364301 0.00439642], rot=[ 0.00842945 -0.02720885  0.01047965], gripper: 0.516 -> 0.450 (45.0mm)
[a1x_robot] Action to be Solved - pos: [0.2429414  0.03191317 0.11692534], quat[x,y,z,w]: [-0.10442749  0.6738495   0.00343791  0.73144373]
I0213 19:25:59.614897 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.615086 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.19480509  1.8661513  -0.75688356  0.39691007 -0.15148892  0.03669063]
Best IK solution: [ 0.19480509  1.8661513  -0.75688356  0.39691007 -0.15148892  0.03669063]
🌟 IK 求解耗时: 56.77 ms
[a1x_robot] IK Solution Found - joints: [ 0.19480509  1.8661513  -0.75688356  0.39691007 -0.15148892  0.03669063], max joint diff: 0.0233 rad (1.33°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 56, terminate: False
last return: -1.0:   0%|                | 644/992999 [02:12<31:57:59,  8.62it/s]Raw action: [ 0.00663713  0.00466191  0.00467077  0.01005475 -0.02893408  0.01138823
 -0.05801458], Scaled action: [ 0.00663713  0.00466191  0.00467077  0.01005475 -0.02893408  0.01138823
 -0.05801458]
EEF delta: pos=[0.00663713 0.00466191 0.00467077], rot=[ 0.01005475 -0.02893408  0.01138823], gripper: 0.497 -> 0.439 (43.9mm)
[a1x_robot] Action to be Solved - pos: [0.24395439 0.03367365 0.11737535], quat[x,y,z,w]: [-0.10201272  0.67115444  0.00937853  0.73420513]
I0213 19:25:59.725506 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.725749 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.19646542  1.8639513  -0.75678736  0.3898516  -0.14057139  0.0330758 ]
Best IK solution: [ 0.19646542  1.8639513  -0.75678736  0.3898516  -0.14057139  0.0330758 ]
🌟 IK 求解耗时: 51.05 ms
[a1x_robot] IK Solution Found - joints: [ 0.19646542  1.8639513  -0.75678736  0.3898516  -0.14057139  0.0330758 ], max joint diff: 0.0213 rad (1.22°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 57, terminate: False
last return: -1.0:   0%|                | 645/992999 [02:13<31:28:48,  8.76it/s][SpaceMouse] Intervention detected: using expert action, -0.16268604937681302
Raw action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16268605], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16268605]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.475 -> 0.313 (31.3mm)
[a1x_robot] Action to be Solved - pos: [0.23855778 0.02914308 0.11283919], quat[x,y,z,w]: [-0.09884987  0.68148846  0.00769963  0.7250813 ]
I0213 19:25:59.834280 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.834475 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.179784    1.870425   -0.7382899   0.39041793 -0.13467148  0.02531513]
Best IK solution: [ 0.179784    1.870425   -0.7382899   0.39041793 -0.13467148  0.02531513]
🌟 IK 求解耗时: 51.63 ms
[a1x_robot] IK Solution Found - joints: [ 0.179784    1.870425   -0.7382899   0.39041793 -0.13467148  0.02531513], max joint diff: 0.0008 rad (0.05°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 58, terminate: False
last return: -1.0:   0%|                | 646/992999 [02:13<31:04:52,  8.87it/s][SpaceMouse] Intervention detected: using expert action, -0.166172459023231
Raw action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16617246], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16617246]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.449 -> 0.282 (28.2mm)
[a1x_robot] Action to be Solved - pos: [0.24028707 0.02875699 0.11375271], quat[x,y,z,w]: [-0.09888362  0.67954978  0.00584876  0.72691122]
I0213 19:25:59.944763 137493517162304 logger.py:71] Updating safety params
I0213 19:25:59.944944 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.17829788  1.8704256  -0.7440425   0.39106384 -0.13787234  0.02595745]
Best IK solution: [ 0.17829788  1.8704256  -0.7440425   0.39106384 -0.13787234  0.02595745]
🌟 IK 求解耗时: 46.73 ms
[a1x_robot] IK Solution Found - joints: [ 0.17829788  1.8704256  -0.7440425   0.39106384 -0.13787234  0.02595745], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 59, terminate: False
last return: -1.0:   0%|                | 647/992999 [02:13<30:53:33,  8.92it/s][SpaceMouse] Intervention detected: using expert action, -0.1647004507610254
Raw action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16470045], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16470045]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.391 -> 0.227 (22.7mm)
[a1x_robot] Action to be Solved - pos: [0.24228219 0.03117725 0.1151584 ], quat[x,y,z,w]: [-0.10083845  0.67554809  0.00678835  0.73035628]
I0213 19:26:00.054636 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.054830 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18787234  1.868936   -0.7504255   0.38957447 -0.14085107  0.0306383 ]
Best IK solution: [ 0.18787234  1.868936   -0.7504255   0.38957447 -0.14085107  0.0306383 ]
🌟 IK 求解耗时: 46.00 ms
[a1x_robot] IK Solution Found - joints: [ 0.18787234  1.868936   -0.7504255   0.38957447 -0.14085107  0.0306383 ], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 60, terminate: False
last return: -1.0:   0%|                | 648/992999 [02:13<30:43:17,  8.97it/s][SpaceMouse] Intervention detected: using expert action, -0.16586087576692743
Raw action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16586088], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16586088]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.346 -> 0.180 (18.0mm)
[a1x_robot] Action to be Solved - pos: [0.24297003 0.03262224 0.11561028], quat[x,y,z,w]: [-0.10193178  0.67413352  0.00959916  0.73147916]
I0213 19:26:00.166927 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.167119 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.19234042  1.8685104  -0.75276595  0.38872334 -0.13893618  0.0293617 ]
Best IK solution: [ 0.19234042  1.8685104  -0.75276595  0.38872334 -0.13893618  0.0293617 ]
🌟 IK 求解耗时: 59.25 ms
[a1x_robot] IK Solution Found - joints: [ 0.19234042  1.8685104  -0.75276595  0.38872334 -0.13893618  0.0293617 ], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 61, terminate: False
last return: -1.0:   0%|                | 649/992999 [02:13<30:42:12,  8.98it/s][SpaceMouse] Intervention detected: using expert action, -0.1686320560686164
Raw action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16863206], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
 -0.16863206]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.320 -> 0.151 (15.1mm)
[a1x_robot] Action to be Solved - pos: [0.24212109 0.03103718 0.11506327], quat[x,y,z,w]: [-0.10048732  0.67572568  0.00840256  0.7302236 ]
I0213 19:26:00.275302 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.275490 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.1861702   1.8685104  -0.74936163  0.38872334 -0.13787234  0.02723404]
Best IK solution: [ 0.1861702   1.8685104  -0.74936163  0.38872334 -0.13787234  0.02723404]
🌟 IK 求解耗时: 51.31 ms
[a1x_robot] IK Solution Found - joints: [ 0.1861702   1.8685104  -0.74936163  0.38872334 -0.13787234  0.02723404], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 62, terminate: False
last return: -1.0:   0%|                | 650/992999 [02:13<31:23:20,  8.78it/s][SpaceMouse] Intervention detected: using expert action, -0.1692210984227252
Raw action: [-0.         0.         0.        -0.        -0.        -0.
 -0.1692211], Scaled action: [-0.         0.         0.        -0.        -0.        -0.
 -0.1692211]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.253 -> 0.084 (8.4mm)
[a1x_robot] Action to be Solved - pos: [0.24132181 0.02955085 0.11457102], quat[x,y,z,w]: [-0.09896114  0.67712518  0.00634408  0.72915563]
I0213 19:26:00.395013 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.395206 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18085106  1.8685104  -0.7461702   0.38872334 -0.13808511  0.02723404]
Best IK solution: [ 0.18085106  1.8685104  -0.7461702   0.38872334 -0.13808511  0.02723404]
🌟 IK 求解耗时: 52.19 ms
[a1x_robot] IK Solution Found - joints: [ 0.18085106  1.8685104  -0.7461702   0.38872334 -0.13808511  0.02723404], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 63, terminate: False
last return: -1.0:   0%|                | 651/992999 [02:13<30:59:03,  8.90it/s]Raw action: [ 0.00704317  0.00026788  0.008427    0.00431651 -0.02910067  0.00065677
 -0.0081375 ], Scaled action: [ 0.00704317  0.00026788  0.008427    0.00431651 -0.02910067  0.00065677
 -0.0081375 ]
EEF delta: pos=[0.00704317 0.00026788 0.008427  ], rot=[ 0.00431651 -0.02910067  0.00065677], gripper: 0.205 -> 0.197 (19.7mm)
[a1x_robot] Action to be Solved - pos: [0.2490701  0.03110679 0.12331183], quat[x,y,z,w]: [-0.09870672  0.66492081  0.00800471  0.74031968]
I0213 19:26:00.504073 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.504261 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18168357  1.8544421  -0.7801724   0.4033345  -0.13878027  0.02360902]
Best IK solution: [ 0.18168357  1.8544421  -0.7801724   0.4033345  -0.13878027  0.02360902]
🌟 IK 求解耗时: 46.89 ms
[a1x_robot] IK Solution Found - joints: [ 0.18168357  1.8544421  -0.7801724   0.4033345  -0.13878027  0.02360902], max joint diff: 0.0321 rad (1.84°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 64, terminate: False
last return: -1.0:   0%|                | 652/992999 [02:13<30:42:24,  8.98it/s]Raw action: [ 0.00558136  0.00051361  0.00590991  0.00503314 -0.02422898  0.00807523
  0.00144793], Scaled action: [ 0.00558136  0.00051361  0.00590991  0.00503314 -0.02422898  0.00807523
  0.00144793]
EEF delta: pos=[0.00558136 0.00051361 0.00590991], rot=[ 0.00503314 -0.02422898  0.00807523], gripper: 0.174 -> 0.176 (17.6mm)
[a1x_robot] Action to be Solved - pos: [0.24830481 0.03245802 0.1211643 ], quat[x,y,z,w]: [-0.10210009  0.66507475  0.01287642  0.73965218]
I0213 19:26:00.613155 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.613358 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18661101  1.8579504  -0.7715537   0.39205962 -0.13649794  0.01696432]
Best IK solution: [ 0.18661101  1.8579504  -0.7715537   0.39205962 -0.13649794  0.01696432]
🌟 IK 求解耗时: 51.33 ms
[a1x_robot] IK Solution Found - joints: [ 0.18661101  1.8579504  -0.7715537   0.39205962 -0.13649794  0.01696432], max joint diff: 0.0211 rad (1.21°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 65, terminate: False
last return: -1.0:   0%|                | 653/992999 [02:13<30:29:46,  9.04it/s][SpaceMouse] Intervention detected: using expert action, 0.15095698599543292
Raw action: [-0.          0.          0.         -0.         -0.         -0.
  0.15095699], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
  0.15095699]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.174 -> 0.325 (32.5mm)
[a1x_robot] Action to be Solved - pos: [0.24268317 0.0307119  0.11514856], quat[x,y,z,w]: [-0.09945781  0.67492905  0.00781929  0.73110723]
I0213 19:26:00.721931 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.722122 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18425532  1.8685104  -0.7497872   0.38659576 -0.1374468   0.0274468 ]
Best IK solution: [ 0.18425532  1.8685104  -0.7497872   0.38659576 -0.1374468   0.0274468 ]
🌟 IK 求解耗时: 44.93 ms
[a1x_robot] IK Solution Found - joints: [ 0.18425532  1.8685104  -0.7497872   0.38659576 -0.1374468   0.0274468 ], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 66, terminate: False
last return: -1.0:   0%|                | 654/992999 [02:14<30:21:22,  9.08it/s][SpaceMouse] Intervention detected: using expert action, 0.18692737993039418
Raw action: [-0.          0.          0.         -0.         -0.         -0.
  0.18692738], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
  0.18692738]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.177 -> 0.364 (36.4mm)
[a1x_robot] Action to be Solved - pos: [0.24183968 0.03002821 0.11507049], quat[x,y,z,w]: [-0.09935477  0.67633222  0.00735575  0.72982823]
I0213 19:26:00.831553 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.831746 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18212765  1.8678724  -0.74851066  0.38957447 -0.1374468   0.02638297]
Best IK solution: [ 0.18212765  1.8678724  -0.74851066  0.38957447 -0.1374468   0.02638297]
🌟 IK 求解耗时: 47.79 ms
[a1x_robot] IK Solution Found - joints: [ 0.18212765  1.8678724  -0.74851066  0.38957447 -0.1374468   0.02638297], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 67, terminate: False
last return: -1.0:   0%|                | 655/992999 [02:14<30:15:00,  9.11it/s][SpaceMouse] Intervention detected: using expert action, 0.17020551274371637
Raw action: [-0.          0.          0.         -0.         -0.         -0.
  0.17020551], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
  0.17020551]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.205 -> 0.376 (37.6mm)
[a1x_robot] Action to be Solved - pos: [0.24517417 0.03078497 0.1175357 ], quat[x,y,z,w]: [-0.10025374  0.67151386  0.0096971   0.73411463]
I0213 19:26:00.940259 137493517162304 logger.py:71] Updating safety params
I0213 19:26:00.940448 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18234043  1.8651063  -0.7599998   0.39042553 -0.13659574  0.02106383]
Best IK solution: [ 0.18234043  1.8651063  -0.7599998   0.39042553 -0.13659574  0.02106383]
🌟 IK 求解耗时: 53.87 ms
[a1x_robot] IK Solution Found - joints: [ 0.18234043  1.8651063  -0.7599998   0.39042553 -0.13659574  0.02106383], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 68, terminate: False
last return: -1.0:   0%|                | 656/992999 [02:14<30:13:50,  9.12it/s][SpaceMouse] Intervention detected: using expert action, 0.193188212250961
Raw action: [-0.          0.          0.         -0.         -0.         -0.
  0.19318821], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
  0.19318821]
EEF delta: pos=[-0.  0.  0.], rot=[-0. -0. -0.], gripper: 0.241 -> 0.434 (43.4mm)
[a1x_robot] Action to be Solved - pos: [0.24766566 0.03150474 0.11898711], quat[x,y,z,w]: [-0.10031276  0.66778386  0.01078291  0.73748613]
I0213 19:26:01.050265 137493517162304 logger.py:71] Updating safety params
I0213 19:26:01.050449 137493517162304 logger.py:71] Updating optimizer params
prev_q: [ 0.18319148  1.8640424  -0.767234    0.38851056 -0.13617021  0.01957447]
Best IK solution: [ 0.18319148  1.8640424  -0.767234    0.38851056 -0.13617021  0.01957447]
🌟 IK 求解耗时: 47.44 ms
[a1x_robot] IK Solution Found - joints: [ 0.18319148  1.8640424  -0.767234    0.38851056 -0.13617021  0.01957447], max joint diff: 0.0000 rad (0.00°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 69, terminate: False
last return: -1.0:   0%|                | 657/992999 [02:14<30:16:00,  9.11it/s][SpaceMouse] Intervention detected: using expert action, 0.16636472132989033
Raw action: [-0.          0.          0.         -0.         -0.         -0.
  0.16636472], Scaled action: [-0.          0.          0.         -0.         -0.         -0.
  0.16636472]

