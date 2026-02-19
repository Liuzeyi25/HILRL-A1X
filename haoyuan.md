### Todo list

/home/dungeon_master/conrft/serl_robot_infra/franka_env/envs/a1x_env.py

172 def step(self, action: np.ndarray) -> tuple: 

需要设计逻辑：action如果是delta eef，直接执行，如果是joint，也是直接执行



EEF delta: pos=[0.00492945 0.00473599 0.00253645], rot=[-0.00649106 -0.00862724  0.00851763], gripper: 0.976 -> 0.967 (96.7mm)
[a1x_robot] Action to be Solved - pos: [ 0.26716073 -0.00477589  0.19231058], quat[x,y,z,w]: [-0.0547898   0.69596588  0.01184798  0.7158835 ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:46.025240 140540777723712 logger.py:71] Updating problem kernel [n_problems: 32 , num_particles: 25 ]
I0210 10:23:46.026203 140540777723712 logger.py:71] Updating problem kernel [n_problems: 32 , num_particles: 4 ]
I0210 10:23:46.087549 140540777723712 logger.py:71] ParallelMPPI: Updating sample set
I0210 10:23:46.088744 140540777723712 logger.py:71] Updating safety params
I0210 10:23:46.088812 140540777723712 logger.py:71] Cloning math.Pose
I0210 10:23:46.145555 140540777723712 logger.py:71] Updating optimizer params
I0210 10:23:46.145925 140540777723712 logger.py:71] Cloning math.Pose
I0210 10:23:46.146389 140540777723712 logger.py:71] Cloning math.Pose
I0210 10:23:46.146672 140540777723712 logger.py:71] Solver was not initialized, warming up solver
I0210 10:23:46.278260 140540777723712 logger.py:71] Updating state_seq buffer reference (created new tensor)
I0210 10:23:46.483924 140540777723712 logger.py:71] Updating state_seq buffer reference (created new tensor)
prev_q: [ 0.00530043  1.8323073  -1.1645405   0.8723076  -0.05949825 -0.08994307]
Best IK solution: [ 0.00530043  1.8323073  -1.1645405   0.8723076  -0.05949825 -0.08994307]
🌟 IK 求解耗时: 972.87 ms
[a1x_robot] IK Solution Found - joints: [ 0.00530043  1.8323073  -1.1645405   0.8723076  -0.05949825 -0.08994307], max joint diff: 0.0254 rad (1.45°)
⏱️  ✓ 执行耗时=994ms, 误差=0.0mm
Step done: False, reward: False, path length: 1, terminate: False
 [Actor] Step 0: Transition actions shape = (7,), intervened = False
  0%|                                                                                                 | 1/1000000 [00:08<2286:04:20,  8.23s/it]EEF delta: pos=[ 6.15075417e-03  5.75981895e-03 -9.62923514e-05], rot=[-0.0027028  -0.00980463  0.00382365], gripper: 0.977 -> 0.963 (96.3mm)
[a1x_robot] Action to be Solved - pos: [ 0.26838203 -0.00375205  0.18967783], quat[x,y,z,w]: [-0.05181662  0.69565472  0.01146933  0.71641329]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:47.140630 140540777723712 logger.py:71] Updating safety params
I0210 10:23:47.140851 140540777723712 logger.py:71] Cloning JointState (breaks ref pointer)
I0210 10:23:47.140935 140540777723712 logger.py:71] Cloning JointState (breaks ref pointer)
I0210 10:23:47.141113 140540777723712 logger.py:71] Updating optimizer params
I0210 10:23:47.141332 140540777723712 logger.py:71] Cloning JointState (breaks ref pointer)
I0210 10:23:47.141426 140540777723712 logger.py:71] Cloning JointState (breaks ref pointer)
I0210 10:23:47.141889 140540777723712 logger.py:71] Cloning JointState (breaks ref pointer)
I0210 10:23:47.142026 140540777723712 logger.py:71] Cloning JointState (breaks ref pointer)
prev_q: [ 0.00771361  1.8380444  -1.1584816   0.85974276 -0.05592974 -0.08274147]
Best IK solution: [ 0.00771361  1.8380444  -1.1584816   0.85974276 -0.05592974 -0.08274147]
🌟 IK 求解耗时: 232.60 ms
[a1x_robot] IK Solution Found - joints: [ 0.00771361  1.8380444  -1.1584816   0.85974276 -0.05592974 -0.08274147], max joint diff: 0.0230 rad (1.32°)
⏱️  ✓ 执行耗时=234ms, 误差=0.0mm
Step done: False, reward: False, path length: 2, terminate: False
  0%|                                                                                                  | 2/1000000 [00:08<993:26:07,  3.58s/it]EEF delta: pos=[ 0.00839717 -0.00250675  0.00343198], rot=[-0.0083305   0.00031428  0.00115932], gripper: 0.977 -> 0.972 (97.2mm)
[a1x_robot] Action to be Solved - pos: [ 0.2754179  -0.00813084  0.19616001], quat[x,y,z,w]: [-0.05631758  0.69432354  0.00773941  0.71741428]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:47.370850 140540777723712 logger.py:71] Updating safety params
I0210 10:23:47.371148 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.00420243  1.852978   -1.2146397   0.8963653  -0.06699817 -0.09609465]
Best IK solution: [-0.00420243  1.852978   -1.2146397   0.8963653  -0.06699817 -0.09609465]
🌟 IK 求解耗时: 135.62 ms
[a1x_robot] IK Solution Found - joints: [-0.00420243  1.852978   -1.2146397   0.8963653  -0.06699817 -0.09609465], max joint diff: 0.0534 rad (3.06°)
⏱️  ✓ 执行耗时=137ms, 误差=0.0mm
Step done: False, reward: False, path length: 3, terminate: False
  0%|                                                                                                  | 3/1000000 [00:08<563:52:14,  2.03s/it]EEF delta: pos=[0.0068374  0.00401212 0.00468715], rot=[-3.09727946e-03 -5.14730625e-03  6.68021967e-05], gripper: 0.977 -> 0.973 (97.3mm)
[a1x_robot] Action to be Solved - pos: [ 2.76213367e-01 -5.90366725e-05  1.96262735e-01], quat[x,y,z,w]: [-0.0523332   0.68966308  0.00916884  0.72217865]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:47.561771 140540777723712 logger.py:71] Updating safety params
I0210 10:23:47.562099 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.02241988  1.8484654  -1.2084382   0.88343483 -0.06004041 -0.06607687]
Best IK solution: [ 0.02241988  1.8484654  -1.2084382   0.88343483 -0.06004041 -0.06607687]
🌟 IK 求解耗时: 136.68 ms
[a1x_robot] IK Solution Found - joints: [ 0.02241988  1.8484654  -1.2084382   0.88343483 -0.06004041 -0.06607687], max joint diff: 0.0476 rad (2.73°)
⏱️  ✓ 执行耗时=138ms, 误差=0.0mm
Step done: False, reward: False, path length: 4, terminate: False
  0%|                                                                                                  | 4/1000000 [00:08<362:12:44,  1.30s/it]EEF delta: pos=[ 0.00927373  0.00921972 -0.00019252], rot=[-0.00612505 -0.00503737  0.007537  ], gripper: 0.977 -> 0.978 (97.8mm)
[a1x_robot] Action to be Solved - pos: [0.27902914 0.00363974 0.19342952], quat[x,y,z,w]: [-0.06041298  0.69400173  0.01058905  0.71735608]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:47.750754 140540777723712 logger.py:71] Updating safety params
I0210 10:23:47.751103 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.03918428  1.8678153  -1.2215496   0.8906559  -0.07009394 -0.06253599]
Best IK solution: [ 0.03918428  1.8678153  -1.2215496   0.8906559  -0.07009394 -0.06253599]
🌟 IK 求解耗时: 132.59 ms
[a1x_robot] IK Solution Found - joints: [ 0.03918428  1.8678153  -1.2215496   0.8906559  -0.07009394 -0.06253599], max joint diff: 0.0396 rad (2.27°)
⏱️  ✓ 执行耗时=134ms, 误差=0.0mm
Step done: False, reward: False, path length: 5, terminate: False
  0%|                                                                                                  | 5/1000000 [00:09<250:10:30,  1.11it/s]EEF delta: pos=[0.00318512 0.01066273 0.00327388], rot=[ 0.00050237 -0.00667623  0.00143153], gripper: 0.977 -> 0.975 (97.5mm)
[a1x_robot] Action to be Solved - pos: [0.27902176 0.00625718 0.19977081], quat[x,y,z,w]: [-0.0497201   0.68709524  0.00538578  0.72484415]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:47.936521 140540777723712 logger.py:71] Updating safety params
I0210 10:23:47.936810 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04593813  1.8518697  -1.231887    0.8977751  -0.06304536 -0.03375916]
Best IK solution: [ 0.04593813  1.8518697  -1.231887    0.8977751  -0.06304536 -0.03375916]
🌟 IK 求解耗时: 100.19 ms
[a1x_robot] IK Solution Found - joints: [ 0.04593813  1.8518697  -1.231887    0.8977751  -0.06304536 -0.03375916], max joint diff: 0.0387 rad (2.22°)
⏱️  ✓ 执行耗时=102ms, 误差=0.0mm
Step done: False, reward: False, path length: 6, terminate: False
  0%|                                                                                                  | 6/1000000 [00:09<179:36:40,  1.55it/s]EEF delta: pos=[0.00947822 0.00333289 0.00395598], rot=[-0.00424999 -0.01127741  0.00889789], gripper: 0.977 -> 0.966 (96.6mm)
[a1x_robot] Action to be Solved - pos: [0.286338   0.00226961 0.20044468], quat[x,y,z,w]: [-0.0565956   0.68438248  0.00786998  0.72688075]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:48.089622 140540777723712 logger.py:71] Updating safety params
I0210 10:23:48.090002 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.03268118  1.8703767  -1.2618883   0.9012018  -0.06810117 -0.0606948 ]
Best IK solution: [ 0.03268118  1.8703767  -1.2618883   0.9012018  -0.06810117 -0.0606948 ]
🌟 IK 求解耗时: 114.56 ms
[a1x_robot] IK Solution Found - joints: [ 0.03268118  1.8703767  -1.2618883   0.9012018  -0.06810117 -0.0606948 ], max joint diff: 0.0521 rad (2.99°)
⏱️  ✓ 执行耗时=116ms, 误差=0.0mm
Step done: False, reward: False, path length: 7, terminate: False
  0%|                                                                                                  | 7/1000000 [00:09<136:04:38,  2.04it/s]EEF delta: pos=[0.01075279 0.0037481  0.00449436], rot=[ 0.00042261 -0.00962658  0.00579511], gripper: 0.977 -> 0.963 (96.3mm)
[a1x_robot] Action to be Solved - pos: [0.28974987 0.00674591 0.19968796], quat[x,y,z,w]: [-0.06038875  0.686734    0.0115071   0.72430463]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:48.258059 140540777723712 logger.py:71] Updating safety params
I0210 10:23:48.258443 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04804397  1.8868977  -1.2828114   0.9132783  -0.06896316 -0.05561659]
Best IK solution: [ 0.04804397  1.8868977  -1.2828114   0.9132783  -0.06896316 -0.05561659]
🌟 IK 求解耗时: 107.59 ms
[a1x_robot] IK Solution Found - joints: [ 0.04804397  1.8868977  -1.2828114   0.9132783  -0.06896316 -0.05561659], max joint diff: 0.0632 rad (3.62°)
⏱️  ✓ 执行耗时=109ms, 误差=0.0mm
Step done: False, reward: False, path length: 8, terminate: False
  0%|                                                                                                  | 8/1000000 [00:09<107:02:33,  2.59it/s]EEF delta: pos=[ 0.0089962  -0.00394639  0.00585873], rot=[-0.00518101 -0.00335822 -0.00194792], gripper: 0.977 -> 0.971 (97.1mm)
[a1x_robot] Action to be Solved - pos: [0.29032389 0.00098755 0.20446244], quat[x,y,z,w]: [-0.05614989  0.6860224   0.00603031  0.72538547]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:48.419193 140540777723712 logger.py:71] Updating safety params
I0210 10:23:48.419398 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.02849371  1.8835287  -1.3030612   0.93345106 -0.06999025 -0.06154813]
Best IK solution: [ 0.02849371  1.8835287  -1.3030612   0.93345106 -0.06999025 -0.06154813]
🌟 IK 求解耗时: 100.55 ms
[a1x_robot] IK Solution Found - joints: [ 0.02849371  1.8835287  -1.3030612   0.93345106 -0.06999025 -0.06154813], max joint diff: 0.0654 rad (3.75°)
⏱️  ✓ 执行耗时=102ms, 误差=0.0mm
Step done: False, reward: False, path length: 9, terminate: False
  0%|                                                                                                   | 9/1000000 [00:09<86:59:33,  3.19it/s]EEF delta: pos=[0.01167697 0.00086547 0.00566349], rot=[ 0.0006757  -0.01552239 -0.0078697 ], gripper: 0.973 -> 0.964 (96.4mm)
[a1x_robot] Action to be Solved - pos: [0.29762213 0.00355731 0.20659917], quat[x,y,z,w]: [-0.05346099  0.67730171  0.00483354  0.73374447]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:48.573969 140540777723712 logger.py:71] Updating safety params
I0210 10:23:48.574350 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.03583425  1.8925185  -1.3264033   0.92431366 -0.06828985 -0.04948819]
Best IK solution: [ 0.03583425  1.8925185  -1.3264033   0.92431366 -0.06828985 -0.04948819]
🌟 IK 求解耗时: 106.57 ms
[a1x_robot] IK Solution Found - joints: [ 0.03583425  1.8925185  -1.3264033   0.92431366 -0.06828985 -0.04948819], max joint diff: 0.0683 rad (3.91°)
⏱️  ✓ 执行耗时=108ms, 误差=0.0mm
Step done: False, reward: False, path length: 10, terminate: False
  0%|                                                                                                  | 10/1000000 [00:09<73:51:50,  3.76it/s]EEF delta: pos=[ 0.01157989  0.0009878  -0.00060019], rot=[-0.00016624 -0.00951755 -0.00940755], gripper: 0.973 -> 0.965 (96.5mm)
[a1x_robot] Action to be Solved - pos: [0.30059555 0.00660966 0.20067133], quat[x,y,z,w]: [-0.05541082  0.68045     0.0062275   0.73067   ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:48.732803 140540777723712 logger.py:71] Updating safety params
I0210 10:23:48.733131 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04610909  1.9116     -1.3244585   0.9126593  -0.06969485 -0.0436717 ]
Best IK solution: [ 0.04610909  1.9116     -1.3244585   0.9126593  -0.06969485 -0.0436717 ]
🌟 IK 求解耗时: 98.25 ms
[a1x_robot] IK Solution Found - joints: [ 0.04610909  1.9116     -1.3244585   0.9126593  -0.06969485 -0.0436717 ], max joint diff: 0.0468 rad (2.68°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 11, terminate: False
  0%|                                                                                                  | 11/1000000 [00:10<64:04:10,  4.34it/s]EEF delta: pos=[0.00886283 0.00542747 0.00045747], rot=[-0.00310251 -0.00897584 -0.00723882], gripper: 0.973 -> 0.967 (96.7mm)
[a1x_robot] Action to be Solved - pos: [0.29935925 0.00799581 0.20438972], quat[x,y,z,w]: [-0.0546285   0.68157315  0.00262478  0.72970328]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:48.886779 140540777723712 logger.py:71] Updating safety params
I0210 10:23:48.887115 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.05253157  1.9076667  -1.3383158   0.93411076 -0.07432766 -0.03106562]
Best IK solution: [ 0.05253157  1.9076667  -1.3383158   0.93411076 -0.07432766 -0.03106562]
🌟 IK 求解耗时: 102.46 ms
[a1x_robot] IK Solution Found - joints: [ 0.05253157  1.9076667  -1.3383158   0.93411076 -0.07432766 -0.03106562], max joint diff: 0.0400 rad (2.29°)
⏱️  ✓ 执行耗时=105ms, 误差=0.0mm
Step done: False, reward: False, path length: 12, terminate: False
  0%|                                                                                                  | 12/1000000 [00:10<58:00:46,  4.79it/s]EEF delta: pos=[0.00901298 0.00383587 0.00017791], rot=[ 0.00184494 -0.01242235 -0.00591719], gripper: 0.973 -> 0.965 (96.5mm)
[a1x_robot] Action to be Solved - pos: [0.30589378 0.00772163 0.20696075], quat[x,y,z,w]: [-0.05071314  0.67183412  0.00330621  0.73895613]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:49.042748 140540777723712 logger.py:71] Updating safety params
I0210 10:23:49.043013 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04834134  1.9117764  -1.3563888   0.92095876 -0.06793291 -0.03141693]
Best IK solution: [ 0.04834134  1.9117764  -1.3563888   0.92095876 -0.06793291 -0.03141693]
🌟 IK 求解耗时: 104.45 ms
[a1x_robot] IK Solution Found - joints: [ 0.04834134  1.9117764  -1.3563888   0.92095876 -0.06793291 -0.03141693], max joint diff: 0.0355 rad (2.04°)
⏱️  ✓ 执行耗时=106ms, 误差=0.0mm
Step done: False, reward: False, path length: 13, terminate: False
  0%|                                                                                                  | 13/1000000 [00:10<53:43:16,  5.17it/s]EEF delta: pos=[0.00896255 0.00169086 0.0046512 ], rot=[ 0.00215341 -0.01373683 -0.00699074], gripper: 0.973 -> 0.964 (96.4mm)
[a1x_robot] Action to be Solved - pos: [0.30835994 0.00754634 0.20914955], quat[x,y,z,w]: [-4.94905152e-02  6.72497817e-01  4.11428927e-04  7.38442419e-01]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:49.199799 140540777723712 logger.py:71] Updating safety params
I0210 10:23:49.200138 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04826976  1.9201765  -1.3810006   0.93905234 -0.07055037 -0.0256979 ]
Best IK solution: [ 0.04826976  1.9201765  -1.3810006   0.93905234 -0.07055037 -0.0256979 ]
🌟 IK 求解耗时: 114.96 ms
[a1x_robot] IK Solution Found - joints: [ 0.04826976  1.9201765  -1.3810006   0.93905234 -0.07055037 -0.0256979 ], max joint diff: 0.0542 rad (3.10°)
⏱️  ✓ 执行耗时=117ms, 误差=0.0mm
Step done: False, reward: False, path length: 14, terminate: False
  0%|                                                                                                  | 14/1000000 [00:10<51:30:44,  5.39it/s]EEF delta: pos=[0.00761241 0.011884   0.0015126 ], rot=[-0.00025493  0.0001972   0.00538118], gripper: 0.973 -> 0.962 (96.2mm)
[a1x_robot] Action to be Solved - pos: [0.30763088 0.01916533 0.20643658], quat[x,y,z,w]: [-0.05479932  0.67893888  0.00371287  0.73213745]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:49.367842 140540777723712 logger.py:71] Updating safety params
I0210 10:23:49.368504 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.08775341  1.9332651  -1.3883982   0.9539452  -0.07563154  0.002137  ]
Best IK solution: [ 0.08775341  1.9332651  -1.3883982   0.9539452  -0.07563154  0.002137  ]
🌟 IK 求解耗时: 102.34 ms
[a1x_robot] IK Solution Found - joints: [ 0.08775341  1.9332651  -1.3883982   0.9539452  -0.07563154  0.002137  ], max joint diff: 0.0516 rad (2.96°)
⏱️  ✓ 执行耗时=104ms, 误差=0.0mm
Step done: False, reward: False, path length: 15, terminate: False
  0%|                                                                                                  | 15/1000000 [00:10<49:00:58,  5.67it/s]EEF delta: pos=[ 0.01344739  0.00282723 -0.00160347], rot=[ 0.00800735 -0.01209133 -0.00260009], gripper: 0.973 -> 0.961 (96.1mm)
[a1x_robot] Action to be Solved - pos: [0.31851321 0.01051812 0.2063296 ], quat[x,y,z,w]: [-0.04618791  0.66879965  0.00299398  0.7420005 ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:49.522857 140540777723712 logger.py:71] Updating safety params
I0210 10:23:49.523057 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.05356108  1.950093   -1.4122635   0.9304034  -0.06293367 -0.01937241]
Best IK solution: [ 0.05356108  1.950093   -1.4122635   0.9304034  -0.06293367 -0.01937241]
🌟 IK 求解耗时: 87.53 ms
[a1x_robot] IK Solution Found - joints: [ 0.05356108  1.950093   -1.4122635   0.9304034  -0.06293367 -0.01937241], max joint diff: 0.0518 rad (2.97°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 16, terminate: False
  0%|                                                                                                  | 16/1000000 [00:10<46:53:47,  5.92it/s]EEF delta: pos=[0.00948914 0.0038713  0.00452961], rot=[ 0.00485059 -0.00694498 -0.01486959], gripper: 0.973 -> 0.968 (96.8mm)
[a1x_robot] Action to be Solved - pos: [0.31768806 0.01147366 0.21417072], quat[x,y,z,w]: [-0.04143538  0.66853732 -0.00482739  0.74250768]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:49.675060 140540777723712 logger.py:71] Updating safety params
I0210 10:23:49.675514 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.05860187  1.9443135  -1.4443341   0.968169   -0.06872892  0.00319309]
Best IK solution: [ 0.05860187  1.9443135  -1.4443341   0.968169   -0.06872892  0.00319309]
🌟 IK 求解耗时: 108.52 ms
[a1x_robot] IK Solution Found - joints: [ 0.05860187  1.9443135  -1.4443341   0.968169   -0.06872892  0.00319309], max joint diff: 0.0667 rad (3.82°)
⏱️  ✓ 执行耗时=110ms, 误差=0.0mm
Step done: False, reward: False, path length: 17, terminate: False
  0%|                                                                                                  | 17/1000000 [00:11<46:19:07,  6.00it/s]EEF delta: pos=[ 0.01206009 -0.00315525  0.00353947], rot=[ 0.0005918  -0.00129084 -0.0121907 ], gripper: 0.973 -> 0.972 (97.2mm)
[a1x_robot] Action to be Solved - pos: [0.32495882 0.01194603 0.21237935], quat[x,y,z,w]: [-0.04577944  0.66810953  0.00112107  0.74265244]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:49.837198 140540777723712 logger.py:71] Updating safety params
I0210 10:23:49.837634 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.05776147  1.9692851  -1.474707    0.97216237 -0.06564861 -0.01212347]
Best IK solution: [ 0.05776147  1.9692851  -1.474707    0.97216237 -0.06564861 -0.01212347]
🌟 IK 求解耗时: 117.70 ms
[a1x_robot] IK Solution Found - joints: [ 0.05776147  1.9692851  -1.474707    0.97216237 -0.06564861 -0.01212347], max joint diff: 0.0802 rad (4.60°)
⏱️  ✓ 执行耗时=119ms, 误差=0.0mm
Step done: False, reward: False, path length: 18, terminate: False
  0%|                                                                                                  | 18/1000000 [00:11<46:41:23,  5.95it/s]EEF delta: pos=[ 0.00742559 -0.00272337  0.00517879], rot=[ 0.00353181 -0.00125128 -0.01305003], gripper: 0.973 -> 0.972 (97.2mm)
[a1x_robot] Action to be Solved - pos: [0.32376373 0.00890576 0.21448127], quat[x,y,z,w]: [-0.03905621  0.66857505 -0.00301727  0.74261222]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:50.007344 140540777723712 logger.py:71] Updating safety params
I0210 10:23:50.007751 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04733063  1.9644735  -1.4782368   0.9809824  -0.06169246 -0.00693417]
Best IK solution: [ 0.04733063  1.9644735  -1.4782368   0.9809824  -0.06169246 -0.00693417]
🌟 IK 求解耗时: 102.24 ms
[a1x_robot] IK Solution Found - joints: [ 0.04733063  1.9644735  -1.4782368   0.9809824  -0.06169246 -0.00693417], max joint diff: 0.0636 rad (3.64°)
⏱️  ✓ 执行耗时=104ms, 误差=0.0mm
Step done: False, reward: False, path length: 19, terminate: False
  0%|                                                                                                  | 19/1000000 [00:11<45:36:03,  6.09it/s]EEF delta: pos=[0.00666164 0.0056329  0.00352822], rot=[-0.00222285  0.00079266 -0.00654921], gripper: 0.973 -> 0.969 (96.9mm)
[a1x_robot] Action to be Solved - pos: [0.32404679 0.0164506  0.21690647], quat[x,y,z,w]: [-0.03971504  0.66897781 -0.00844136  0.74217259]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:50.163057 140540777723712 logger.py:71] Updating safety params
I0210 10:23:50.163370 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.07425051  1.9680862  -1.4967306   0.9993657  -0.073334    0.02625972]
Best IK solution: [ 0.07425051  1.9680862  -1.4967306   0.9993657  -0.073334    0.02625972]
🌟 IK 求解耗时: 94.54 ms
[a1x_robot] IK Solution Found - joints: [ 0.07425051  1.9680862  -1.4967306   0.9993657  -0.073334    0.02625972], max joint diff: 0.0580 rad (3.32°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 20, terminate: False
  0%|                                                                                                  | 20/1000000 [00:11<44:31:32,  6.24it/s]EEF delta: pos=[ 0.00893378 -0.00084381  0.00492682], rot=[-6.07171096e-05 -8.77704844e-03 -9.71405488e-03], gripper: 0.973 -> 0.959 (95.9mm)
[a1x_robot] Action to be Solved - pos: [0.3307487  0.00949516 0.21994417], quat[x,y,z,w]: [-0.03785153  0.66412358 -0.00521239  0.74664581]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:50.314821 140540777723712 logger.py:71] Updating safety params
I0210 10:23:50.315183 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 4.8765264e-02  1.9804835e+00 -1.5315666e+00  1.0065298e+00
 -6.3771196e-02 -1.1827903e-03]
Best IK solution: [ 4.8765264e-02  1.9804835e+00 -1.5315666e+00  1.0065298e+00
 -6.3771196e-02 -1.1827903e-03]
🌟 IK 求解耗时: 101.90 ms
[a1x_robot] IK Solution Found - joints: [ 4.8765264e-02  1.9804835e+00 -1.5315666e+00  1.0065298e+00
 -6.3771196e-02 -1.1827903e-03], max joint diff: 0.0652 rad (3.73°)
⏱️  ✓ 执行耗时=104ms, 误差=0.0mm
Step done: False, reward: False, path length: 21, terminate: False
  0%|                                                                                                  | 21/1000000 [00:11<44:08:02,  6.29it/s]EEF delta: pos=[6.26383023e-03 4.71828971e-05 7.29555590e-03], rot=[-0.00142222  0.00583985 -0.0072399 ], gripper: 0.973 -> 0.966 (96.6mm)
[a1x_robot] Action to be Solved - pos: [0.3301861  0.00975938 0.22251389], quat[x,y,z,w]: [-0.03717356  0.66868774 -0.0062234   0.74258744]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:50.470120 140540777723712 logger.py:71] Updating safety params
I0210 10:23:50.470443 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04977396  1.9862962  -1.5558633   1.0373971  -0.06418605  0.00261199]
Best IK solution: [ 0.04977396  1.9862962  -1.5558633   1.0373971  -0.06418605  0.00261199]
🌟 IK 求解耗时: 104.63 ms
[a1x_robot] IK Solution Found - joints: [ 0.04977396  1.9862962  -1.5558633   1.0373971  -0.06418605  0.00261199], max joint diff: 0.0795 rad (4.55°)
⏱️  ✓ 执行耗时=106ms, 误差=0.0mm
Step done: False, reward: False, path length: 22, terminate: False
  0%|                                                                                                  | 22/1000000 [00:11<44:05:18,  6.30it/s]EEF delta: pos=[ 0.00840855 -0.00519084  0.00519703], rot=[-0.00247611  0.01186312 -0.00918273], gripper: 0.973 -> 0.956 (95.6mm)
[a1x_robot] Action to be Solved - pos: [0.33436687 0.00811607 0.22373753], quat[x,y,z,w]: [-0.03797065  0.67062841 -0.00790514  0.74077883]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:50.635058 140540777723712 logger.py:71] Updating safety params
I0210 10:23:50.636403 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 4.5164194e-02  2.0046732e+00 -1.5929927e+00  1.0612773e+00
 -6.7159057e-02 -7.2443107e-04]
Best IK solution: [ 4.5164194e-02  2.0046732e+00 -1.5929927e+00  1.0612773e+00
 -6.7159057e-02 -7.2443107e-04]
🌟 IK 求解耗时: 120.73 ms
[a1x_robot] IK Solution Found - joints: [ 4.5164194e-02  2.0046732e+00 -1.5929927e+00  1.0612773e+00
 -6.7159057e-02 -7.2443107e-04], max joint diff: 0.0883 rad (5.06°)
⏱️  ✓ 执行耗时=124ms, 误差=0.0mm
Step done: False, reward: False, path length: 23, terminate: False
  0%|                                                                                                  | 23/1000000 [00:11<45:33:26,  6.10it/s]EEF delta: pos=[ 0.00737059 -0.00097707  0.00459956], rot=[ 0.0033818   0.00518953 -0.01025233], gripper: 0.973 -> 0.962 (96.2mm)
[a1x_robot] Action to be Solved - pos: [0.33633245 0.00892822 0.22569787], quat[x,y,z,w]: [-0.03338219  0.6678129  -0.00788536  0.74353842]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:50.804774 140540777723712 logger.py:71] Updating safety params
I0210 10:23:50.805084 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04547777  2.0067546  -1.6061972   1.0645834  -0.0612038   0.0061128 ]
Best IK solution: [ 0.04547777  2.0067546  -1.6061972   1.0645834  -0.0612038   0.0061128 ]
🌟 IK 求解耗时: 99.08 ms
[a1x_robot] IK Solution Found - joints: [ 0.04547777  2.0067546  -1.6061972   1.0645834  -0.0612038   0.0061128 ], max joint diff: 0.0741 rad (4.24°)
⏱️  ✓ 执行耗时=101ms, 误差=0.0mm
Step done: False, reward: False, path length: 24, terminate: False
  0%|                                                                                                  | 24/1000000 [00:12<44:36:04,  6.23it/s]EEF delta: pos=[0.00568572 0.00094538 0.00360625], rot=[0.00054621 0.00335438 0.00162053], gripper: 0.971 -> 0.964 (96.4mm)
[a1x_robot] Action to be Solved - pos: [0.33566771 0.00991784 0.22627106], quat[x,y,z,w]: [-0.03678959  0.66854348 -0.00618729  0.74273674]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:50.956611 140540777723712 logger.py:71] Updating safety params
I0210 10:23:50.956907 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04924605  2.0059288  -1.6079861   1.0694355  -0.06358002  0.00259407]
Best IK solution: [ 0.04924605  2.0059288  -1.6079861   1.0694355  -0.06358002  0.00259407]
🌟 IK 求解耗时: 99.42 ms
[a1x_robot] IK Solution Found - joints: [ 0.04924605  2.0059288  -1.6079861   1.0694355  -0.06358002  0.00259407], max joint diff: 0.0571 rad (3.27°)
⏱️  ✓ 执行耗时=101ms, 误差=0.0mm
Step done: False, reward: False, path length: 25, terminate: False
  0%|                                                                                                  | 25/1000000 [00:12<43:54:00,  6.33it/s]EEF delta: pos=[0.00718755 0.00119688 0.00353622], rot=[-0.00072536  0.01029947 -0.00513318], gripper: 0.970 -> 0.967 (96.7mm)
[a1x_robot] Action to be Solved - pos: [0.34019062 0.01041445 0.22924594], quat[x,y,z,w]: [-0.03342612  0.67198477 -0.00891582  0.73975649]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:51.111968 140540777723712 logger.py:71] Updating safety params
I0210 10:23:51.112309 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.0498365   2.029664   -1.6644506   1.1115587  -0.06290291  0.01215187]
Best IK solution: [ 0.0498365   2.029664   -1.6644506   1.1115587  -0.06290291  0.01215187]
🌟 IK 求解耗时: 126.44 ms
[a1x_robot] IK Solution Found - joints: [ 0.0498365   2.029664   -1.6644506   1.1115587  -0.06290291  0.01215187], max joint diff: 0.0774 rad (4.44°)
⏱️  ✓ 执行耗时=129ms, 误差=0.0mm
Step done: False, reward: False, path length: 26, terminate: False
  0%|                                                                                                  | 26/1000000 [00:12<45:55:43,  6.05it/s]EEF delta: pos=[ 0.0111896  -0.0100691   0.00652912], rot=[ 0.00666748  0.01323553 -0.00748575], gripper: 0.970 -> 0.950 (95.0mm)
[a1x_robot] Action to be Solved - pos: [ 0.3470833  -0.00047354  0.2331165 ], quat[x,y,z,w]: [-0.03036754  0.67145112 -0.00699518  0.74039332]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:51.292494 140540777723712 logger.py:71] Updating safety params
I0210 10:23:51.292844 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01440644  2.0540037  -1.7266018   1.1456395  -0.05257363 -0.0212686 ]
Best IK solution: [ 0.01440644  2.0540037  -1.7266018   1.1456395  -0.05257363 -0.0212686 ]
🌟 IK 求解耗时: 102.87 ms
[a1x_robot] IK Solution Found - joints: [ 0.01440644  2.0540037  -1.7266018   1.1456395  -0.05257363 -0.0212686 ], max joint diff: 0.1228 rad (7.03°)
⏱️  ✓ 执行耗时=104ms, 误差=0.0mm
Step done: False, reward: False, path length: 27, terminate: False
  0%|                                                                                                  | 27/1000000 [00:12<45:12:52,  6.14it/s]EEF delta: pos=[ 0.00262092 -0.00188645  0.00554257], rot=[0.00229715 0.01011308 0.00506779], gripper: 0.970 -> 0.956 (95.6mm)
[a1x_robot] Action to be Solved - pos: [0.33907731 0.00757056 0.23194957], quat[x,y,z,w]: [-0.03654086  0.67023814 -0.0037255   0.74123662]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:51.447858 140540777723712 logger.py:71] Updating safety params
I0210 10:23:51.448178 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04030551  2.0217652  -1.6649219   1.1143637  -0.05858411 -0.00910487]
Best IK solution: [ 0.04030551  2.0217652  -1.6649219   1.1143637  -0.05858411 -0.00910487]
🌟 IK 求解耗时: 104.64 ms
[a1x_robot] IK Solution Found - joints: [ 0.04030551  2.0217652  -1.6649219   1.1143637  -0.05858411 -0.00910487], max joint diff: 0.0581 rad (3.33°)
⏱️  ✓ 执行耗时=106ms, 误差=0.0mm
Step done: False, reward: False, path length: 28, terminate: False
  0%|                                                                                                  | 28/1000000 [00:12<44:49:40,  6.20it/s]EEF delta: pos=[ 0.00363135 -0.00160689  0.00486548], rot=[ 0.00481628  0.01152307 -0.0051437 ], gripper: 0.970 -> 0.951 (95.1mm)
[a1x_robot] Action to be Solved - pos: [0.34054645 0.00774589 0.2358277 ], quat[x,y,z,w]: [-0.03149896  0.67707162 -0.00516155  0.73522459]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:51.605921 140540777723712 logger.py:71] Updating safety params
I0210 10:23:51.606230 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 3.9084233e-02  2.0415199e+00 -1.7207489e+00  1.1687429e+00
 -5.3475004e-02 -3.9122978e-04]
Best IK solution: [ 3.9084233e-02  2.0415199e+00 -1.7207489e+00  1.1687429e+00
 -5.3475004e-02 -3.9122978e-04]
🌟 IK 求解耗时: 114.03 ms
[a1x_robot] IK Solution Found - joints: [ 3.9084233e-02  2.0415199e+00 -1.7207489e+00  1.1687429e+00
 -5.3475004e-02 -3.9122978e-04], max joint diff: 0.0631 rad (3.61°)
⏱️  ✓ 执行耗时=116ms, 误差=0.0mm
Step done: False, reward: False, path length: 29, terminate: False
  0%|                                                                                                  | 29/1000000 [00:12<45:16:05,  6.14it/s]EEF delta: pos=[ 0.00908468 -0.00125674  0.00541407], rot=[ 0.00795305  0.00684975 -0.00143514], gripper: 0.964 -> 0.946 (94.6mm)
[a1x_robot] Action to be Solved - pos: [ 3.54086933e-01 -2.54925877e-04  2.40973722e-01], quat[x,y,z,w]: [-0.02838127  0.67021364 -0.00400633  0.74161455]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:51.772618 140540777723712 logger.py:71] Updating safety params
I0210 10:23:51.772928 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01259273  2.0832634  -1.8194304   1.205649   -0.04527962 -0.02421754]
Best IK solution: [ 0.01259273  2.0832634  -1.8194304   1.205649   -0.04527962 -0.02421754]
🌟 IK 求解耗时: 102.94 ms
[a1x_robot] IK Solution Found - joints: [ 0.01259273  2.0832634  -1.8194304   1.205649   -0.04527962 -0.02421754], max joint diff: 0.1062 rad (6.09°)
⏱️  ✓ 执行耗时=104ms, 误差=0.0mm
Step done: False, reward: False, path length: 30, terminate: False
  0%|                                                                                                  | 30/1000000 [00:13<44:38:46,  6.22it/s]EEF delta: pos=[ 0.00711891 -0.00443602  0.00324047], rot=[0.00677821 0.01504168 0.0043217 ], gripper: 0.964 -> 0.940 (94.0mm)
[a1x_robot] Action to be Solved - pos: [0.35083758 0.00097074 0.23438423], quat[x,y,z,w]: [-0.03317911  0.67308584 -0.00152428  0.73881816]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:51.927946 140540777723712 logger.py:71] Updating safety params
I0210 10:23:51.928308 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01717911  2.072676   -1.7664572   1.171247   -0.0485466  -0.02991829]
Best IK solution: [ 0.01717911  2.072676   -1.7664572   1.171247   -0.0485466  -0.02991829]
🌟 IK 求解耗时: 108.81 ms
[a1x_robot] IK Solution Found - joints: [ 0.01717911  2.072676   -1.7664572   1.171247   -0.0485466  -0.02991829], max joint diff: 0.0860 rad (4.93°)
⏱️  ✓ 执行耗时=111ms, 误差=0.0mm
Step done: False, reward: False, path length: 31, terminate: False
  0%|                                                                                                  | 31/1000000 [00:13<44:40:43,  6.22it/s]EEF delta: pos=[ 0.00776918 -0.00382613  0.00639252], rot=[-0.00022974  0.01057699  0.00690793], gripper: 0.964 -> 0.949 (94.9mm)
[a1x_robot] Action to be Solved - pos: [0.3466605  0.00315611 0.24150638], quat[x,y,z,w]: [-0.03472911  0.68192046 -0.0015105   0.73059982]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:52.093905 140540777723712 logger.py:71] Updating safety params
I0210 10:23:52.094238 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.024513    2.0777636  -1.8150207   1.239175   -0.05129948 -0.0242816 ]
Best IK solution: [ 0.024513    2.0777636  -1.8150207   1.239175   -0.05129948 -0.0242816 ]
🌟 IK 求解耗时: 118.90 ms
[a1x_robot] IK Solution Found - joints: [ 0.024513    2.0777636  -1.8150207   1.239175   -0.05129948 -0.0242816 ], max joint diff: 0.1059 rad (6.07°)
⏱️  ✓ 执行耗时=121ms, 误差=0.0mm
Step done: False, reward: False, path length: 32, terminate: False
  0%|                                                                                                  | 32/1000000 [00:13<45:58:52,  6.04it/s]EEF delta: pos=[0.00810926 0.00144115 0.00531965], rot=[0.00967007 0.00678193 0.00013914], gripper: 0.955 -> 0.935 (93.5mm)
[a1x_robot] Action to be Solved - pos: [0.35816072 0.002324   0.24876453], quat[x,y,z,w]: [-2.63489141e-02  6.70416867e-01  8.82307837e-05  7.41516656e-01]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:52.266479 140540777723712 logger.py:71] Updating safety params
I0210 10:23:52.266827 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01722991  2.1072106  -1.904398    1.2672819  -0.03693344 -0.02208459]
Best IK solution: [ 0.01722991  2.1072106  -1.904398    1.2672819  -0.03693344 -0.02208459]
🌟 IK 求解耗时: 114.67 ms
[a1x_robot] IK Solution Found - joints: [ 0.01722991  2.1072106  -1.904398    1.2672819  -0.03693344 -0.02208459], max joint diff: 0.1076 rad (6.16°)
⏱️  ✓ 执行耗时=116ms, 误差=0.0mm
Step done: False, reward: False, path length: 33, terminate: False
  0%|                                                                                                  | 33/1000000 [00:13<46:12:51,  6.01it/s]EEF delta: pos=[ 0.01064873 -0.00092005  0.00071572], rot=[ 0.003101   -0.00075676  0.00237654], gripper: 0.955 -> 0.932 (93.2mm)
[a1x_robot] Action to be Solved - pos: [0.36168895 0.00049627 0.23834612], quat[x,y,z,w]: [-3.17124729e-02  6.71251108e-01 -2.63066300e-04  7.40551281e-01]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:52.434299 140540777723712 logger.py:71] Updating safety params
I0210 10:23:52.434771 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01414471  2.116021   -1.8648676   1.221153   -0.044369   -0.03259578]
Best IK solution: [ 0.01414471  2.116021   -1.8648676   1.221153   -0.044369   -0.03259578]
🌟 IK 求解耗时: 107.52 ms
[a1x_robot] IK Solution Found - joints: [ 0.01414471  2.116021   -1.8648676   1.221153   -0.044369   -0.03259578], max joint diff: 0.0830 rad (4.75°)
⏱️  ✓ 执行耗时=110ms, 误差=0.0mm
Step done: False, reward: False, path length: 34, terminate: False
  0%|                                                                                                  | 34/1000000 [00:13<45:45:34,  6.07it/s]EEF delta: pos=[ 0.00728884 -0.00071461  0.00479995], rot=[-0.00045061  0.01259343 -0.00240968], gripper: 0.955 -> 0.943 (94.3mm)
[a1x_robot] Action to be Solved - pos: [0.35368024 0.0019434  0.2459082 ], quat[x,y,z,w]: [-0.0320121   0.6869951  -0.00217448  0.72595332]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:52.595710 140540777723712 logger.py:71] Updating safety params
I0210 10:23:52.596022 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01970084  2.1222754  -1.9199259   1.3132144  -0.04825794 -0.02386292]
Best IK solution: [ 0.01970084  2.1222754  -1.9199259   1.3132144  -0.04825794 -0.02386292]
🌟 IK 求解耗时: 98.92 ms
[a1x_robot] IK Solution Found - joints: [ 0.01970084  2.1222754  -1.9199259   1.3132144  -0.04825794 -0.02386292], max joint diff: 0.1078 rad (6.18°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 35, terminate: False
  0%|                                                                                                  | 35/1000000 [00:13<44:45:11,  6.21it/s]EEF delta: pos=[ 0.00605864 -0.00139839  0.00343556], rot=[ 0.00810453  0.01180465 -0.00876195], gripper: 0.948 -> 0.932 (93.2mm)
[a1x_robot] Action to be Solved - pos: [0.36053383 0.00133882 0.25237565], quat[x,y,z,w]: [-2.09887848e-02  6.75442943e-01  4.40093813e-05  7.37113491e-01]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:52.749599 140540777723712 logger.py:71] Updating safety params
I0210 10:23:52.749892 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01219234  2.133199   -1.9731997   1.3234327  -0.02935618 -0.01886697]
Best IK solution: [ 0.01219234  2.133199   -1.9731997   1.3234327  -0.02935618 -0.01886697]
🌟 IK 求解耗时: 109.30 ms
[a1x_robot] IK Solution Found - joints: [ 0.01219234  2.133199   -1.9731997   1.3234327  -0.02935618 -0.01886697], max joint diff: 0.0943 rad (5.40°)
⏱️  ✓ 执行耗时=111ms, 误差=0.0mm
Step done: False, reward: False, path length: 36, terminate: False
  0%|                                                                                                  | 36/1000000 [00:14<44:58:40,  6.18it/s]EEF delta: pos=[ 0.00802795  0.00201165 -0.00168222], rot=[0.00620606 0.00552776 0.00474169], gripper: 0.948 -> 0.930 (93.0mm)
[a1x_robot] Action to be Solved - pos: [0.36930468 0.00324509 0.24212763], quat[x,y,z,w]: [-0.02993823  0.66784309  0.00285467  0.74369426]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:52.911459 140540777723712 logger.py:71] Updating safety params
I0210 10:23:52.911854 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01946387  2.1454763  -1.9388261   1.2566161  -0.03785058 -0.02903393]
Best IK solution: [ 0.01946387  2.1454763  -1.9388261   1.2566161  -0.03785058 -0.02903393]
🌟 IK 求解耗时: 109.24 ms
[a1x_robot] IK Solution Found - joints: [ 0.01946387  2.1454763  -1.9388261   1.2566161  -0.03785058 -0.02903393], max joint diff: 0.0635 rad (3.64°)
⏱️  ✓ 执行耗时=111ms, 误差=0.0mm
Step done: False, reward: False, path length: 37, terminate: False
  0%|                                                                                                  | 37/1000000 [00:14<44:58:20,  6.18it/s]EEF delta: pos=[0.00297474 0.00049496 0.00424253], rot=[-0.00042335  0.00650677 -0.00115456], gripper: 0.948 -> 0.935 (93.5mm)
[a1x_robot] Action to be Solved - pos: [0.35586568 0.00191914 0.24985785], quat[x,y,z,w]: [-0.03032658  0.68839904 -0.00265798  0.72469303]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:53.072887 140540777723712 logger.py:71] Updating safety params
I0210 10:23:53.073125 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01903347  2.1397095  -1.9738644   1.3535426  -0.04661567 -0.02132383]
Best IK solution: [ 0.01903347  2.1397095  -1.9738644   1.3535426  -0.04661567 -0.02132383]
🌟 IK 求解耗时: 103.07 ms
[a1x_robot] IK Solution Found - joints: [ 0.01903347  2.1397095  -1.9738644   1.3535426  -0.04661567 -0.02132383], max joint diff: 0.0668 rad (3.83°)
⏱️  ✓ 执行耗时=105ms, 误差=0.0mm
Step done: False, reward: False, path length: 38, terminate: False
  0%|                                                                                                  | 38/1000000 [00:14<44:29:15,  6.24it/s]EEF delta: pos=[ 0.01092457 -0.01145892  0.01185957], rot=[-0.00604153  0.00060293  0.01095487], gripper: 0.947 -> 0.928 (92.8mm)
[a1x_robot] Action to be Solved - pos: [ 0.36951448 -0.00938316  0.26364525], quat[x,y,z,w]: [-0.02750091  0.67627129  0.00284167  0.73613366]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:53.230175 140540777723712 logger.py:71] Updating safety params
I0210 10:23:53.230727 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.0164801   2.199895   -2.1641755   1.4490958  -0.03161149 -0.06079958]
Best IK solution: [-0.0164801   2.199895   -2.1641755   1.4490958  -0.03161149 -0.06079958]
🌟 IK 求解耗时: 127.01 ms
[a1x_robot] IK Solution Found - joints: [-0.0164801   2.199895   -2.1641755   1.4490958  -0.03161149 -0.06079958], max joint diff: 0.2086 rad (11.95°)
⏱️  ✓ 执行耗时=129ms, 误差=0.0mm
Step done: False, reward: False, path length: 39, terminate: False
  0%|                                                                                                  | 39/1000000 [00:14<46:07:33,  6.02it/s]EEF delta: pos=[ 0.0127509  -0.01319036  0.00627366], rot=[0.00519248 0.00913553 0.0069669 ], gripper: 0.947 -> 0.925 (92.5mm)
[a1x_robot] Action to be Solved - pos: [ 0.38323008 -0.01125714  0.25513596], quat[x,y,z,w]: [-0.02689775  0.66150113  0.00417591  0.74945002]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:53.409615 140540777723712 logger.py:71] Updating safety params
I0210 10:23:53.409814 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.02215718  2.217289   -2.147042    1.3747559  -0.02654832 -0.06787021]
Best IK solution: [-0.02215718  2.217289   -2.147042    1.3747559  -0.02654832 -0.06787021]
🌟 IK 求解耗时: 97.02 ms
[a1x_robot] IK Solution Found - joints: [-0.02215718  2.217289   -2.147042    1.3747559  -0.02654832 -0.06787021], max joint diff: 0.1945 rad (11.14°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 40, terminate: False
  0%|                                                                                                  | 40/1000000 [00:14<44:55:45,  6.18it/s]EEF delta: pos=[ 0.00647559 -0.00465614  0.00648914], rot=[-0.0177833  -0.00085296  0.01612638], gripper: 0.939 -> 0.921 (92.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.36301196 -0.00294943  0.25517415], quat[x,y,z,w]: [-0.0420379   0.68658012 -0.00248925  0.72583356]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:53.561480 140540777723712 logger.py:71] Updating safety params
I0210 10:23:53.561884 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.00963477  2.1771493  -2.073525    1.4104267  -0.06192443 -0.04810846]
Best IK solution: [ 0.00963477  2.1771493  -2.073525    1.4104267  -0.06192443 -0.04810846]
🌟 IK 求解耗时: 92.64 ms
[a1x_robot] IK Solution Found - joints: [ 0.00963477  2.1771493  -2.073525    1.4104267  -0.06192443 -0.04810846], max joint diff: 0.1080 rad (6.19°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 41, terminate: False
  0%|                                                                                                  | 41/1000000 [00:14<44:02:50,  6.31it/s]EEF delta: pos=[0.00425952 0.00320901 0.00135048], rot=[0.00159157 0.01617116 0.0015603 ], gripper: 0.939 -> 0.943 (94.3mm)
[a1x_robot] Action to be Solved - pos: [0.36614017 0.00057584 0.25913244], quat[x,y,z,w]: [-0.03064223  0.68622664  0.00622695  0.7267154 ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:53.713481 140540777723712 logger.py:71] Updating safety params
I0210 10:23:53.713859 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01115059  2.2005405  -2.1421762   1.454618   -0.0336601  -0.04199603]
Best IK solution: [ 0.01115059  2.2005405  -2.1421762   1.454618   -0.0336601  -0.04199603]
🌟 IK 求解耗时: 99.62 ms
[a1x_robot] IK Solution Found - joints: [ 0.01115059  2.2005405  -2.1421762   1.454618   -0.0336601  -0.04199603], max joint diff: 0.0894 rad (5.12°)
⏱️  ✓ 执行耗时=102ms, 误差=0.0mm
Step done: False, reward: False, path length: 42, terminate: False
  0%|                                                                                                  | 42/1000000 [00:15<43:38:02,  6.37it/s]EEF delta: pos=[ 0.00635937 -0.00200202 -0.00112864], rot=[0.00066439 0.00656745 0.00241534], gripper: 0.937 -> 0.927 (92.7mm)
[a1x_robot] Action to be Solved - pos: [ 0.38521703 -0.01147984  0.25945485], quat[x,y,z,w]: [-0.02908258  0.6601725   0.005827    0.75052816]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:53.866486 140540777723712 logger.py:71] Updating safety params
I0210 10:23:53.867070 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.02256384  2.2346108  -2.2058933   1.4127561  -0.02674659 -0.0737844 ]
Best IK solution: [-0.02256384  2.2346108  -2.2058933   1.4127561  -0.02674659 -0.0737844 ]
🌟 IK 求解耗时: 98.67 ms
[a1x_robot] IK Solution Found - joints: [-0.02256384  2.2346108  -2.2058933   1.4127561  -0.02674659 -0.0737844 ], max joint diff: 0.0787 rad (4.51°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 43, terminate: False
  0%|                                                                                                  | 43/1000000 [00:15<43:13:33,  6.43it/s]EEF delta: pos=[0.00658785 0.00056368 0.00364781], rot=[-0.00853034 -0.00683647  0.00453307], gripper: 0.933 -> 0.921 (92.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.37460942 -0.00523644  0.25706317], quat[x,y,z,w]: [-0.03925209  0.68136168 -0.00254987  0.7308892 ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:54.016877 140540777723712 logger.py:71] Updating safety params
I0210 10:23:54.017178 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 1.9696497e-03  2.2300074e+00 -2.1882775e+00  1.4575177e+00
 -5.7391245e-02 -5.2047294e-02]
Best IK solution: [ 1.9696497e-03  2.2300074e+00 -2.1882775e+00  1.4575177e+00
 -5.7391245e-02 -5.2047294e-02]
🌟 IK 求解耗时: 103.50 ms
[a1x_robot] IK Solution Found - joints: [ 1.9696497e-03  2.2300074e+00 -2.1882775e+00  1.4575177e+00
 -5.7391245e-02 -5.2047294e-02], max joint diff: 0.0932 rad (5.34°)
⏱️  ✓ 执行耗时=105ms, 误差=0.0mm
Step done: False, reward: False, path length: 44, terminate: False
  0%|                                                                                                  | 44/1000000 [00:15<43:10:03,  6.43it/s]EEF delta: pos=[ 0.0031678  -0.00406631 -0.00047959], rot=[-0.01132658  0.01058481 -0.00038949], gripper: 0.933 -> 0.931 (93.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.36918732 -0.0054206   0.2569917 ], quat[x,y,z,w]: [-0.03341986  0.69030964  0.00136076  0.72274052]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:54.172954 140540777723712 logger.py:71] Updating safety params
I0210 10:23:54.173259 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.00224469  2.2251635  -2.1799376   1.4784651  -0.04408488 -0.05249195]
Best IK solution: [-0.00224469  2.2251635  -2.1799376   1.4784651  -0.04408488 -0.05249195]
🌟 IK 求解耗时: 102.51 ms
[a1x_robot] IK Solution Found - joints: [-0.00224469  2.2251635  -2.1799376   1.4784651  -0.04408488 -0.05249195], max joint diff: 0.0540 rad (3.09°)
⏱️  ✓ 执行耗时=104ms, 误差=0.0mm
Step done: False, reward: False, path length: 45, terminate: False
  0%|                                                                                                  | 45/1000000 [00:15<43:08:06,  6.44it/s]EEF delta: pos=[ 8.30498338e-03 -2.25330330e-03  4.59023286e-05], rot=[ 0.00032122 -0.00753024  0.00570825], gripper: 0.933 -> 0.925 (92.5mm)
[a1x_robot] Action to be Solved - pos: [ 0.38995696 -0.01044264  0.2623294 ], quat[x,y,z,w]: [-0.03265307  0.65609202  0.01041489  0.75390223]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:54.329226 140540777723712 logger.py:71] Updating safety params
I0210 10:23:54.329505 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.02028047  2.2597935  -2.2730095   1.444124   -0.02432301 -0.08305866]
Best IK solution: [-0.02028047  2.2597935  -2.2730095   1.444124   -0.02432301 -0.08305866]
🌟 IK 求解耗时: 99.46 ms
[a1x_robot] IK Solution Found - joints: [-0.02028047  2.2597935  -2.2730095   1.444124   -0.02432301 -0.08305866], max joint diff: 0.0890 rad (5.10°)
⏱️  ✓ 执行耗时=101ms, 误差=0.0mm
Step done: False, reward: False, path length: 46, terminate: False
  0%|                                                                                                  | 46/1000000 [00:15<42:58:55,  6.46it/s]EEF delta: pos=[ 0.00950743  0.00357001 -0.00086111], rot=[-0.008413   -0.00391655  0.00601269], gripper: 0.933 -> 0.926 (92.6mm)
[a1x_robot] Action to be Solved - pos: [ 0.38466533 -0.00375101  0.25848962], quat[x,y,z,w]: [-0.04059091  0.6755105  -0.00361553  0.73622339]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:54.482096 140540777723712 logger.py:71] Updating safety params
I0210 10:23:54.482478 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.00669954  2.2800276  -2.2990327   1.5025918  -0.06078656 -0.04832548]
Best IK solution: [ 0.00669954  2.2800276  -2.2990327   1.5025918  -0.06078656 -0.04832548]
🌟 IK 求解耗时: 129.91 ms
[a1x_robot] IK Solution Found - joints: [ 0.00669954  2.2800276  -2.2990327   1.5025918  -0.06078656 -0.04832548], max joint diff: 0.1082 rad (6.20°)
⏱️  ✓ 执行耗时=132ms, 误差=0.0mm
Step done: False, reward: False, path length: 47, terminate: False
  0%|                                                                                                  | 47/1000000 [00:15<45:23:29,  6.12it/s]EEF delta: pos=[ 0.00571937 -0.00035304  0.0025371 ], rot=[-0.00438812 -0.0022728  -0.00074604], gripper: 0.933 -> 0.919 (91.9mm)
[a1x_robot] Action to be Solved - pos: [ 0.3771118  -0.00663351  0.25974265], quat[x,y,z,w]: [-0.03410753  0.68534412 -0.00106357  0.7274194 ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:54.666555 140540777723712 logger.py:71] Updating safety params
I0210 10:23:54.666846 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.00431673  2.2664728  -2.2800326   1.5234342  -0.04806238 -0.05255009]
Best IK solution: [-0.00431673  2.2664728  -2.2800326   1.5234342  -0.04806238 -0.05255009]
🌟 IK 求解耗时: 100.52 ms
[a1x_robot] IK Solution Found - joints: [-0.00431673  2.2664728  -2.2800326   1.5234342  -0.04806238 -0.05255009], max joint diff: 0.0979 rad (5.61°)
⏱️  ✓ 执行耗时=102ms, 误差=0.0mm
Step done: False, reward: False, path length: 48, terminate: False
  0%|                                                                                                  | 48/1000000 [00:15<44:37:28,  6.22it/s]EEF delta: pos=[0.0073169  0.00116639 0.00226699], rot=[ 9.27642337e-04  9.48230736e-05 -3.35895061e-03], gripper: 0.933 -> 0.935 (93.5mm)
[a1x_robot] Action to be Solved - pos: [ 0.38864873 -0.00642194  0.2651665 ], quat[x,y,z,w]: [-0.03320558  0.66728481  0.00150751  0.74406055]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:54.818380 140540777723712 logger.py:71] Updating safety params
I0210 10:23:54.818737 140540777723712 logger.py:71] Updating optimizer params
prev_q: [-0.00540236  2.3036447  -2.3873894   1.5445408  -0.04149215 -0.05686491]
Best IK solution: [-0.00540236  2.3036447  -2.3873894   1.5445408  -0.04149215 -0.05686491]
🌟 IK 求解耗时: 116.03 ms
[a1x_robot] IK Solution Found - joints: [-0.00540236  2.3036447  -2.3873894   1.5445408  -0.04149215 -0.05686491], max joint diff: 0.1461 rad (8.37°)
⏱️  ✓ 执行耗时=118ms, 误差=0.0mm
Step done: False, reward: False, path length: 49, terminate: False
  0%|                                                                                                  | 49/1000000 [00:16<45:13:36,  6.14it/s]EEF delta: pos=[0.00635701 0.004479   0.00128539], rot=[-0.00056051  0.00018862 -0.00579713], gripper: 0.933 -> 0.926 (92.6mm)
[a1x_robot] Action to be Solved - pos: [0.38815491 0.00093507 0.2634689 ], quat[x,y,z,w]: [-0.03681622  0.67361365 -0.00409106  0.73815478]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:54.987560 140540777723712 logger.py:71] Updating safety params
I0210 10:23:54.987874 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01772523  2.318635   -2.410916    1.5708     -0.05736198 -0.03134061]
Best IK solution: [ 0.01772523  2.318635   -2.410916    1.5708     -0.05736198 -0.03134061]
🌟 IK 求解耗时: 107.98 ms
[a1x_robot] IK Solution Found - joints: [ 0.01772523  2.318635   -2.410916    1.5708     -0.05736198 -0.03134061], max joint diff: 0.1247 rad (7.15°)
⏱️  ✓ 执行耗时=110ms, 误差=0.0mm
Step done: False, reward: False, path length: 50, terminate: False
  0%|                                                                                                  | 50/1000000 [00:16<45:04:56,  6.16it/s]EEF delta: pos=[0.00637412 0.00286707 0.00095027], rot=[-0.00687174 -0.00617139  0.00398918], gripper: 0.933 -> 0.921 (92.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.3852468  -0.00363509  0.26130885], quat[x,y,z,w]: [-0.0379052   0.678926   -0.00261214  0.73322292]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:55.149903 140540777723712 logger.py:71] Updating safety params
I0210 10:23:55.150301 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.00563153  2.3090901  -2.3793056   1.5630075  -0.05576653 -0.04651663]
Best IK solution: [ 0.00563153  2.3090901  -2.3793056   1.5630075  -0.05576653 -0.04651663]
🌟 IK 求解耗时: 98.53 ms
[a1x_robot] IK Solution Found - joints: [ 0.00563153  2.3090901  -2.3793056   1.5630075  -0.05576653 -0.04651663], max joint diff: 0.0976 rad (5.59°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 51, terminate: False
  0%|                                                                                                  | 51/1000000 [00:16<44:16:12,  6.27it/s]EEF delta: pos=[0.00691492 0.00484361 0.00517316], rot=[-0.00746929 -0.00965799 -0.00144067], gripper: 0.933 -> 0.918 (91.8mm)
[a1x_robot] Action to be Solved - pos: [ 0.38878641 -0.00167004  0.27024769], quat[x,y,z,w]: [-0.03410175  0.67145099 -0.0071649   0.74022923]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:55.300302 140540777723712 logger.py:71] Updating safety params
I0210 10:23:55.300606 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01159384  2.3062637  -2.4114842   1.5708     -0.05779869 -0.02963334]
Best IK solution: [ 0.01159384  2.3062637  -2.4114842   1.5708     -0.05779869 -0.02963334]
🌟 IK 求解耗时: 101.59 ms
[a1x_robot] IK Solution Found - joints: [ 0.01159384  2.3062637  -2.4114842   1.5708     -0.05779869 -0.02963334], max joint diff: 0.0725 rad (4.16°)
⏱️  ✓ 执行耗时=103ms, 误差=0.0mm
Step done: False, reward: False, path length: 52, terminate: False
  0%|                                                                                                  | 52/1000000 [00:16<43:50:26,  6.34it/s]EEF delta: pos=[0.00669171 0.00875202 0.00604752], rot=[-0.01047112 -0.02207482  0.01075658], gripper: 0.933 -> 0.936 (93.6mm)
[a1x_robot] Action to be Solved - pos: [0.39352923 0.00965932 0.27291897], quat[x,y,z,w]: [-0.04339852  0.6602206  -0.00346243  0.74980887]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:55.455898 140540777723712 logger.py:71] Updating safety params
I0210 10:23:55.456243 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04188979  2.3189936  -2.4513228   1.5708     -0.06797923 -0.01889022]
Best IK solution: [ 0.04188979  2.3189936  -2.4513228   1.5708     -0.06797923 -0.01889022]
🌟 IK 求解耗时: 101.18 ms
[a1x_robot] IK Solution Found - joints: [ 0.04188979  2.3189936  -2.4513228   1.5708     -0.06797923 -0.01889022], max joint diff: 0.0541 rad (3.10°)
⏱️  ✓ 执行耗时=103ms, 误差=0.0mm
Step done: False, reward: False, path length: 53, terminate: False
  0%|                                                                                                  | 53/1000000 [00:16<43:35:47,  6.37it/s]EEF delta: pos=[0.00677446 0.00213785 0.00072968], rot=[-0.00532686 -0.01038128 -0.00750855], gripper: 0.931 -> 0.924 (92.4mm)
[a1x_robot] Action to be Solved - pos: [ 0.39376272 -0.00083439  0.26487378], quat[x,y,z,w]: [-0.03485497  0.66862167 -0.01059411  0.74270987]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:55.610417 140540777723712 logger.py:71] Updating safety params
I0210 10:23:55.610735 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01422726  2.3324525  -2.4416351   1.5708     -0.06397234 -0.02338829]
Best IK solution: [ 0.01422726  2.3324525  -2.4416351   1.5708     -0.06397234 -0.02338829]
🌟 IK 求解耗时: 106.96 ms
[a1x_robot] IK Solution Found - joints: [ 0.01422726  2.3324525  -2.4416351   1.5708     -0.06397234 -0.02338829], max joint diff: 0.0504 rad (2.89°)
⏱️  ✓ 执行耗时=109ms, 误差=0.0mm
Step done: False, reward: False, path length: 54, terminate: False
  0%|                                                                                                  | 54/1000000 [00:16<43:53:15,  6.33it/s]EEF delta: pos=[ 0.00476191 -0.00018286  0.00147875], rot=[-0.00550958 -0.01285305 -0.01112288], gripper: 0.930 -> 0.911 (91.1mm)
[a1x_robot] Action to be Solved - pos: [ 0.39292956 -0.00234826  0.26867901], quat[x,y,z,w]: [-0.0313808   0.66218783 -0.01484451  0.74853334]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:55.769500 140540777723712 logger.py:71] Updating safety params
I0210 10:23:55.769833 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.01120413  2.3248353  -2.4491572   1.5708     -0.06513959 -0.01622184]
Best IK solution: [ 0.01120413  2.3248353  -2.4491572   1.5708     -0.06513959 -0.01622184]
🌟 IK 求解耗时: 106.24 ms
[a1x_robot] IK Solution Found - joints: [ 0.01120413  2.3248353  -2.4491572   1.5708     -0.06513959 -0.01622184], max joint diff: 0.0413 rad (2.37°)
⏱️  ✓ 执行耗时=108ms, 误差=0.0mm
Step done: False, reward: False, path length: 55, terminate: False
  0%|                                                                                                  | 55/1000000 [00:17<43:55:48,  6.32it/s]EEF delta: pos=[ 0.00456166  0.00092892 -0.00052709], rot=[-0.00404874 -0.00658503  0.01000993], gripper: 0.930 -> 0.921 (92.1mm)
[a1x_robot] Action to be Solved - pos: [0.39652565 0.0065891  0.2689686 ], quat[x,y,z,w]: [-0.04404612  0.65563626 -0.00270312  0.75378626]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:55.929255 140540777723712 logger.py:71] Updating safety params
I0210 10:23:55.929576 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.03406273  2.3366954  -2.476737    1.5708     -0.06650854 -0.02924664]
Best IK solution: [ 0.03406273  2.3366954  -2.476737    1.5708     -0.06650854 -0.02924664]
🌟 IK 求解耗时: 111.14 ms
[a1x_robot] IK Solution Found - joints: [ 0.03406273  2.3366954  -2.476737    1.5708     -0.06650854 -0.02924664], max joint diff: 0.0387 rad (2.21°)
⏱️  ✓ 执行耗时=113ms, 误差=0.0mm
Step done: False, reward: False, path length: 56, terminate: False
  0%|                                                                                                  | 56/1000000 [00:17<44:24:39,  6.25it/s]EEF delta: pos=[0.00652693 0.00291034 0.00585689], rot=[-0.00503732 -0.0008041  -0.00174292], gripper: 0.930 -> 0.930 (93.0mm)
[a1x_robot] Action to be Solved - pos: [0.3998642  0.00196932 0.27279073], quat[x,y,z,w]: [-0.03467186  0.6602911  -0.01520615  0.75005487]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:56.094966 140540777723712 logger.py:71] Updating safety params
I0210 10:23:56.095302 140540777723712 logger.py:71] Updating optimizer params
prev_q: [[ 0.01489362  2.3251064  -2.442766    1.5619149  -0.06531915 -0.01829787]]
⚠️ IK solve failed: local variable 'best' referenced before assignment
⏱️  命令→状态读取 = 102.0ms
Step done: False, reward: False, path length: 57, terminate: False
  0%|                                                                                                  | 57/1000000 [00:17<44:00:13,  6.31it/s]EEF delta: pos=[0.01119651 0.00637126 0.0039478 ], rot=[-0.00299684 -0.00887968  0.00508469], gripper: 0.925 -> 0.915 (91.5mm)
[a1x_robot] Action to be Solved - pos: [0.40502694 0.00195803 0.27171736], quat[x,y,z,w]: [-0.03248458  0.65542872 -0.0162771   0.75438253]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:56.249564 140540777723712 logger.py:71] Updating safety params
I0210 10:23:56.250258 140540777723712 logger.py:71] Updating optimizer params
prev_q: [[ 0.00617021  2.325532   -2.448085    1.5619149  -0.06574468 -0.01617021]]
⚠️ IK solve failed: local variable 'best' referenced before assignment
⏱️  命令→状态读取 = 106.6ms
Step done: False, reward: False, path length: 58, terminate: False
  0%|                                                                                                  | 58/1000000 [00:17<44:02:27,  6.31it/s]EEF delta: pos=[0.00377733 0.00586935 0.00157208], rot=[-6.89950911e-03 -9.36732395e-05  5.33239963e-03], gripper: 0.925 -> 0.926 (92.6mm)
[a1x_robot] Action to be Solved - pos: [0.40027425 0.01168527 0.27102828], quat[x,y,z,w]: [-0.04642955  0.65230104 -0.00371012  0.75652752]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:56.407836 140540777723712 logger.py:71] Updating safety params
I0210 10:23:56.408341 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.0481922   2.3437707  -2.4956195   1.5708     -0.07304375 -0.01894666]
Best IK solution: [ 0.0481922   2.3437707  -2.4956195   1.5708     -0.07304375 -0.01894666]
🌟 IK 求解耗时: 115.44 ms
[a1x_robot] IK Solution Found - joints: [ 0.0481922   2.3437707  -2.4956195   1.5708     -0.07304375 -0.01894666], max joint diff: 0.0267 rad (1.53°)
⏱️  ✓ 执行耗时=118ms, 误差=0.0mm
Step done: False, reward: False, path length: 59, terminate: False
  0%|                                                                                                  | 59/1000000 [00:17<44:59:27,  6.17it/s]EEF delta: pos=[0.00218038 0.00668029 0.00585123], rot=[-0.00471345  0.00157997  0.00765547], gripper: 0.925 -> 0.934 (93.4mm)
[a1x_robot] Action to be Solved - pos: [0.39950038 0.01408124 0.27519328], quat[x,y,z,w]: [-0.04829485  0.65189457 -0.00083564  0.7567697 ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:56.578899 140540777723712 logger.py:71] Updating safety params
I0210 10:23:56.579097 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.053652    2.3341331  -2.4899204   1.5708     -0.07302006 -0.01844678]
Best IK solution: [ 0.053652    2.3341331  -2.4899204   1.5708     -0.07302006 -0.01844678]
🌟 IK 求解耗时: 97.64 ms
[a1x_robot] IK Solution Found - joints: [ 0.053652    2.3341331  -2.4899204   1.5708     -0.07302006 -0.01844678], max joint diff: 0.0177 rad (1.01°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 60, terminate: False
  0%|                                                                                                  | 60/1000000 [00:17<44:17:13,  6.27it/s]EEF delta: pos=[-0.00204183 -0.00379928  0.0025893 ], rot=[-0.00823706 -0.00796513  0.00315823], gripper: 0.925 -> 0.908 (90.8mm)
[a1x_robot] Action to be Solved - pos: [0.39539151 0.00087245 0.27188488], quat[x,y,z,w]: [-0.04632053  0.64849467 -0.00679855  0.75977816]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:56.729793 140540777723712 logger.py:71] Updating safety params
I0210 10:23:56.730162 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.02165813  2.311541   -2.4370835   1.5379937  -0.07389566 -0.04032217]
Best IK solution: [ 0.02165813  2.311541   -2.4370835   1.5379937  -0.07389566 -0.04032217]
🌟 IK 求解耗时: 140.11 ms
[a1x_robot] IK Solution Found - joints: [ 0.02165813  2.311541   -2.4370835   1.5379937  -0.07389566 -0.04032217], max joint diff: 0.0391 rad (2.24°)
⏱️  ✓ 执行耗时=142ms, 误差=0.0mm
Step done: False, reward: False, path length: 61, terminate: False
  0%|                                                                                                  | 61/1000000 [00:18<47:02:19,  5.90it/s]EEF delta: pos=[ 0.006105   -0.00278566  0.00084856], rot=[-0.00183731  0.00788907  0.00988072], gripper: 0.925 -> 0.914 (91.4mm)
[a1x_robot] Action to be Solved - pos: [0.40486995 0.00786927 0.27142121], quat[x,y,z,w]: [-4.91837423e-02  6.50220802e-01 -7.28891696e-04  7.58151262e-01]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:56.923814 140540777723712 logger.py:71] Updating safety params
I0210 10:23:56.924139 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.03850786  2.356322   -2.518494    1.5708     -0.07233045 -0.03730872]
Best IK solution: [ 0.03850786  2.356322   -2.518494    1.5708     -0.07233045 -0.03730872]
🌟 IK 求解耗时: 99.88 ms
[a1x_robot] IK Solution Found - joints: [ 0.03850786  2.356322   -2.518494    1.5708     -0.07233045 -0.03730872], max joint diff: 0.0296 rad (1.69°)
⏱️  ✓ 执行耗时=102ms, 误差=0.0mm
Step done: False, reward: False, path length: 62, terminate: False
  0%|                                                                                                  | 62/1000000 [00:18<45:44:21,  6.07it/s]EEF delta: pos=[ 0.00323096 -0.0030351   0.00331199], rot=[-0.00718726  0.00380218  0.01030603], gripper: 0.925 -> 0.925 (92.5mm)
[a1x_robot] Action to be Solved - pos: [0.40237779 0.01112458 0.27445071], quat[x,y,z,w]: [-0.05478532  0.64665076  0.00154373  0.76081469]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:57.077054 140540777723712 logger.py:71] Updating safety params
I0210 10:23:57.077345 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04719566  2.345618   -2.5133862   1.5708     -0.07602284 -0.03861365]
Best IK solution: [ 0.04719566  2.345618   -2.5133862   1.5708     -0.07602284 -0.03861365]
🌟 IK 求解耗时: 95.54 ms
[a1x_robot] IK Solution Found - joints: [ 0.04719566  2.345618   -2.5133862   1.5708     -0.07602284 -0.03861365], max joint diff: 0.0242 rad (1.39°)
⏱️  ✓ 执行耗时=100ms, 误差=0.0mm
Step done: False, reward: False, path length: 63, terminate: False
  0%|                                                                                                  | 63/1000000 [00:18<44:38:07,  6.22it/s]EEF delta: pos=[ 0.00657058  0.00706516 -0.00196475], rot=[ 0.00051569 -0.00302915  0.00095458], gripper: 0.924 -> 0.919 (91.9mm)
[a1x_robot] Action to be Solved - pos: [0.40605436 0.00968498 0.26731344], quat[x,y,z,w]: [-0.04699487  0.64498061 -0.00580425  0.76273049]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:57.228990 140540777723712 logger.py:71] Updating safety params
I0210 10:23:57.229370 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04346921  2.3699431  -2.53768     1.5708     -0.0769608  -0.02175348]
Best IK solution: [ 0.04346921  2.3699431  -2.53768     1.5708     -0.0769608  -0.02175348]
🌟 IK 求解耗时: 108.85 ms
[a1x_robot] IK Solution Found - joints: [ 0.04346921  2.3699431  -2.53768     1.5708     -0.0769608  -0.02175348], max joint diff: 0.0862 rad (4.94°)
⏱️  ✓ 执行耗时=111ms, 误差=0.0mm
Step done: False, reward: False, path length: 64, terminate: False
  0%|                                                                                                  | 64/1000000 [00:18<44:47:12,  6.20it/s]
🔍 [键盘监听器] 检测到按键: Key.ctrl, type=<enum 'Key'>
EEF delta: pos=[ 0.00660802 -0.00111441  0.00611403], rot=[-5.62560232e-03 -3.66702443e-06  1.38758812e-02], gripper: 0.924 -> 0.928 (92.8mm)
[a1x_robot] Action to be Solved - pos: [0.40548515 0.00423188 0.27844967], quat[x,y,z,w]: [-5.52493574e-02  6.44330319e-01  8.51683063e-05  7.62748937e-01]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:57.391771 140540777723712 logger.py:71] Updating safety params
I0210 10:23:57.392010 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.02886952  2.3495433  -2.5355883   1.5708     -0.07551693 -0.05311392]
Best IK solution: [ 0.02886952  2.3495433  -2.5355883   1.5708     -0.07551693 -0.05311392]
🌟 IK 求解耗时: 116.87 ms
[a1x_robot] IK Solution Found - joints: [ 0.02886952  2.3495433  -2.5355883   1.5708     -0.07551693 -0.05311392], max joint diff: 0.0509 rad (2.92°)
⏱️  ✓ 执行耗时=119ms, 误差=0.0mm
Step done: False, reward: False, path length: 65, terminate: False
  0%|                                                                                                  | 65/1000000 [00:18<45:37:55,  6.09it/s]EEF delta: pos=[0.00484513 0.00340647 0.00279314], rot=[-0.0029045   0.00569932  0.00741106], gripper: 0.924 -> 0.928 (92.8mm)
[a1x_robot] Action to be Solved - pos: [0.40583121 0.01425786 0.27637621], quat[x,y,z,w]: [-0.05666098  0.64191928  0.00281363  0.76467069]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:57.561802 140540777723712 logger.py:71] Updating safety params
I0210 10:23:57.562123 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.05497577  2.352037   -2.5354185   1.5708     -0.07753754 -0.03775564]
Best IK solution: [ 0.05497577  2.352037   -2.5354185   1.5708     -0.07753754 -0.03775564]
🌟 IK 求解耗时: 102.34 ms
[a1x_robot] IK Solution Found - joints: [ 0.05497577  2.352037   -2.5354185   1.5708     -0.07753754 -0.03775564], max joint diff: 0.0263 rad (1.51°)
⏱️  ✓ 执行耗时=104ms, 误差=0.0mm
Step done: False, reward: False, path length: 66, terminate: False
  0%|                                                                                                  | 66/1000000 [00:18<44:53:50,  6.19it/s]
🔍 [键盘监听器] 检测到按键: 'x', type=<class 'pynput.keyboard._xorg.KeyCode'>
🔍 [键盘监听器] 字符键: 'x'
^XEEF delta: pos=[ 0.00718292 -0.00204812  0.00205506], rot=[ 0.00394516 -0.00373996 -0.00097825], gripper: 0.924 -> 0.926 (92.6mm)
[a1x_robot] Action to be Solved - pos: [0.41154567 0.00531931 0.27449532], quat[x,y,z,w]: [-0.05031928  0.6358116  -0.00102494  0.77020161]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:57.717743 140540777723712 logger.py:71] Updating safety params
I0210 10:23:57.718079 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.03174985  2.3747072  -2.5737286   1.5708     -0.07177858 -0.04688413]
Best IK solution: [ 0.03174985  2.3747072  -2.5737286   1.5708     -0.07177858 -0.04688413]
🌟 IK 求解耗时: 106.63 ms
[a1x_robot] IK Solution Found - joints: [ 0.03174985  2.3747072  -2.5737286   1.5708     -0.07177858 -0.04688413], max joint diff: 0.0433 rad (2.48°)
⏱️  ✓ 执行耗时=108ms, 误差=0.0mm
Step done: False, reward: False, path length: 67, terminate: False
  0%|                                                                                                  | 67/1000000 [00:19<44:45:32,  6.21it/s]EEF delta: pos=[-0.00032212  0.00467311  0.00149093], rot=[ 0.00408295  0.01366856 -0.00234169], gripper: 0.924 -> 0.924 (92.4mm)
[a1x_robot] Action to be Solved - pos: [0.40420404 0.00948774 0.27507355], quat[x,y,z,w]: [-5.08405012e-02  6.40923872e-01 -3.45166841e-04  7.65918870e-01]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:57.875138 140540777723712 logger.py:71] Updating safety params
I0210 10:23:57.875390 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.04128814  2.352765   -2.535556    1.5708     -0.07213491 -0.03831245]
Best IK solution: [ 0.04128814  2.352765   -2.535556    1.5708     -0.07213491 -0.03831245]
🌟 IK 求解耗时: 107.11 ms
[a1x_robot] IK Solution Found - joints: [ 0.04128814  2.352765   -2.535556    1.5708     -0.07213491 -0.03831245], max joint diff: 0.0113 rad (0.65°)
⏱️  ✓ 执行耗时=109ms, 误差=0.0mm
Step done: False, reward: False, path length: 68, terminate: False
  0%|                                                                                                  | 68/1000000 [00:19<44:30:36,  6.24it/s]EEF delta: pos=[0.00798833 0.00601789 0.00364222], rot=[ 0.00602359  0.0004024  -0.00096273], gripper: 0.924 -> 0.930 (93.0mm)
[a1x_robot] Action to be Solved - pos: [0.41242242 0.01959104 0.27718421], quat[x,y,z,w]: [-0.05299808  0.63509931  0.00449382  0.77059709]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:58.035451 140540777723712 logger.py:71] Updating safety params
I0210 10:23:58.035775 140540777723712 logger.py:71] Updating optimizer params
prev_q: [[ 0.05319149  2.3529787  -2.535319    1.5614893  -0.07638298 -0.03744681]]
⚠️ IK solve failed: local variable 'best' referenced before assignment
⏱️  命令→状态读取 = 110.1ms
Step done: False, reward: False, path length: 69, terminate: False
  0%|                                                                                                  | 69/1000000 [00:19<44:35:50,  6.23it/s]EEF delta: pos=[ 0.002357   -0.00079549  0.0055292 ], rot=[ 0.00601273  0.00902016 -0.00053996], gripper: 0.924 -> 0.929 (92.9mm)
[a1x_robot] Action to be Solved - pos: [0.40961033 0.00642833 0.2801068 ], quat[x,y,z,w]: [-0.04871553  0.63357655  0.00131554  0.77214365]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:58.197143 140540777723712 logger.py:71] Updating safety params
I0210 10:23:58.197446 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.0345632   2.3597713  -2.5671017   1.5708     -0.06765286 -0.04312515]
Best IK solution: [ 0.0345632   2.3597713  -2.5671017   1.5708     -0.06765286 -0.04312515]
🌟 IK 求解耗时: 108.46 ms
[a1x_robot] IK Solution Found - joints: [ 0.0345632   2.3597713  -2.5671017   1.5708     -0.06765286 -0.04312515], max joint diff: 0.0092 rad (0.53°)
⏱️  ✓ 执行耗时=110ms, 误差=0.0mm
Step done: False, reward: False, path length: 70, terminate: False
  0%|                                                                                                  | 70/1000000 [00:19<44:40:44,  6.22it/s]EEF delta: pos=[ 0.00559147 -0.00016234  0.00055529], rot=[0.00406643 0.01184898 0.00780371], gripper: 0.924 -> 0.914 (91.4mm)
[a1x_robot] Action to be Solved - pos: [0.41114081 0.00911835 0.27326034], quat[x,y,z,w]: [-0.0514282   0.63941511  0.00485971  0.7671244 ]
假设输入是[x,y,z,w]，转换为[w,x,y,z]
I0210 10:23:58.358774 140540777723712 logger.py:71] Updating safety params
I0210 10:23:58.359111 140540777723712 logger.py:71] Updating optimizer params
prev_q: [ 0.03957798  2.3732584  -2.5659835   1.5708     -0.06370053 -0.04725473]
Best IK solution: [ 0.03957798  2.3732584  -2.5659835   1.5708     -0.06370053 -0.04725473]
🌟 IK 求解耗时: 100.16 ms
[a1x_robot] IK Solution Found - joints: [ 0.03957798  2.3732584  -2.5659835   1.5708     -0.06370053 -0.04725473], max joint diff: 0.0249 rad (1.43°)
⏱️  ✓ 执行耗时=102ms, 误差=0.0mm
Step done: False, reward: False, path length: 71, terminate: False
  0%|                                                                       