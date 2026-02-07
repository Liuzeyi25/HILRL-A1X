nconrft baseline：
 


硬件系统功能开发日志：
2. 测试相机
python /home/dungeon_master/conrft/examples/test_script/test_cameras.py

3. 测试机械臂ros2节点功能
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots && python test_eef_control.py

4. 回到初始位置
 ~/conrft/reset_a1x.py 

5. chunk多步累积误差检测 优化chunk执行的误差，但是执行速度还有一点卡顿
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots
python test_cumulative_error.py

6. 那么一步多次执行呢？
测试文档见：/home/dungeon_master/conrft/serl_robot_infra/franka_env/robots/MOVEMENT_STRATEGY_SUMMARY.md

7. 完善action chunk的执行，在推理时延和精准度上实现了trade off，测试代码：
 cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots
python test_eef_chunk_command.py

8. 测试gello follow的功能以及遥控的切换 （空格切换遥控到跟随）
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots
python3 test_gello_simple.py

gello波特率 57600

simple test gello的控制模式python3 test_gello_command.py

9.测试特teleoperate功能
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots && python3 test_teleoperation_performance.py <<< $'y\n1'

10. 测试action space，会全面测试reset以及遥控跟随
/home/dungeon_master/conrft/examples/test_script
python verify_action_space.py

11. 夹爪控制问题

env:
A1XTaskEnv
GelloIntervention
SERLObsWrapper
ChunkingWrapper
A1XGripperPenaltyWrapper

action_space
现在需要修改一下GelloIntervention以及A1XTaskEnv，Gello需要


12. 加入系统加入力矩
ros2 topic echo /hdas/feedback_arm




右肚 18
lrwxrwxrwx 1 root root 13 Jan 13 16:30 /dev/v4l/by-path/pci-0000:12:00.0-usb-0:10:1.0-video-index0 -> ../../video18

右指尖 16n
lrwxrwxrwx 1 root root 13 Jan 13 16:30 /dev/v4l/by-path/pci-0000:12:00.0-usb-0:5:1.0-video-index0 -> ../../video16

左指尖 6
lrwxrwxrwx 1 root root 12 Jan 13 21:23 /dev/v4l/by-path/pci-0000:12:00.0-usb-0:2:1.0-video-index0 -> ../../video6


左肚 14
lrwxrwxrwx 1 root root 13 Jan 13 16:30 /dev/v4l/by-path/pci-0000:12:00.0-usb-0:4:1.0-video-index0 -> ../../video14



/dev/cam_left_1 -> /dev/video6
/dev/cam_left_2 -> /dev/video18
/dev/cam_right_1 -> /dev/video16 
/dev/cam_right_2 -> /dev/video14

top video16
wrist video6

# left_1  (usb-0:2)
SUBSYSTEM=="video4linux", KERNELS=="1-3:1.0", ATTR{index}=="0", SYMLINK+="cam_left_1"

# left_2  (usb-0:4)
SUBSYSTEM=="video4linux", KERNELS=="3-3:1.0", ATTR{index}=="0", SYMLINK+="cam_left_2"

# right_1 (usb-0:5)
SUBSYSTEM=="video4linux", KERNELS=="3-2:1.0", ATTR{index}=="0", SYMLINK+="cam_right_1"

# right_2 (usb-0:10)
SUBSYSTEM=="video4linux", KERNELS=="3-1:1.0", ATTR{index}=="0", SYMLINK+="cam_right_2"


prompt

2. 测试mlp以及不同Vit架构的训练-测试性能

