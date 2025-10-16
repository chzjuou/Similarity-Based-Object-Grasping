import os
import numpy as np
import math
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from robotiq_gripper import robotiq

# Connect robot and gripper
with open('configs/robot_ip.txt', 'r') as f:
    robot_ip = f.read().strip()
rtde_frequency = 500.0
rtde_c = RTDEControl(robot_ip, rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
rtde_r = RTDEReceive(robot_ip, rtde_frequency)
gripper = robotiq()
gripper.connect(robot_ip, 54321)
gripper.reset()
gripper.activate()
gripper.wait_activate_complete()

try:
    init_jvs = np.loadtxt('configs/observation.txt')
    rtde_c.moveJ(init_jvs)

    originalTime = os.path.getmtime('signals/signal4.txt')
    while True:
        #break
        if os.path.getmtime('signals/signal4.txt') > originalTime:
            print("Start robotic manipulation.")
            time.sleep(0.1)
            break

    pregrasp_jvs = np.loadtxt('results/pregrasp.txt')
    grasp_jvs = np.loadtxt('results/grasp.txt')
    middle_jvs = np.loadtxt('results/middle.txt')
    rtde_c.moveJ(pregrasp_jvs)
    rtde_c.moveJ(grasp_jvs)
    gripper.move(255, 255, 50)  # close
    gripper.wait_move_complete()
    rtde_c.moveJ(middle_jvs)
    rtde_c.moveJ(grasp_jvs)
    gripper.move(0, 255, 50)  # open
    gripper.wait_move_complete()
    rtde_c.moveJ(pregrasp_jvs)
    rtde_c.moveJ(init_jvs)

except KeyboardInterrupt:
    print("Control Interrupted!")
    rtde_c.servoStop()
    rtde_c.stopScript()
    gripper.disconnect()

finally:
    rtde_c.servoStop()
    rtde_c.stopScript()
    gripper.disconnect()
