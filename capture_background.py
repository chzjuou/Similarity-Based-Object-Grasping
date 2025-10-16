import numpy as np
import math
import cv2
import pyrealsense2 as rs
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# Boot camera
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe = rs.pipeline()
profile = pipe.start(cfg)

# Connect robot
with open('configs/robot_ip.txt', 'r') as f:
    robot_ip = f.read().strip()
rtde_frequency = 500.0
rtde_c = RTDEControl(robot_ip, rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
rtde_r = RTDEReceive(robot_ip, rtde_frequency)

try:
    # Move to observation pose
    obs_jvs = np.loadtxt('configs/observation.txt')
    rtde_c.moveJ(obs_jvs)
    
    # Capture background image
    for k in range(30):
        frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite('results/bg.png', color_image)
    
    # Get scene point cloud
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    ply = rs.save_to_ply('results/scene.ply')
    ply.set_option(rs.save_to_ply.option_ply_binary, True)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)
    ply.process(frames)  # Save the frames to the .ply file

except KeyboardInterrupt:
    print("Control Interrupted!")
    rtde_c.servoStop()
    rtde_c.stopScript()
    pipe.stop()

finally:
    rtde_c.servoStop()
    rtde_c.stopScript()
    pipe.stop()
