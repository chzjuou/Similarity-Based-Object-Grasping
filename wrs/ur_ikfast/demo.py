from ur_ikfast import ur_kinematics
import time
import numpy as np

ur5e_arm = ur_kinematics.URKinematics('ur5e')

joint_angles = [-1.31981798, -0.95748763, -2.48307993,  0.31503193,  1.33570048, -0.04153884]  # in radians
print("joint angles", joint_angles)

pose_quat = ur5e_arm.forward(joint_angles)
pose_matrix = ur5e_arm.forward(joint_angles, 'matrix')
rotmat = np.array([[0,0,-1],[1,0,0],[0,-1,0]])
rot_matrix = np.dot(pose_matrix[:3, :3], rotmat)
mat1 = np.linalg.inv(pose_matrix[:3, :3])
mat2 = np.array([[-0.01620061,  0.01525131, -0.99975244],
 [ 0.03753716, -0.99916951, -0.01585069],
 [-0.9991639,  -0.03778466,  0.01561466]])
#pose_matrix = np.dot(mat1, mat2)

print("forward() quaternion \n", pose_quat)
print("forward() matrix \n", rot_matrix)

start = time.time()
# print("inverse() all", ur3e_arm.inverse(pose_quat, True))
print("inverse() one from quat", ur5e_arm.inverse(pose_quat, False, q_guess=joint_angles))

print("inverse() one from matrix", ur5e_arm.inverse(pose_matrix, False, q_guess=joint_angles))
print(time.time()-start)
