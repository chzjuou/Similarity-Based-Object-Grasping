import os
import copy
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur5e.ur5e as rbt
import robot_sim.end_effectors.gripper.robotiq140.robotiq140 as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
import sys
sys.path.append('../../../ur_ikfast')
from ur_ikfast.ur_ikfast import ur_kinematics
from sklearn.decomposition import PCA

class UR5EConveyorBelt(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="ur5e_conveyorbelt", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_stand = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(4),
                                     name='base_stand')
        self.base_stand.jnts[1]['loc_pos'] = np.array([.9, -1.5, -0.06])
        self.base_stand.jnts[2]['loc_pos'] = np.array([0, 1.23, 0])
        self.base_stand.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.base_stand.jnts[4]['loc_pos'] = np.array([-.9, .27, 0.06])
        self.base_stand.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur5e_base.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_stand.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand.reinitialize()
        # arm
        arm_homeconf = np.zeros(6)
        # arm_homeconf[0] = math.pi / 2
        arm_homeconf[1] = -math.pi * 2 / 3
        arm_homeconf[2] = math.pi / 3
        arm_homeconf[3] = -math.pi / 2
        arm_homeconf[4] = -math.pi / 2
        self.arm = rbt.UR5E(pos=self.base_stand.jnts[-1]['gl_posq'],
                            rotmat=self.base_stand.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf,
                            name='arm', enable_cc=False)
        # camera
        self.camera = jl.JLChain(pos=self.arm.jnts[-1]['gl_posq'],
                                 rotmat=np.dot(self.arm.jnts[-1]['gl_rotmatq'],
                                               rm.rotmat_from_axangle([0, 0, 1], math.pi)),
                                 homeconf=np.zeros(0), name='camera')
        self.camera.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "camera.stl"),
            cdprimit_type="box", expand_radius=.01)
        self.camera.lnks[0]['rgba'] = [1, 1, 1, 0]
        self.camera.reinitialize()

        # gripper
        self.hnd = hnd.Robotiq140(pos=self.arm.jnts[-1]['gl_posq'],
                                  rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                  name='hnd_s', enable_cc=False)
        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-0.1, 0.0, 0.14 - 0.82),
                                              x=.35 + radius, y=.3 + radius, z=.14 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, -.3),
                                              x=.112 + radius, y=.112 + radius, z=.3 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.base_stand, [0])
        self.cc.add_cdlnks(self.arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.camera, [0])
        self.cc.add_cdlnks(self.hnd.lft_outer, [0, 1, 2, 3, 4])
        self.cc.add_cdlnks(self.hnd.rgt_outer, [1, 2, 3, 4])
        activelist = [#self.base_stand.lnks[0],
                      # self.base_stand.lnks[1],
                      # self.base_stand.lnks[2],
                      # self.base_stand.lnks[3],
                      # self.base_stand.lnks[4],
                      self.camera.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft_outer.lnks[0],
                      self.hnd.lft_outer.lnks[1],
                      self.hnd.lft_outer.lnks[2],
                      self.hnd.lft_outer.lnks[3],
                      self.hnd.lft_outer.lnks[4],
                      self.hnd.rgt_outer.lnks[1],
                      self.hnd.rgt_outer.lnks[2],
                      self.hnd.rgt_outer.lnks[3],
                      self.hnd.rgt_outer.lnks[4]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.base_stand.lnks[0],
                    # self.base_stand.lnks[1],
                    # self.base_stand.lnks[2],
                    # self.base_stand.lnks[3],
                    # self.base_stand.lnks[4],
                    self.arm.lnks[1]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.lft_outer.lnks[4],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[2]]
        intolist = [self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.lft_outer.lnks[4],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[3]]
        intolist = [self.camera.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.lft_outer.lnks[4],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold('arm', objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_stand.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_stand.jnts[-1]['gl_posq'], rotmat=self.base_stand.jnts[-1]['gl_rotmatq'])
        self.camera.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.camera.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name='hnd_s', jawwidth=0.0):
        self.hnd.jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.base_stand.lnks[0],
                    # self.base_stand.lnks[1],
                    # self.base_stand.lnks[2],
                    # self.base_stand.lnks[3],
                    # self.base_stand.lnks[4],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_stand.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.camera.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                  tcp_loc_pos=tcp_loc_pos,
                                  tcp_loc_rotmat=tcp_loc_rotmat,
                                  toggle_tcpcs=False,
                                  toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.base_stand.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.camera.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                  tcp_loc_pos=tcp_loc_pos,
                                  tcp_loc_rotmat=tcp_loc_rotmat,
                                  toggle_tcpcs=False,
                                  toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.hnd.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def plot_gripper_pro_max(center, R, width, depth, score=1.0, color=None):
    height = 0.004
    finger_width = 0.004
    tail_length = 0.06
    depth_base = 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def downsample_points_grid(points, grid_size):
    if len(points) == 0:
        return points

    # Compute the grid indices for each point
    grid_indices = np.floor(points / grid_size).astype(int)

    # Use a dictionary to group points by their grid cell
    grid_dict = {}
    for idx, grid_coord in enumerate(map(tuple, grid_indices)):
        if grid_coord not in grid_dict:
            grid_dict[grid_coord] = []
        grid_dict[grid_coord].append(points[idx])

    # Choose a representative point from each grid cell (e.g., the centroid)
    downsampled_points = [
        np.mean(cell_points, axis=0) for cell_points in grid_dict.values()
    ]

    return np.array(downsampled_points)

def SOBB(pcd):
    plane_normal = np.array([0, 0, 1])
    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    projection_matrix = np.eye(3) - np.outer(plane_normal_normalized, plane_normal_normalized)
    projected_points = np.asarray(pcd.points) @ projection_matrix.T
    downsampled_points = downsample_points_grid(projected_points[:, :2], 0.002)
    zeros_column = np.zeros((*downsampled_points.shape[:-1], 1))
    downsampled_points = np.concatenate((downsampled_points, zeros_column), axis=-1)
    pca = PCA(n_components=3)
    pca.fit(downsampled_points)
    sigma1 = pca.singular_values_[0]
    sigma2 = pca.singular_values_[1]
    sigma3 = pca.singular_values_[2]
    singular_values = [sigma1, sigma2, sigma3]
    index1 = np.argsort(singular_values)[-1]
    index2 = np.argsort(singular_values)[-2]
    vec1 = pca.components_[index1]
    vec2 = pca.components_[index2]
    vec3 = plane_normal_normalized
    rotmat = np.column_stack([vec1, vec2, vec3])
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.rotate(rotmat.T, center=(0, 0, 0))
    aabb_rotated = pcd_copy.get_axis_aligned_bounding_box()
    center = aabb_rotated.get_center()
    max_bound = aabb_rotated.get_max_bound()
    min_bound = aabb_rotated.get_min_bound()
    obb_rotated = o3d.geometry.OrientedBoundingBox(center, np.identity(3), max_bound - min_bound)
    obb_rotated.rotate(rotmat, center=(0, 0, 0))
    return obb_rotated


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from manipulation.pick_place_planner import PickPlacePlanner
    from scipy.stats import norm
    import open3d as o3d
    import subprocess
    import shlex
    import random

    # Initialization
    ur5e_arm = ur_kinematics.URKinematics('ur5e')
    base = wd.World(cam_pos=[2, -2, 2], lookat_pos=[-0.2, 0, 0.2])
    gm.gen_frame().attach_to(base)
    component_name = 'arm'
    robot_s = UR5EConveyorBelt(enable_cc=True)

    # Load initial settings
    inijnts = np.loadtxt('../../../../configs/observation.txt')
    robot_s.fk(component_name, inijnts)
    robot_s.gen_meshmodel().attach_to(base)
    cali = np.loadtxt('../../../../configs/calibration.txt').reshape(4,4)
    cali_pos = cali[0:3, 3]
    cali_rot = cali[0:3, 0:3]
    axis_rot = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)

    # Compute camera pose
    robot_s.arm.jnts[-1]['gl_rotmatq'] = robot_s.arm.jnts[-1]['gl_rotmatq'].dot(axis_rot)
    cam_pos = robot_s.arm.jnts[-1]['gl_posq'] + robot_s.arm.jnts[-1]['gl_rotmatq'].dot(cali_pos)
    cam_rotmat = robot_s.arm.jnts[-1]['gl_rotmatq']
    mat1 = np.c_[cam_rotmat, cam_pos]
    mat1 = np.r_[mat1, np.array([[0, 0, 0, 1]])]
    #gm.gen_frame(cam_pos, cam_rotmat).attach_to(base)
    np.savetxt('../../../../results/campos.txt', mat1)

    # Start similarity matching
    cmd = "python similarity_matching.py"
    subprocess.Popen(shlex.split(cmd), cwd='.').wait()
        
    # Load similar candidates, transformation matrices, and scene point cloud
    with open('../../../../results/candidates.txt') as f:
        candidates = f.read()
    candidates = candidates.split('\n')
    transformation = np.load('../../../../results/tf_matrix.npy')
    scene = o3d.io.read_point_cloud('../../../../results/scene.ply')
    scene.rotate(rm.rotmat_from_axangle([1,0,0], math.pi), [0,0,0])
    scene.transform(mat1)
    scene = scene.voxel_down_sample(0.01)
    scene_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(scene, 0.01)
    platform = cm.CollisionModel(scene_mesh)
    platform.set_rgba([1,1,1, 1])
    platform.change_cdprimitive_type('point_cloud', expand_radius=0.005)
    platform.attach_to(base)
    #base.run()

    # Start similarity-based grasp planning
    unstable_grasps, unstable_pregrasps, unstable_leaves = [], [], []
    for i in range(len(candidates) - 1):
        # Load preplanned grasps of reference models
        ref_model = candidates[i]
        print(ref_model)
        model2obj = transformation[i]
        model = cm.CollisionModel(f'../../../../database/mesh/{ref_model}.ply')
        tcp = np.loadtxt('../../../0000_book/preplanned_grasps/tcp_result_{}.txt'.format(ref_model))
        jaw = np.loadtxt('../../../0000_book/preplanned_grasps/grip_result_{}.txt'.format(ref_model))
        grarotmat = np.loadtxt('../../../0000_book/preplanned_grasps/rot_result_{}.txt'.format(ref_model))
        jaw_width = np.loadtxt('../../../0000_book/preplanned_grasps/jaw_width_{}.txt'.format(ref_model))

        # Compute object pose
        mat1 = np.c_[cam_rotmat, cam_pos]
        mat1 = np.r_[mat1, np.array([[0, 0, 0, 1]])]
        model2obj = np.linalg.inv(model2obj)
        trapos = model2obj[0:3, 3]
        trarotmat = model2obj[0:3, 0:3]
        mat2 = np.c_[trarotmat, trapos]
        mat2 = np.r_[mat2, np.array([[0, 0, 0, 1]])]
        mat = np.dot(mat1, mat2)
        obj_pos = mat[0:3, 3]
        obj_rot = mat[0:3, 0:3]
        #gm.gen_frame(obj_pos, obj_rot).attach_to(base)

        # Generate reference model in the simulation
        model_copy = model.copy()
        model_copy.change_cdprimitive_type('point_cloud', expand_radius=0.001)
        model_copy.set_pos(obj_pos)
        model_copy.set_rotmat(obj_rot)
        model_copy.set_rgba([1, 0, 0, .5])
        model_show = model_copy.copy()
        model_show.attach_to(base)

        # Build collision model for target object
        pcd = o3d.io.read_point_cloud('../../../../results/obj_down.ply')
        pcd.transform(mat1)
        gm.gen_pointcloud(pcd.points, [[1, 1, 0, 1]], 3).attach_to(base)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        target = cm.CollisionModel(mesh)
        target.change_cdprimitive_type('point_cloud', expand_radius=0.005)
        #base.run()

        # Start grasp planning
        index = 0
        signal = 0
        potential_grasps, potential_pregrasps, potential_leaves = [], [], []
        for index in range(len(tcp)):
            #index = random.randint(0, len(tcp) - 1)

            # Transfer grasp knowledge from reference model to target object
            robot_s_copy = robot_s.copy()
            planner = PickPlacePlanner(robot_s_copy)
            gripos = tcp[index]
            jawcenter = obj_rot.dot(jaw[index]) + obj_pos
            grirot = rm.rotmat_from_euler(grarotmat[index][0], grarotmat[index][1], grarotmat[index][2])
            wid = jaw_width[index]
            start_tcp_pos = obj_rot.dot(gripos) + obj_pos
            start_tcp_rotmat = obj_rot.dot(grirot)

            # Remove grasps from the bottom
            direction = start_tcp_rotmat.dot([0, 0, 1])
            angle = rm.angle_between_vectors(direction, [0, 0, 1])
            if angle < math.pi / 2:
                continue

            # Remove grasps close to the edge
            extents = pcd.get_axis_aligned_bounding_box().get_extent()
            deviation_x = extents[0] / 2
            deviation_y = extents[1] / 2
            deviation_z = extents[2] / 2
            objcenter = pcd.get_axis_aligned_bounding_box().get_center()
            distance_x = jawcenter[0] - objcenter[0]
            distance_y = jawcenter[1] - objcenter[1]
            distance_z = jawcenter[2] - objcenter[2]
            score = min(
                norm(loc=0, scale=deviation_x).pdf(distance_x) * math.sqrt(2 * math.pi * deviation_x ** 2),
                norm(loc=0, scale=deviation_y).pdf(distance_y) * math.sqrt(2 * math.pi * deviation_y ** 2),
                norm(loc=0, scale=deviation_z).pdf(distance_z) * math.sqrt(2 * math.pi * deviation_z ** 2))
            if score < 0.7:
                print('Grasp is not stable!')
                continue

            # Compute ik and check collisions
            pos = start_tcp_pos
            pos = np.array([pos[1], -pos[0], pos[2]])
            rot = rm.quaternion_from_matrix(start_tcp_rotmat)
            rot = np.array([rot[1], rot[2], -rot[0], rot[3]])
            pose_matrix = np.concatenate([pos, rot])
            pick = ur5e_arm.inverse(pose_matrix, False, q_guess=inijnts)
            if pick is None:
                print("Cannot generate the initial IK grasp!")
                continue
            robot_s_copy.fk(component_name, pick)
            robot_s_copy.jaw_to(component_name, wid)
            if robot_s_copy.is_collided([platform]):
                print("Cannot generate the initial grasp!")
                continue

            # Plan approach and leave motions
            approach = planner.inik_slvr.gen_rel_linear_motion(component_name=component_name,
                                                                goal_tcp_pos=start_tcp_pos,
                                                                goal_tcp_rotmat=start_tcp_rotmat,
                                                                direction=start_tcp_rotmat[:, 2],
                                                                distance=0.1,
                                                                obstacle_list=[platform],
                                                                granularity=0.01,
                                                                seed_jnt_values=pick,
                                                                type='sink')
            if approach is None:
                print("Cannot generate the approach motion!")
                continue

            leave = planner.inik_slvr.gen_rel_linear_motion(component_name=component_name,
                                                            goal_tcp_pos=start_tcp_pos,
                                                            goal_tcp_rotmat=start_tcp_rotmat,
                                                            direction=np.array([0, 0, -1]),
                                                            distance=0.15,
                                                            obstacle_list=[platform],
                                                            granularity=0.01,
                                                            seed_jnt_values=None,
                                                            type='sink')
            if leave is None:
                print("Cannot generate the leave motion!")
                continue

            robot_s.fk(component_name, jnt_values=approach[-1])
            robot_s.jaw_to(component_name, 0.14)
            if robot_s.is_collided([target]):
                print("Gripper and object is collided")
                continue

            # Two-stage stability-aware grasp fine-tuning
            lft = jawcenter + start_tcp_rotmat.dot([0, 0.07, 0])
            rgt = jawcenter - start_tcp_rotmat.dot([0, 0.07, 0])
            stick = cm.gen_stick(lft, rgt, 0.005)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            grip_normal = rgt - lft
            collision_points, collision_normals = target.ray_hit(lft, rgt)
            anglelist, pointlist, cdindexlist = [], [], []
            if collision_points != []:
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                collision_points = np.asarray(collision_points)
                [j, cdindex0, _] = pcd_tree.search_knn_vector_3d(collision_points[0], 1)  # points[index]
                point0 = points[cdindex0[0]]
                pcd.paint_uniform_color([1, 1, 0])
                pcd.colors[cdindex0[0]] = [1, 0, 0]
                [k, idx0, _] = pcd_tree.search_knn_vector_3d(point0, 50)
                np.asarray(pcd.colors)[idx0[1:], :] = [0, 0, 1]
                contact_normal = normals[cdindex0[0]]
                angle0 = np.arccos(np.dot(contact_normal, grip_normal) / \
                                   (np.linalg.norm(contact_normal) * np.linalg.norm(grip_normal)))
                if angle0 > math.pi / 2:
                    angle0 = math.pi - angle0
                anglelist.append(angle0)
                pointlist.append(point0)
                cdindexlist.append(cdindex0)
                print('angle0: ', angle0 / math.pi * 180)
                if len(collision_points) > 1:
                    [j, cdindex1, _] = pcd_tree.search_knn_vector_3d(collision_points[-1], 1)  # points[index]
                    point1 = points[cdindex1[0]]
                    pcd.paint_uniform_color([1, 1, 0])
                    pcd.colors[cdindex1[0]] = [1, 0, 0]
                    [k, idx1, _] = pcd_tree.search_knn_vector_3d(point1, 50)
                    np.asarray(pcd.colors)[idx1[1:], :] = [0, 0, 1]
                    contact_normal = normals[cdindex1[0]]
                    angle1 = np.arccos(np.dot(contact_normal, grip_normal) / \
                                       (np.linalg.norm(contact_normal) * np.linalg.norm(grip_normal)))
                    if angle1 > math.pi / 2:
                        angle1 = math.pi - angle1
                    anglelist.append(angle1)
                    pointlist.append(point1)
                    cdindexlist.append(cdindex1)
                    print('angle1: ', angle1 / math.pi * 180)

                non_robust, non_robust_index = 0, 0
                if all([angle / math.pi * 180 < 30 for angle in anglelist]):
                    for point in pointlist:
                        [k, kdx, _] = pcd_tree.search_knn_vector_3d(point, 5)
                        for neighbour1 in kdx[1:]:
                            neighbour_normal1 = normals[neighbour1]
                            neighbour_angle1 = np.arccos(np.dot(neighbour_normal1, contact_normal) / \
                                                         (np.linalg.norm(neighbour_normal1) * np.linalg.norm(
                                                             contact_normal)))
                            if neighbour_angle1 > math.pi / 2:
                                neighbour_angle1 = math.pi - neighbour_angle1
                            if neighbour_angle1 / math.pi * 180 > 15:
                                non_robust = 1
                                break
                        if non_robust == 1:
                            break
                        else:
                            non_robust_index += 1
                    if non_robust == 0:
                        bounds = SOBB(pcd)
                        bounds_center = bounds.center
                        bounds_rotation = bounds.R
                        homomat = np.c_[bounds_rotation, bounds_center]
                        homomat = np.r_[homomat, np.array([[0, 0, 0, 1]])]
                        bounds_box = cm.gen_box(extent=bounds.extent, homomat=homomat)
                        collision_points, collision_normals = bounds_box.ray_hit(lft, rgt)
                        if collision_points == []:
                            adjust_conf = approach[-1]
                            break
                        collision_points = np.asarray(sorted(collision_points, key=lambda x: x[0]))
                        medium_point = (collision_points[0] + collision_points[-1]) / 2
                        gm.gen_sphere(medium_point, 0.01, [1, 1, 1, 1]).attach_to(base)
                        pos = start_tcp_pos + medium_point - jawcenter
                        pos = np.array([pos[1], -pos[0], pos[2]])
                        rot = rm.quaternion_from_matrix(start_tcp_rotmat)
                        rot = np.array([rot[1], rot[2], -rot[0], rot[3]])
                        pose_matrix = np.concatenate([pos, rot])
                        adjust_conf = ur5e_arm.inverse(pose_matrix, False, q_guess=inijnts)
                        if adjust_conf is not None:
                            robot_s.fk(component_name, jnt_values=adjust_conf)
                            if robot_s.is_collided([target, platform]):
                                adjust_conf = approach[-1]
                                robot_s.fk(component_name, jnt_values=adjust_conf)
                            else:
                                robot_s.gen_meshmodel(rgba=[1, 1, 1, 0.5]).attach_to(base)
                        else:
                            adjust_conf = approach[-1]

                elif any([angle / math.pi * 180 > 45 for angle in anglelist]):
                    unstable_grasps.append(approach[-1])
                    unstable_pregrasps.append(approach[0])
                    unstable_leaves.append(leave[0])
                    print("Grasp is too non-robust")
                    continue

                if non_robust == 1 or any([30 < angle / math.pi * 180 < 45 for angle in anglelist]):
                    adjust_conf = None
                    if non_robust_index == 1:
                        idx = idx1
                    else:
                        idx = idx0
                    if any([30 < angle / math.pi * 180 < 45 for angle in anglelist]):
                        if np.where(np.asarray([30 < angle / math.pi * 180 < 45 for angle in anglelist]) == True)[0][0] == 1:
                            idx = idx1
                        else:
                            idx = idx0
                    for neighbour in idx:
                        neighbour_normal = normals[neighbour]
                        neighbour_angle = np.arccos(np.dot(neighbour_normal, grip_normal) / \
                                                    (np.linalg.norm(neighbour_normal) * np.linalg.norm(grip_normal)))
                        if neighbour_angle > math.pi / 2:
                            neighbour_angle = math.pi - neighbour_angle
                        if neighbour_angle / math.pi * 180 < 30:
                            [l, ldx, _] = pcd_tree.search_knn_vector_3d(points[neighbour], 5)
                            non_robust = 0
                            for neighbour2 in ldx[1:]:
                                neighbour_normal2 = normals[neighbour2]
                                neighbour_angle2 = np.arccos(np.dot(neighbour_normal2, neighbour_normal) / \
                                                             (np.linalg.norm(neighbour_normal2) * np.linalg.norm(
                                                                 neighbour_normal)))
                                if neighbour_angle2 > math.pi / 2:
                                    neighbour_angle2 = math.pi - neighbour_angle2
                                if neighbour_angle2 / math.pi * 180 > 15:
                                    non_robust = 1
                                    break

                            if non_robust == 0:
                                for ind in cdindexlist:
                                    adjust1 = points[neighbour] - points[ind].flatten()
                                    pos = start_tcp_pos + adjust1
                                    pos = np.array([pos[1], -pos[0], pos[2]])
                                    rot = rm.quaternion_from_matrix(start_tcp_rotmat)
                                    rot = np.array([rot[1], rot[2], -rot[0], rot[3]])
                                    pose_matrix = np.concatenate([pos, rot])
                                    adjust_conf = ur5e_arm.inverse(pose_matrix, False, q_guess=inijnts)
                                    if adjust_conf is not None:
                                        robot_s.fk(component_name, jnt_values=adjust_conf)
                                        if robot_s.is_collided([target, platform]):
                                            print('First fine-tuning is failed!')
                                            adjust_conf = None
                                            continue
                                        else:
                                            break
                                if adjust_conf is not None:
                                    lft = jawcenter + adjust1 + start_tcp_rotmat.dot([0, 0.07, 0])  # 0.07
                                    rgt = jawcenter + adjust1 - start_tcp_rotmat.dot([0, 0.07, 0])  # 0.07
                                    collision_points, collision_normals = target.ray_hit(lft, rgt)
                                    for collision_point in collision_points:
                                        [j, cdindex, _] = pcd_tree.search_knn_vector_3d(collision_point, 1)
                                        contact_normal = normals[cdindex[0]]
                                        angle = np.arccos(np.dot(contact_normal, grip_normal) / \
                                                          (np.linalg.norm(contact_normal) * np.linalg.norm(
                                                              grip_normal)))
                                        if angle > math.pi / 2:
                                            angle = math.pi - angle
                                        if angle / math.pi * 180 > 15:
                                            non_robust = 1
                                    if non_robust == 1:
                                        print('Still not robust!')
                                        adjust_conf = None
                                        continue

                                    bounds = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
                                    bounds_center = bounds.center
                                    bounds_rotation = bounds.R
                                    homomat = np.c_[bounds_rotation, bounds_center]
                                    homomat = np.r_[homomat, np.array([[0, 0, 0, 1]])]
                                    bounds_box = cm.gen_box(extent=bounds.extent, homomat=homomat)
                                    collision_points, collision_normals = bounds_box.ray_hit(lft, rgt)
                                    if collision_points == []:
                                        print('Second fine-tuning is failed!')
                                        adjust_conf = None
                                        continue
                                    collision_points = np.asarray(sorted(collision_points, key=lambda x: x[0]))
                                    medium_point = (collision_points[0] + collision_points[-1]) / 2
                                    gm.gen_sphere(medium_point, 0.01, [1, 1, 1, 1]).attach_to(base)
                                    adjust2 = medium_point - jawcenter

                                    pos = start_tcp_pos + adjust2
                                    pos = np.array([pos[1], -pos[0], pos[2]])
                                    rot = rm.quaternion_from_matrix(start_tcp_rotmat)
                                    rot = np.array([rot[1], rot[2], -rot[0], rot[3]])
                                    pose_matrix = np.concatenate([pos, rot])
                                    adjust_conf = ur5e_arm.inverse(pose_matrix, False, q_guess=inijnts)
                                    if adjust_conf is not None:
                                        robot_s.fk(component_name, jnt_values=adjust_conf)
                                        if robot_s.is_collided([target, platform]):
                                            pos = start_tcp_pos + adjust1
                                            pos = np.array([pos[1], -pos[0], pos[2]])
                                            rot = rm.quaternion_from_matrix(start_tcp_rotmat)
                                            rot = np.array([rot[1], rot[2], -rot[0], rot[3]])
                                            pose_matrix = np.concatenate([pos, rot])
                                            adjust_conf = ur5e_arm.inverse(pose_matrix, False, q_guess=inijnts)
                                            break
                                        lft = jawcenter + adjust2 + start_tcp_rotmat.dot([0, 0.07, 0])
                                        rgt = jawcenter + adjust2 - start_tcp_rotmat.dot([0, 0.07, 0])
                                        gm.gen_sphere(lft, rgba=[1, 1, 1, 1]).attach_to(base)
                                        gm.gen_sphere(rgt, rgba=[1, 1, 1, 1]).attach_to(base)
                                        gm.gen_stick(lft, rgt, rgba=[1, 1, 1, 1]).attach_to(base)
                                        break

                if adjust_conf is None:
                    continue
                print('Successfully plan a motion!')
                signal = 1
                break
            else:
                homomat = np.c_[
                    start_tcp_rotmat, robot_s.arm.jnts[-1]['gl_posq'] + robot_s.arm.jnts[-1]['gl_rotmatq'].dot(
                        [0, 0, 0.205 - 0.03])]
                homomat = np.r_[homomat, np.array([[0, 0, 0, 1]])]
                cube = cm.gen_box([0.01, 0.14, 0.06], homomat, [1, 0, 0, 0.5])
                #cube.attach_to(base)
                if cube.is_mcdwith([target]) and not robot_s.is_collided([target]):
                    potential_grasps.append(approach[-1])
                    potential_pregrasps.append(approach[0])
                    potential_leaves.append(leave[0])

        if signal == 1:
            break
        elif potential_grasps != []:
            bounds = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            bounds_center = bounds.center
            mindistance = 999
            for i in range(len(potential_grasps)):
                robot_s.fk(component_name, potential_grasps[i])
                distance = math.dist(bounds_center,
                                     robot_s.arm.jnts[-1]['gl_posq'] + robot_s.arm.jnts[-1]['gl_rotmatq'].dot(
                                         [0, 0, 0.205]))
                if distance < mindistance:
                    mindistance = distance
                    best_grasp = potential_grasps[i]
                    best_pregrasp = potential_pregrasps[i]
                    best_leave = potential_leaves[i]
            robot_s.fk(component_name, best_grasp)
            robot_s.gen_meshmodel(toggle_tcpcs=False).attach_to(base)
            signal = 2
            print('Using potential grasps!')
            break
    if signal == 0:
        if unstable_grasps != []:
            bounds = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            bounds_center = bounds.center
            mindistance = 999
            for i in range(len(unstable_grasps)):
                robot_s.fk(component_name, unstable_grasps[i])
                distance = math.dist(bounds_center,
                                     robot_s.arm.jnts[-1]['gl_posq'] + robot_s.arm.jnts[-1]['gl_rotmatq'].dot(
                                         [0, 0, 0.205]))
                if distance < mindistance:
                    mindistance = distance
                    best_grasp = unstable_grasps[i]
                    best_pregrasp = unstable_pregrasps[i]
                    best_leave = unstable_leaves[i]
            pregrasp = np.asarray(best_pregrasp)
            grasp = np.asarray(best_grasp)
            middle = np.asarray(best_leave)
            print('Using unstable grasps!')
        else:
            raise RuntimeError('No feasible grasp!')
    elif signal == 1:
        pregrasp = np.asarray(approach[0])
        grasp = np.asarray(adjust_conf)
        middle = np.asarray(leave[0])
    elif signal == 2:
        pregrasp = np.asarray(best_pregrasp)
        grasp = np.asarray(best_grasp)
        middle = np.asarray(best_leave)

    # Visualization
    robot_s.fk(component_name, jnt_values=pregrasp)
    robot_s.jaw_to(component_name, 0.14)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    robot_s_meshmodel.attach_to(base)
    robot_s.fk(component_name, jnt_values=grasp)
    robot_s.jaw_to(component_name, 0.14)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    robot_s_meshmodel.attach_to(base)

    # Save planning results
    for i in range(6):
        if pregrasp[i] > math.pi:
            pregrasp[i] = pregrasp[i] - 2 * math.pi
        if pregrasp[i] < -math.pi:
            pregrasp[i] = pregrasp[i] + 2 * math.pi
        if grasp[i] > math.pi:
            grasp[i] = grasp[i] - 2 * math.pi
        if grasp[i] < -math.pi:
            grasp[i] = grasp[i] + 2 * math.pi
        if middle[i] > math.pi:
            middle[i] = middle[i] - 2 * math.pi
        if middle[i] < -math.pi:
            middle[i] = middle[i] + 2 * math.pi
    np.savetxt('../../../../results/pregrasp.txt', pregrasp)
    np.savetxt('../../../../results/grasp.txt', grasp)
    np.savetxt('../../../../results/middle.txt', middle)
    with open('../../../../signals/signal4.txt', 'w') as f:
        f.write('Start robotic manipulation.')

    base.run()
