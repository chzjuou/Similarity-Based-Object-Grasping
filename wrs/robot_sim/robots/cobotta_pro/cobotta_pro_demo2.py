import os
import math
import time

import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.cobotta_pro_arm.cobotta_pro_arm as cbta
import robot_sim.end_effectors.gripper.cobotta_gripper.cobotta_gripper as cbtg
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq
import robot_sim.robots.robot_interface as ri


class CobottaPro(ri.RobotInterface):

    def __init__(self, pos=np.array([0, 0, -0.8]), rotmat=np.eye(3), name="cobotta_pro", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # robotstand
        self.robotstand = jl.JLChain(pos=pos,
                                    rotmat=rotmat,
                                    homeconf=np.zeros(0),
                                    name='robotstand')
        self.robotstand.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.robotstand.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "robotstand.stl")
        self.robotstand.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.robotstand.reinitialize()
        # robotbase
        self.robotbase = jl.JLChain(pos=self.robotstand.jnts[-1]['gl_posq'],
                                    rotmat=self.robotstand.jnts[-1]['gl_rotmatq'],
                                    homeconf=np.zeros(0),
                                    name='robotbase')
        self.robotbase.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.robotbase.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "robotbase.stl")
        self.robotbase.lnks[0]['rgba'] = [.35,.35,.35,1]
        self.robotbase.reinitialize()
        # arm
        arm_homeconf = np.zeros(6)
        arm_homeconf[1] = -math.pi / 6
        arm_homeconf[2] = math.pi / 2
        arm_homeconf[4] = math.pi / 6
        self.arm = cbta.CobottaProArm(pos=self.robotbase.jnts[-1]['gl_posq'] + np.array([0, 0, 0.8]),
                                   rotmat=self.robotbase.jnts[-1]['gl_rotmatq'],
                                   homeconf=arm_homeconf,
                                   name='arm', enable_cc=False)
        # ext_hnd
        self.ext_hnd = jl.JLChain(pos=self.arm.jnts[-1]['gl_posq'],
                                  rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                  homeconf=np.zeros(0), name='ext_hnd')
        self.ext_hnd.jnts[1]['loc_pos'] = np.array([0, 0, 0.17])
        self.ext_hnd.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "robothand.stl"),
            cdprimit_type="box", expand_radius=.01)
        self.ext_hnd.lnks[0]['rgba'] = [1,1,1,1]
        self.ext_hnd.reinitialize()
        # camera
        self.camera = jl.JLChain(pos=self.arm.jnts[-1]['gl_posq'],
                                  rotmat=np.dot(self.arm.jnts[-1]['gl_rotmatq'],
                                            rm.rotmat_from_axangle([0,0,1], math.pi)),
                                  homeconf=np.zeros(0), name='camera')
        self.camera.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "camera.stl"),
            cdprimit_type="box", expand_radius=.01)
        self.camera.lnks[0]['rgba'] = [1, 1, 1, 1]
        self.camera.reinitialize()
        # gripper cbtg.CobottaGripper
        self.hnd = rtq.Robotiq85(pos=self.ext_hnd.jnts[-1]['gl_posq'],
                                       rotmat=self.ext_hnd.jnts[-1]['gl_rotmatq'],
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

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        # self.hnd.lft_outer.lnks[3]['collision_model'].change_cdprimitive_type('box', 0.01)
        # self.hnd.rgt_outer.lnks[3]['collision_model'].change_cdprimitive_type('box', 0.01)
        self.cc.add_cdlnks(self.robotstand, [0])
        self.cc.add_cdlnks(self.robotbase, [0])
        self.cc.add_cdlnks(self.ext_hnd, [0])
        self.cc.add_cdlnks(self.camera, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        # self.cc.add_cdlnks(self.hnd.jlc, [0, 1, 2])
        self.cc.add_cdlnks(self.hnd.lft_outer, [0, 1, 2, 3])
        self.cc.add_cdlnks(self.hnd.rgt_outer, [1, 2, 3])
        activelist = [self.robotstand.lnks[0],
                      self.robotbase.lnks[0],
                      self.ext_hnd.lnks[0],
                      self.camera.lnks[0],
                      self.arm.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      # self.hnd.jlc.lnks[0],
                      # self.hnd.jlc.lnks[1],
                      # self.hnd.jlc.lnks[2]
                      self.hnd.lft_outer.lnks[0],
                      self.hnd.lft_outer.lnks[1],
                      self.hnd.lft_outer.lnks[2],
                      self.hnd.lft_outer.lnks[3],
                      self.hnd.rgt_outer.lnks[1],
                      self.hnd.rgt_outer.lnks[2],
                      self.hnd.rgt_outer.lnks[3]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.robotstand.lnks[0],
                    self.robotbase.lnks[0],
                    self.arm.lnks[0],
                    self.arm.lnks[1]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.ext_hnd.lnks[0],
                    self.camera.lnks[0]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.robotstand.lnks[0],
                    self.robotbase.lnks[0],
                    self.arm.lnks[0],
                    self.arm.lnks[1]]
        intolist = [self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3]]
                    # self.hnd.jlc.lnks[0],
                    # self.hnd.jlc.lnks[1],
                    # self.hnd.jlc.lnks[2]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.ext_hnd.lnks[0],
                    self.camera.lnks[0]]
        intolist = [self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5]]
        self.cc.set_cdpair(fromlist, intolist)
        # TODO is the following update needed?
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.robotstand.fix_to(pos=pos, rotmat=rotmat)
        self.robotbase.fix_to(pos=self.robotstand.jnts[-1]['gl_posq'], rotmat=self.robotstand.jnts[-1]['gl_rotmatq'])
        self.arm.fix_to(pos=self.robotbase.jnts[-1]['gl_posq'], rotmat=self.robotbase.jnts[-1]['gl_rotmatq'])
        self.ext_hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        self.camera.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.ext_hnd.jnts[-1]['gl_posq'], rotmat=self.ext_hnd.jnts[-1]['gl_rotmatq'])
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
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            self.ext_hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
            self.camera.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
            self.hnd.fix_to(pos=self.ext_hnd.jnts[-1]['gl_posq'], rotmat=self.ext_hnd.jnts[-1]['gl_rotmatq'])
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

    def get_gl_tcp(self, manipulator_name='arm'):
        return self.manipulator_dict[manipulator_name].get_gl_tcp()

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
        intolist = [self.arm.lnks[0],
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
                       name='cobotta_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.robotbase.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
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
                      name='cobotta_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.robotstand.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                     tcp_loc_pos=tcp_loc_pos,
                                     tcp_loc_rotmat=tcp_loc_rotmat,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.robotbase.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.ext_hnd.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.camera.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
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
        self.hnd.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel

    def ikfast(self, pos, rotmat, seedjnt):
        mat = np.c_[rotmat, pos]
        listToStr = ' '.join(str(item) for innerlist in mat for item in innerlist)
        os.system(f'./ikfast {listToStr}')
        solutions = np.loadtxt('solutions.txt')
        distance = []
        for solution in solutions:
            distance.append(math.dist(solution, seedjnt))
        best_solution = solutions[np.argmin(distance)]
        return best_solution


if __name__ == '__main__':
    import random
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from manipulation.pick_place_planner import PickPlacePlanner
    import open3d as o3d
    import pymeshlab.pmeshlab as mlab

    # Initialization
    base = wd.World(cam_pos=[-4.3, -3, 2.4], lookat_pos=[0, 0, 0.65])

    gm.gen_frame().attach_to(base)
    robot_s = CobottaPro(enable_cc=True)
    component_name = 'arm'
    init_jnts = np.array([2.87498, 0.0626788, -1.7573, 0.872849, -0.364781, 2.1801])
    robot_s.fk(component_name, init_jnts)
    robot_s.jaw_to(component_name, 0.085)
    # robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)

    shelf = cm.CollisionModel('./meshes/shelf.stl') # shelf_v1, shelf_v2
    rack_0 = cm.CollisionModel('./meshes/rack_0.stl')
    rack_1 = cm.CollisionModel('./meshes/rack_1.stl')
    rack_2 = cm.CollisionModel('./meshes/rack_2.stl')
    rack_3 = cm.CollisionModel('./meshes/rack_3.stl')
    rack_4 = cm.CollisionModel('./meshes/rack_4.stl')
    rack_5 = cm.CollisionModel('./meshes/rack_5.stl')
    floor = cm.CollisionModel('./meshes/floor.stl')
    backboard = cm.CollisionModel('./meshes/backboard.stl')
    sideboard = cm.CollisionModel('./meshes/sideboard.stl')
    container0 = cm.CollisionModel('./meshes/container0.stl')
    container1 = cm.CollisionModel('./meshes/container1.stl')
    container2 = cm.CollisionModel('./meshes/container2.stl')
    container3 = cm.CollisionModel('./meshes/container3.stl')
    container4 = cm.CollisionModel('./meshes/container4.stl')
    right_wall = cm.CollisionModel('./meshes/right_wall.stl')
    back_wall = cm.CollisionModel('./meshes/back_wall.stl')
    front_wall = cm.CollisionModel('./meshes/front_wall.stl')

    pos = np.array([0.88, -0, 0])
    rot = rm.rotmat_from_axangle([0, 0, 1], math.pi/2)
    backboard.change_cdprimitive_type('box', expand_radius=0.03)
    obstacle_list = [shelf, rack_0, rack_1, rack_2, rack_3, rack_4, rack_5,
                     floor, backboard, sideboard]
    for obstacle in obstacle_list:
        obstacle.set_pos(pos + np.array([0,0,-0.88]))
        obstacle.set_rotmat(rot)
        obstacle.set_rgba([1, 0, 0, 1])
        obstacle.attach_to(base)
    obstacle_list.remove(shelf)
    obstacle_list.extend([right_wall, back_wall])
    rack_2.set_pos(pos + [0, 0, -0.87] + [-0.325, 0, 0])  # -0.325
    for container in [container0, container1, container2, container3, container4]:
        container.set_pos(np.array([0, 0.6, -0.8]))
        container.set_rotmat(np.eye(3))
        container.set_rgba([0, 0, 1, 1])
        container.attach_to(base)
        obstacle_list.append(container)
    # base.run()

    # Import target object and preplanned grasps
    with open('candidates.txt') as f:
        candidates = f.read()
    candidates = candidates.split('\n')
    print(candidates)
    transformation = np.load('matrix.npy')

    # Plan pick-and-place
    signal = 0
    for i in range(len(candidates)-1):
        robot_s.fk(component_name, init_jnts)
        tcp_pos = robot_s.arm.jnts[-1]['gl_posq']
        tcp_rotmat = robot_s.arm.jnts[-1]['gl_rotmatq']
        cali_pos = np.array([-0.115, -0.012, 0.00]) # -0.105, 0.033, 0.00
        cali_rotmat = rm.rotmat_from_euler(0.02, -0.05, -1.53)
        cam_pos = tcp_rotmat.dot(cali_pos) + tcp_pos
        cam_rotmat = tcp_rotmat.dot(cali_rotmat)
        #gm.gen_frame(cam_pos, cam_rotmat).attach_to(base)
        mat1 = np.c_[cam_rotmat, cam_pos]
        mat1 = np.r_[mat1, np.array([[0, 0, 0, 1]])]
        model = candidates[i]
        model2obj = transformation[i]
        object = cm.CollisionModel(f'/home/ch/research/irex/model_database/{model}.ply')
        tcp = np.loadtxt(f'../../../0000_book/preplanned_grasps/tcp_result_{model}c.txt')
        jaw = np.loadtxt(f'../../../0000_book/preplanned_grasps/grip_result_{model}c.txt')
        grarotmat = np.loadtxt(f'../../../0000_book/preplanned_grasps/rot_result_{model}c.txt')
        jaw_width = np.loadtxt(f'../../../0000_book/preplanned_grasps/jaw_width_{model}c.txt')

        model2obj = np.linalg.inv(model2obj)
        trapos = model2obj[0:3, 3]
        trarotmat = model2obj[0:3, 0:3]
        mat2 = np.c_[trarotmat, trapos]
        mat2 = np.r_[mat2, np.array([[0, 0, 0, 1]])]
        mat = np.dot(mat1, mat2)
        objpos = mat[0:3, 3]
        objrotmat = mat[0:3, 0:3]

        object_copy = object.copy()
        object_copy.change_cdprimitive_type('point_cloud', expand_radius=0.001)
        object_copy.set_pos(objpos + [-0.31, 0, 0]) # -0.325
        object_copy.set_rotmat(objrotmat)
        object_copy.set_rgba([1, 1, 1, .5])  # .9, .75, .35, .5
        object_show = object_copy.copy()
        object_show.attach_to(base)

        pcd = o3d.io.read_point_cloud('/home/ch/research/irex/downsample.ply')
        pcd.transform(mat1)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        target = cm.CollisionModel(mesh)
        target.change_cdprimitive_type('point_cloud', expand_radius=0.001)
        target.set_pos(target.get_pos() + [-0.31, 0, 0])

        ms = mlab.MeshSet()
        ms.load_new_mesh('/home/ch/research/irex/downsample.ply')
        ms.compute_normal_for_point_clouds(k=10, smoothiter=5)
        ms.save_current_mesh('object.ply', save_vertex_normal=True)
        pcd = o3d.io.read_point_cloud('object.ply')
        pcd.transform(mat1)
        pcd.translate([-0.31, 0, 0]) #-0.325
        gm.gen_pointcloud(pcd.points, [[1, 1, 0, 1]], 3).attach_to(base)
        pcd = o3d.io.read_point_cloud('object.ply')
        pcd.transform(np.linalg.inv(mat2))
        extents = pcd.get_axis_aligned_bounding_box().get_extent()

        pcdx = o3d.io.read_point_cloud('/home/ch/research/irex/downsample2.ply')
        pcdx.transform(mat1)
        pcdx.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdx, o3d.utility.DoubleVector(radii))
        mesh.translate([-0.31, 0, 0]) #-0.325
        obstacle = cm.CollisionModel(mesh)
        obstacle.change_cdprimitive_type('point_cloud', expand_radius=0.001)
        obstacle.attach_to(base)
        obstacle_list.append(obstacle)
        # base.run()


        pregrasp = np.array([75, 15, -123, 87, -104, 75]) / 180 * math.pi
        robot_s.fk(component_name, pregrasp)
        # robot_s.gen_meshmodel().attach_to(base)
        '''
        for index in range(len(tcp)):
            # index = random.randint(0, len(tcp) - 1)
            robot_s2 = robot_s.copy()
            planner = PickPlacePlanner(robot_s2)
            grirot = rm.rotmat_from_euler(grarotmat[index][0], grarotmat[index][1], grarotmat[index][2])
            gripos = jaw[index] - grirot.dot(np.array([0, 0, .315]))
            wid = jaw_width[index]
            if wid > 0.085:
                continue

            start_jaw_center_pos = objrotmat.dot(gripos) + objpos + [-0.31, 0, 0] #-0.325
            start_jaw_center_rotmat = objrotmat.dot(grirot)
            start = robot_s2.ikfast(start_jaw_center_pos, start_jaw_center_rotmat, pregrasp)
            if start is None:
                print("IK not solvable for the initial grasp!")
                continue
            robot_s2.fk(component_name, start)
            robot_s2.jaw_to(component_name, wid)
            if robot_s2.is_collided(obstacle_list):
                print("Initial grasp is collided!")
                continue

            approach1 = planner.inik_slvr.gen_rel_linear_motion(component_name=component_name,
                                                                goal_tcp_pos=start_jaw_center_pos,
                                                                goal_tcp_rotmat=start_jaw_center_rotmat,
                                                                direction=start_jaw_center_rotmat[:, 2],
                                                                distance=0.05,
                                                                obstacle_list=obstacle_list,
                                                                granularity=0.02,
                                                                seed_jnt_values=start,
                                                                type='sink')

            if approach1 is None or (approach1[0] == approach1[-1]).all():
                print("Cannot generate the pick motion!")
                continue

            middle1 = planner.rrtc_planner.plan(component_name=component_name,
                                                start_conf=pregrasp,
                                                goal_conf=approach1[0],
                                                obstacle_list=obstacle_list + [front_wall] + [target],
                                                otherrobot_list=[],
                                                ext_dist=.8,
                                                max_iter=300)

            middle1_show = planner.rrtc_planner.plan(component_name=component_name,
                                                start_conf=pregrasp,
                                                goal_conf=approach1[0],
                                                obstacle_list=obstacle_list + [front_wall] + [target],
                                                otherrobot_list=[],
                                                ext_dist=.05,
                                                max_iter=300)

            if middle1 is None or middle1_show is None or len(middle1) > 10:
                print("Cannot generate the initial motion!")
                continue

            robot_s2.fk(component_name, approach1[-1])
            robot_s2.hold(component_name, target, wid)
            j = 0
            while j < 50:
                j += 1
                goal_obj_pos = [0.56, -0.15, 0.39 + extents[2] / 2] #-0.18
                goal_obj_rotmat = rm.rotmat_from_euler(random.randint(0, 1) * math.pi, 0, random.uniform(-1, 1) * math.pi) #random.randint(0, 1) *
                goal_jaw_center_pos = goal_obj_rotmat.dot(gripos) + goal_obj_pos
                goal_jaw_center_rotmat = goal_obj_rotmat.dot(grirot)
                goal = robot_s2.ikfast(goal_jaw_center_pos, goal_jaw_center_rotmat, start)
                if goal is None:
                    print("IK not solvable for the final grasp!")
                    continue
                robot_s2.fk(component_name, goal)
                if robot_s2.is_collided(obstacle_list):
                    print("Final grasp is collided!")
                    continue
                approach2 = planner.inik_slvr.gen_rel_linear_motion(component_name=component_name,
                                                                    goal_tcp_pos=goal_jaw_center_pos,
                                                                    goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                    direction=np.array([0, 0, -1]),
                                                                    distance=0.05,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=0.02,
                                                                    seed_jnt_values=None,
                                                                    type='sink')
                if approach2 is None or (approach2[0] == approach2[-1]).all():
                    print("Cannot generate the place motion!")
                    continue

                middle2 = planner.rrtc_planner.plan(component_name=component_name,
                                                   start_conf=approach1[-1],
                                                   goal_conf=approach2[0],
                                                   obstacle_list=obstacle_list + [front_wall],
                                                   otherrobot_list=[],
                                                   ext_dist=.8,
                                                   max_iter=300)

                middle2_show = planner.rrtc_planner.plan(component_name=component_name,
                                                    start_conf=approach1[-1],
                                                    goal_conf=approach2[0],
                                                    obstacle_list=obstacle_list + [front_wall],
                                                    otherrobot_list=[],
                                                    ext_dist=.1,
                                                    max_iter=300)

                if middle2 is None or middle2_show is None or len(middle2) > 10:
                    print("Cannot generate the intermediate motion!")
                    continue

                else:
                    break
            if j == 50:
                continue

            print('Successfully plan a motion!')
            signal = 1
            break
        if signal == 1:
            break

    if signal == 0:
        raise RuntimeError('No feasible grasp!')
    '''
        obstacle_list.remove(obstacle)

    # Plan second pick-and-place
    with open('candidates2.txt') as f:
        candidates = f.read()
    candidates = candidates.split('\n')
    print(candidates)
    transformation = np.load('matrix2.npy')
    signal = 0
    for i in range(len(candidates) - 1):
        robot_s.fk(component_name, init_jnts)
        tcp_pos = robot_s.arm.jnts[-1]['gl_posq']
        tcp_rotmat = robot_s.arm.jnts[-1]['gl_rotmatq']
        cali_pos = np.array([-0.115, -0.012, 0.00])  # -0.105, 0.033, 0.00
        cali_rotmat = rm.rotmat_from_euler(0.02, -0.05, -1.53)
        cam_pos = tcp_rotmat.dot(cali_pos) + tcp_pos
        cam_rotmat = tcp_rotmat.dot(cali_rotmat)
        # gm.gen_frame(cam_pos, cam_rotmat).attach_to(base)
        mat3 = np.c_[cam_rotmat, cam_pos]
        mat3 = np.r_[mat3, np.array([[0, 0, 0, 1]])]
        model = candidates[i]
        model2obj = transformation[i]
        object = cm.CollisionModel(f'/home/ch/research/irex/model_database/{model}.ply')
        tcp = np.loadtxt(f'../../../0000_book/preplanned_grasps/tcp_result_{model}c.txt')
        jaw = np.loadtxt(f'../../../0000_book/preplanned_grasps/grip_result_{model}c.txt')
        grarotmat = np.loadtxt(f'../../../0000_book/preplanned_grasps/rot_result_{model}c.txt')
        jaw_width = np.loadtxt(f'../../../0000_book/preplanned_grasps/jaw_width_{model}c.txt')

        model2obj = np.linalg.inv(model2obj)
        trapos = model2obj[0:3, 3]
        trarotmat = model2obj[0:3, 0:3]
        mat4 = np.c_[trarotmat, trapos]
        mat4 = np.r_[mat4, np.array([[0, 0, 0, 1]])]
        mat = np.dot(mat3, mat4)
        objpos = mat[0:3, 3]
        objrotmat = mat[0:3, 0:3]

        object_copy = object.copy()
        object_copy.change_cdprimitive_type('point_cloud', expand_radius=0.001)
        object_copy.set_pos(objpos + [-0.31, 0, 0])  # -0.325
        object_copy.set_rotmat(objrotmat)
        object_copy.set_rgba([1, 1, 1, .5])  # .9, .75, .35, .5
        object_show = object_copy.copy()
        object_show.attach_to(base)

        pcd = o3d.io.read_point_cloud('/home/ch/research/irex/downsample2.ply')
        pcd.transform(mat3)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        target = cm.CollisionModel(mesh)
        target.change_cdprimitive_type('point_cloud', expand_radius=0.001)
        target.set_pos(target.get_pos() + [-0.31, 0, 0])

        ms = mlab.MeshSet()
        ms.load_new_mesh('/home/ch/research/irex/downsample2.ply')
        ms.compute_normal_for_point_clouds(k=10, smoothiter=5)
        ms.save_current_mesh('object.ply', save_vertex_normal=True)
        pcd = o3d.io.read_point_cloud('object.ply')
        pcd.transform(mat3)
        pcd.translate([-0.31, 0, 0])  # -0.325
        gm.gen_pointcloud(pcd.points, [[1, 1, 0, 1]], 3).attach_to(base)
        pcd = o3d.io.read_point_cloud('object.ply')
        pcd.transform(np.linalg.inv(mat4))
        extents = pcd.get_axis_aligned_bounding_box().get_extent()

        pcdx = o3d.io.read_point_cloud('/home/ch/research/irex/downsample.ply')
        pcdx.transform(np.linalg.inv(mat2))
        pcdx.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdx, o3d.utility.DoubleVector(radii))
        #mesh.translate([-0.31, 0, 0])  # -0.325
        obstacle = cm.CollisionModel(mesh)
        #obstacle.set_pos(np.array([0.56, -0.15, 0.39 + extents[2] / 2]))
        obstacle.change_cdprimitive_type('point_cloud', expand_radius=0.001)
        obstacle.attach_to(base)
        obstacle_list.append(obstacle)

        for index in range(len(tcp)):
            # index = random.randint(0, len(tcp) - 1)
            robot_s2 = robot_s.copy()
            planner = PickPlacePlanner(robot_s2)
            grirot = rm.rotmat_from_euler(grarotmat[index][0], grarotmat[index][1], grarotmat[index][2])
            gripos = jaw[index] - grirot.dot(np.array([0, 0, .315]))
            wid = jaw_width[index]
            if wid > 0.085:
                continue

            start_jaw_center_pos = objrotmat.dot(gripos) + objpos + [-0.31, 0, 0]  # -0.325
            start_jaw_center_rotmat = objrotmat.dot(grirot)
            start = robot_s2.ikfast(start_jaw_center_pos, start_jaw_center_rotmat, pregrasp)
            if start is None:
                print("IK not solvable for the initial grasp!")
                continue
            robot_s2.fk(component_name, start)
            robot_s2.jaw_to(component_name, wid)
            if robot_s2.is_collided(obstacle_list):
                print("Initial grasp is collided!")
                continue
            robot_s2.gen_meshmodel().attach_to(base)
            base.run()

            approach3 = planner.inik_slvr.gen_rel_linear_motion(component_name=component_name,
                                                                goal_tcp_pos=start_jaw_center_pos,
                                                                goal_tcp_rotmat=start_jaw_center_rotmat,
                                                                direction=start_jaw_center_rotmat[:, 2],
                                                                distance=0.05,
                                                                obstacle_list=obstacle_list,
                                                                granularity=0.02,
                                                                seed_jnt_values=start,
                                                                type='sink')

            if approach3 is None or (approach3[0] == approach3[-1]).all():
                print("Cannot generate the pick motion!")
                continue

            middle1 = planner.rrtc_planner.plan(component_name=component_name,
                                                start_conf=approach2[-1],
                                                goal_conf=approach3[0],
                                                obstacle_list=obstacle_list + [front_wall] + [target],
                                                otherrobot_list=[],
                                                ext_dist=.8,
                                                max_iter=300)

            middle1_show = planner.rrtc_planner.plan(component_name=component_name,
                                                     start_conf=pregrasp,
                                                     goal_conf=approach3[0],
                                                     obstacle_list=obstacle_list + [front_wall] + [target],
                                                     otherrobot_list=[],
                                                     ext_dist=.05,
                                                     max_iter=300)

            if middle1 is None or middle1_show is None or len(middle1) > 10:
                print("Cannot generate the initial motion!")
                continue

            robot_s2.fk(component_name, approach3[-1])
            robot_s2.hold(component_name, target, wid)
            j = 0
            while j < 50:
                j += 1
                goal_obj_pos = [0.56, -0.18, 0.385 + extents[2] / 2]  # -0.14
                goal_obj_rotmat = rm.rotmat_from_euler(random.randint(0, 1) * math.pi, 0,
                                                       random.uniform(-1, 1) * math.pi)  # random.randint(0, 1) *
                goal_jaw_center_pos = goal_obj_rotmat.dot(gripos) + goal_obj_pos
                goal_jaw_center_rotmat = goal_obj_rotmat.dot(grirot)
                goal = robot_s2.ikfast(goal_jaw_center_pos, goal_jaw_center_rotmat, start)
                if goal is None:
                    print("IK not solvable for the final grasp!")
                    continue
                robot_s2.fk(component_name, goal)
                if robot_s2.is_collided(obstacle_list):
                    print("Final grasp is collided!")
                    continue
                approach2 = planner.inik_slvr.gen_rel_linear_motion(component_name=component_name,
                                                                    goal_tcp_pos=goal_jaw_center_pos,
                                                                    goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                    direction=np.array([0, 0, -1]),
                                                                    distance=0.05,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=0.02,
                                                                    seed_jnt_values=None,
                                                                    type='sink')
                if approach2 is None or (approach2[0] == approach2[-1]).all():
                    print("Cannot generate the place motion!")
                    continue

                middle2 = planner.rrtc_planner.plan(component_name=component_name,
                                                    start_conf=approach1[-1],
                                                    goal_conf=approach2[0],
                                                    obstacle_list=obstacle_list + [front_wall],
                                                    otherrobot_list=[],
                                                    ext_dist=.8,
                                                    max_iter=300)

                middle2_show = planner.rrtc_planner.plan(component_name=component_name,
                                                         start_conf=approach1[-1],
                                                         goal_conf=approach2[0],
                                                         obstacle_list=obstacle_list + [front_wall],
                                                         otherrobot_list=[],
                                                         ext_dist=.1,
                                                         max_iter=300)

                if middle2 is None or middle2_show is None or len(middle2) > 10:
                    print("Cannot generate the intermediate motion!")
                    continue
                else:
                    break
            if j == 50:
                continue

            print('Successfully plan a motion!')
            signal = 1
            break
        if signal == 1:
            break

    if signal == 0:
        raise RuntimeError('No feasible grasp!')

    # robot_s.fk(component_name, jnt_values=approach1[0])
    # robot_s.jaw_to(component_name, 0.085)
    # robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    # robot_s_meshmodel.attach_to(base)
    # robot_s.fk(component_name, jnt_values=approach1[-1])
    # robot_s.jaw_to(component_name, wid)
    # robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    # robot_s_meshmodel.attach_to(base)
    # robot_s2.fk(component_name, jnt_values=approach2[0])
    # robot_s_meshmodel = robot_s2.gen_meshmodel(toggle_tcpcs=False)
    # robot_s_meshmodel.attach_to(base)
    # robot_s2.fk(component_name, jnt_values=approach2[-1])
    # robot_s_meshmodel = robot_s2.gen_meshmodel(toggle_tcpcs=False)
    # robot_s_meshmodel.attach_to(base)

    # np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/grasp.txt', approach1)
    # np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/place.txt', approach2)
    print(len(middle1), len(middle2))
    np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/grasp.txt', np.concatenate([middle1, approach1]))
    np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/place.txt', np.concatenate([middle2, approach2]))

    poslist = []
    rotlist = []
    for jnt in np.concatenate([middle1, approach1]):
        robot_s.fk(component_name, jnt)
        poslist.append(robot_s.arm.jnts[-1]['gl_posq'])
        rot = rm.rotmat_to_euler(robot_s.arm.jnts[-1]['gl_rotmatq'])
        rot = rm.quaternion_from_euler(rot[0], rot[1], rot[2])
        rotlist.append([rot[1], rot[2], rot[3], rot[0]])
        # robot_s.gen_meshmodel().attach_to(base)
    np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/grasp_pos.txt', poslist)
    np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/grasp_rot.txt', rotlist)

    poslist = []
    rotlist = []
    for jnt in np.concatenate([middle2, approach2]):
        robot_s.fk(component_name, jnt)
        poslist.append(robot_s.arm.jnts[-1]['gl_posq'])
        rot = rm.rotmat_to_euler(robot_s.arm.jnts[-1]['gl_rotmatq'])
        rot = rm.quaternion_from_euler(rot[0], rot[1], rot[2])
        rotlist.append([rot[1], rot[2], rot[3], rot[0]])
        # robot_s.gen_meshmodel().attach_to(base)
    np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/place_pos.txt', poslist)
    np.savetxt('/home/ch/catkin_ws/src/cobotta_tutorials/scripts/place_rot.txt', rotlist)

    path = middle1_show + approach1 + middle2_show + approach2
    robot_s.fk(component_name, jnt_values=path[0])
    robot_s_meshmodel = [robot_s.gen_meshmodel(toggle_tcpcs=False)]
    object_show = [object_show]
    robot_s_meshmodel[0].attach_to(base)
    robot_s2.fk(component_name, approach1[-1])
    robot_s2.hold(component_name, object_copy, wid)
    counter = [0]
    def animation(path, robot_s_meshmodel, object_show, counter, task):
        if counter[0] < len(middle1_show + approach1):
            robot_s_meshmodel[0].detach()
            jnt_values = path[counter[0]]
            robot_s.fk(component_name, jnt_values=jnt_values)
            robot_s_meshmodel[0] = robot_s.gen_meshmodel(toggle_tcpcs=False)
            robot_s_meshmodel[0].attach_to(base)
        elif counter[0] < len(path):
            if counter[0] == len(middle1_show + approach1):
                object_show[0].detach()
            robot_s_meshmodel[0].detach()
            jnt_values = path[counter[0]]
            robot_s2.fk(component_name, jnt_values=jnt_values)
            robot_s_meshmodel[0] = robot_s2.gen_meshmodel(toggle_tcpcs=False)
            robot_s_meshmodel[0].attach_to(base)
        else:
            counter[0] = 0
            object_show[0].attach_to(base)
            robot_s_meshmodel[0].detach()
            robot_s.fk(component_name, jnt_values=path[0])
            robot_s_meshmodel[0] = robot_s.gen_meshmodel(toggle_tcpcs=False)
            robot_s_meshmodel[0].attach_to(base)
        counter[0] += 1
        return task.again

    taskMgr.doMethodLater(0.05, animation, 'PickAndPlace',
                           extraArgs=[path, robot_s_meshmodel, object_show, counter], appendTask=True)

    base.run()