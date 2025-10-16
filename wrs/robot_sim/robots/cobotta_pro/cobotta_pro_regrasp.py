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

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta_pro", enable_cc=True):
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
        self.robotbase.jnts[1]['loc_pos'] = np.array([0, 0, 0.8])
        self.robotbase.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "robotbase.stl")
        self.robotbase.lnks[0]['rgba'] = [.35,.35,.35,1]
        self.robotbase.reinitialize()
        # arm
        arm_homeconf = np.zeros(6)
        arm_homeconf[1] = -math.pi / 6
        arm_homeconf[2] = math.pi / 2
        arm_homeconf[4] = math.pi / 6
        self.arm = cbta.CobottaProArm(pos=self.robotbase.jnts[-1]['gl_posq'],
                                   rotmat=self.robotbase.jnts[-1]['gl_rotmatq'],
                                   homeconf=arm_homeconf,
                                   name='arm', enable_cc=False)
        # ext_hnd
        self.ext_hnd = jl.JLChain(pos=self.arm.jnts[-1]['gl_posq'],
                                  rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                  homeconf=np.zeros(0), name='ext_hnd')
        self.ext_hnd.jnts[1]['loc_pos'] = np.array([0, 0, 0.1215])
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
        self.hnd.lft_outer.lnks[3]['collision_model'].change_cdprimitive_type('box', 0.01)
        self.hnd.rgt_outer.lnks[3]['collision_model'].change_cdprimitive_type('box', 0.01)
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


if __name__ == '__main__':
    import random
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import trimesh
    from manipulation.pick_place_planner import PickPlacePlanner

    # Initialization
    base = wd.World(cam_pos=[-4.3, -3, 2.4], lookat_pos=[0, 0, 0.65])

    gm.gen_frame().attach_to(base)
    robot_s = CobottaPro(enable_cc=True)
    component_name = 'arm'
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

    pos = np.array([0.8, 0.1, 0])
    rot = rm.rotmat_from_axangle([0, 0, 1], math.pi/2)
    backboard.change_cdprimitive_type('box', expand_radius=0.03)
    obstacle_list = [shelf, rack_0, rack_1, rack_2, rack_3, rack_4, rack_5,
                     floor, backboard, sideboard]
    for obstacle in obstacle_list:
        obstacle.set_pos(pos)
        obstacle.set_rotmat(rot)
        obstacle.set_rgba([1, 0, 0, 1])
        obstacle.attach_to(base)
    obstacle_list.remove(shelf)
    for container in [container0, container1, container2, container3, container4]:
        container.set_pos([0, 0.6, 0])
        container.set_rotmat(np.eye(3))
        container.set_rgba([0, 0, 1, 1])
        container.attach_to(base)
        obstacle_list.append(container)
    homomat = np.eye(4)
    homomat[0:3, 3] = [0, -0.5, 1]
    platform = cm.gen_box([0.2, 0.2, 0.02], homomat, [0, 1, 0, 1])
    platform.attach_to(base)
    obstacle_list.append(platform)
    # base.run()

    # Import target object and preplanned grasps
    model = 'bottle'
    object = cm.CollisionModel(f'./meshes/{model}.ply')
    extents = trimesh.load(f'./meshes/{model}.ply').extents
    tcp = np.loadtxt(f'../../../0000_book/preplanned_grasps/tcp_result_{model}.txt')
    jaw = np.loadtxt(f'../../../0000_book/preplanned_grasps/grip_result_{model}.txt')
    grarotmat = np.loadtxt(f'../../../0000_book/preplanned_grasps/rot_result_{model}.txt')
    jaw_width = np.loadtxt(f'../../../0000_book/preplanned_grasps/jaw_width_{model}.txt')

    objpos = [0.7, 0, 1.1]  # [0, -0.5, 0.6]
    objrotmat = np.eye(3)
    object_copy = object.copy()
    object_copy.change_cdprimitive_type('point_cloud', expand_radius=0.001)
    object_copy.set_pos(objpos)
    object_copy.set_rotmat(objrotmat)
    object_copy.set_rgba([.9, .75, .35, 1])
    object_show = [object_copy.copy()]
    object_show[0].attach_to(base)
    # base.run()

    # Plan pick-and-place
    signal = 0
    for index in range(len(tcp)):
        index = random.randint(0, len(tcp) - 1)
        robot_s2 = robot_s.copy()
        planner = PickPlacePlanner(robot_s2)
        gripos = jaw[index]
        grirot = rm.rotmat_from_euler(grarotmat[index][0], grarotmat[index][1], grarotmat[index][2])
        wid = jaw_width[index]
        if wid > 0.085:
            continue
        start_jaw_center_pos = objrotmat.dot(gripos) + objpos
        start_jaw_center_rotmat = objrotmat.dot(grirot)
        start = robot_s2.ik(component_name, start_jaw_center_pos, start_jaw_center_rotmat,
                            robot_s2.get_jnt_values(component_name))
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
                                                            granularity=0.01,
                                                            seed_jnt_values=None,
                                                            type='sink')
        if approach1 is None:
            print("Cannot generate the pick motion!")
            continue
        middle1 = planner.rrtc_planner.plan(component_name=component_name,
                                            start_conf=robot_s.get_jnt_values(component_name),
                                            goal_conf=approach1[0],
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=[],
                                            ext_dist=.1,
                                            max_iter=300)
        if middle1 is None:
            print("Cannot generate the initial motion!")
            continue

        robot_s2.fk(component_name, approach1[-1])
        robot_s2.hold(component_name, object_copy, wid)
        j = 0
        while j < 20:
            j += 1
            mid_obj_pos = [0, -0.5, 1.01 + extents[2] / 2]
            mid_obj_rotmat = rm.rotmat_from_euler(0, 0, random.uniform(-1, 1) * math.pi)
            mid_jaw_center_pos = mid_obj_rotmat.dot(gripos) + mid_obj_pos
            mid_jaw_center_rotmat = mid_obj_rotmat.dot(grirot)
            mid = robot_s2.ik(component_name, mid_jaw_center_pos, mid_jaw_center_rotmat, approach1[-1])
            if mid is None:
                print("IK not solvable for the middle1 grasp!")
                continue
            robot_s2.fk(component_name, mid)
            if robot_s2.is_collided(obstacle_list):
                print("Middle1 grasp is collided!")
                continue
            middle = planner.rrtc_planner.plan(component_name=component_name,
                                                start_conf=approach1[-1],
                                                goal_conf=mid,
                                                obstacle_list=obstacle_list,
                                                otherrobot_list=[],
                                                ext_dist=.1,
                                                max_iter=300)
            if middle is None:
                print("Cannot generate the intermediate motion!")
                continue
            else:
                break
        if j == 20:
            continue

        object_copy.set_pos(mid_obj_pos)
        object_copy.set_rotmat(mid_obj_rotmat)
        i = 0
        for index in range(len(tcp)):
            i += 1
            index = random.randint(0, len(tcp) - 1)
            robot_s3 = robot_s.copy()
            planner = PickPlacePlanner(robot_s3)
            gripos = jaw[index]
            grirot = rm.rotmat_from_euler(grarotmat[index][0], grarotmat[index][1], grarotmat[index][2])
            wid = jaw_width[index]
            if wid > 0.085:
                continue
            start_jaw_center_pos = mid_obj_rotmat.dot(gripos) + mid_obj_pos
            start_jaw_center_rotmat = mid_obj_rotmat.dot(grirot)
            start = robot_s3.ik(component_name, start_jaw_center_pos,
                                start_jaw_center_rotmat, mid)
            if start is None:
                print("IK not solvable for the middle2 grasp!")
                continue
            robot_s3.fk(component_name, start)
            if robot_s3.is_collided(obstacle_list):
                print("Middle2 grasp is collided!")
                continue

            regrasp = planner.rrtc_planner.plan(component_name=component_name,
                                                start_conf=mid,
                                                goal_conf=start,
                                                obstacle_list=obstacle_list + [object_copy],
                                                otherrobot_list=[],
                                                ext_dist=.1,
                                                max_iter=300)
            if regrasp is None:
                print("Cannot generate the regrasp motion!")
                continue

            robot_s3.fk(component_name, start)
            robot_s3.hold(component_name, object_copy, wid)
            j = 0
            while j < 20:
                j += 1
                goal_obj_pos = [0, 0.6, 0.76 + extents[2] / 2] #0.76
                goal_obj_rotmat = rm.rotmat_from_euler(0, 0, random.uniform(-1, 1) * math.pi)
                goal_jaw_center_pos = goal_obj_rotmat.dot(gripos) + goal_obj_pos
                goal_jaw_center_rotmat = goal_obj_rotmat.dot(grirot)
                goal = robot_s3.ik(component_name, goal_jaw_center_pos, goal_jaw_center_rotmat, approach1[-1])
                if goal is None:
                    print("IK not solvable for the final grasp!")
                    continue
                robot_s3.fk(component_name, goal)
                if robot_s3.is_collided(obstacle_list):
                    print("Final grasp is collided!")
                    continue
                approach2 = planner.inik_slvr.gen_rel_linear_motion(component_name=component_name,
                                                                    goal_tcp_pos=goal_jaw_center_pos,
                                                                    goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                    direction=np.array([0, 0, -1]),
                                                                    distance=0.05,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=0.01,
                                                                    seed_jnt_values=None,
                                                                    type='sink')
                if approach2 is None:
                    print("Cannot generate the place motion!")
                    continue
                middle2 = planner.rrtc_planner.plan(component_name=component_name,
                                                   start_conf=start,
                                                   goal_conf=approach2[0],
                                                   obstacle_list=obstacle_list,
                                                   otherrobot_list=[],
                                                   ext_dist=.1,
                                                   max_iter=300)
                if middle2 is None:
                    print("Cannot generate the intermediate motion!")
                    continue
                else:
                    break
            if j == 20:
                continue
            break
        if i == len(tcp):
            object_copy.set_pos(objpos)
            object_copy.set_rotmat(objrotmat)
            continue
        print('Successfully plan a motion!')
        signal = 1
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

    pregrasp = np.asarray(approach1[0])
    grasp = np.asarray(approach1[-1])
    preplace = np.asarray(approach2[0])
    place = np.asarray(approach2[-1])
    for i in range(6):
        if pregrasp[i] > math.pi:
            pregrasp[i] = pregrasp[i] - 2 * math.pi
        if pregrasp[i] < -math.pi:
            pregrasp[i] = pregrasp[i] + 2 * math.pi
        if grasp[i] > math.pi:
            grasp[i] = grasp[i] - 2 * math.pi
        if grasp[i] < -math.pi:
            grasp[i] = grasp[i] + 2 * math.pi
        if preplace[i] > math.pi:
            preplace[i] = preplace[i] - 2 * math.pi
        if preplace[i] < -math.pi:
            preplace[i] = preplace[i] + 2 * math.pi
        if place[i] > math.pi:
            place[i] = place[i] - 2 * math.pi
        if place[i] < -math.pi:
            place[i] = place[i] + 2 * math.pi

    np.savetxt('pregrasp.txt', pregrasp)
    np.savetxt('grasp.txt', grasp)
    np.savetxt('preplace.txt', preplace)
    np.savetxt('place.txt', place)

    path = middle1 + approach1 + middle + regrasp + middle2 + approach2
    robot_s.fk(component_name, jnt_values=path[0])
    robot_s_meshmodel = [robot_s.gen_meshmodel(toggle_tcpcs=False)]
    robot_s_meshmodel[0].attach_to(base)
    counter = [0]
    def animation(path, robot_s_meshmodel, object_show, counter, task):
        if counter[0] < len(middle1 + approach1):
            robot_s_meshmodel[0].detach()
            jnt_values = path[counter[0]]
            robot_s.fk(component_name, jnt_values=jnt_values)
            robot_s_meshmodel[0] = robot_s.gen_meshmodel(toggle_tcpcs=False)
            robot_s_meshmodel[0].attach_to(base)
        elif counter[0] < len(middle1 + approach1 + middle):
            if counter[0] == len(middle1 + approach1):
                object_show[0].detach()
            robot_s_meshmodel[0].detach()
            jnt_values = path[counter[0]]
            robot_s2.fk(component_name, jnt_values=jnt_values)
            robot_s_meshmodel[0] = robot_s2.gen_meshmodel(toggle_tcpcs=False)
            robot_s_meshmodel[0].attach_to(base)
        elif counter[0] < len(middle1 + approach1 + middle + regrasp):
            if counter[0] == len(middle1 + approach1 + middle):
                object_copy.attach_to(base)
            robot_s_meshmodel[0].detach()
            jnt_values = path[counter[0]]
            robot_s.fk(component_name, jnt_values=jnt_values)
            robot_s_meshmodel[0] = robot_s.gen_meshmodel(toggle_tcpcs=False)
            robot_s_meshmodel[0].attach_to(base)
        elif counter[0] < len(path):
            if counter[0] == len(middle1 + approach1 + middle + regrasp):
                object_copy.detach()
            robot_s_meshmodel[0].detach()
            jnt_values = path[counter[0]]
            robot_s3.fk(component_name, jnt_values=jnt_values)
            robot_s_meshmodel[0] = robot_s3.gen_meshmodel(toggle_tcpcs=False)
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