import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.robotiq140.robotiq140 as rtq140
import math

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
# object_box = cm.gen_box(extent=[.02, .06, 1])
# object_box.set_rgba([.7, .5, .3, .7])
# object_box.attach_to(base)
object_bunny = cm.CollisionModel("/home/ch/research/Panasonic/box_pcd/oreo.ply")
object_bunny.set_rgba([.9, .75, .35, .3])
object_bunny.attach_to(base)
# hnd_s
gripper_s = rtq140.Robotiq140()
gripper_s.gen_meshmodel(toggle_jntscs=True, toggle_tcpcs=True).attach_to(base)
grasp_info_list = gpa.plan_grasps(gripper_s, object_bunny, openning_direction = 'loc_y', max_samples=100, angle_between_contact_normals=math.radians(140),
                                      min_dist_between_sampled_contact_points=.002, rotation_interval=math.radians(22.5), contact_offset=.008)
print(len(grasp_info_list))
for grasp_info in grasp_info_list:
    aw_width, gl_jaw_center, gl_jaw_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.fix_to(hnd_pos, hnd_rotmat)
    gripper_s.jaw_to(aw_width)
    gripper_s.gen_meshmodel().attach_to(base)
base.run()