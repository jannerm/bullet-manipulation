import time
import numpy as np
import pdb

import roboverse.bullet as bullet
import roboverse.devices as devices
import pybullet as p

from roboverse.utils.shapenet_utils import load_shapenet_object, \
    import_shapenet_metadata
obj_path_map, path_scaling_map = import_shapenet_metadata()

bullet.connect()
bullet.setup()

## load meshes
table = bullet.objects.table()
# spam = bullet.objects.spam()
# box = bullet.objects.box()
# box_open_top = bullet.objects.long_box_open_top()
# tray = bullet.objects.widow200_tray()
drawer = bullet.objects.drawer_with_tray_inside()
lifted_long_box_open_top = bullet.objects.lifted_long_box_open_top()

object_names = []

object_path_dict = dict(
    [(obj, path) for obj, path in obj_path_map.items() if obj in object_names])
scaling = dict(
    [(path, path_scaling_map[path]) for _, path in object_path_dict.items()])

for object_name in object_names:
    load_shapenet_object(obj_path_map[object_name], scaling, np.array([.8, -0.07, -.2]), scale_local=0.5)

open = -1
while True:
    time.sleep(0.01)
    bullet.step()
    drawer_pos = p.getBasePositionAndOrientation(drawer)[0]
    print("drawer_pos", drawer_pos)
    joint_names = [bullet.get_joint_info(drawer, j, 'joint_name') for j in range(p.getNumJoints(drawer))]
    link_names = [bullet.get_joint_info(drawer, j, 'link_name') for j in range(p.getNumJoints(drawer))]
    drawer_link_idx = link_names.index('base')
    print("base pos", bullet.get_link_state(drawer, drawer_link_idx, "pos"))
    print("joint_names", joint_names)
    drawer_frame_joint_idx = joint_names.index('base_frame_joint')
    # p.resetJointState(drawer, 1, drawer_pos - np.array([0, -1, 0]))
    p.setJointMotorControl2(drawer, drawer_frame_joint_idx, controlMode=p.POSITION_CONTROL, targetPosition=open, force=10)
    if np.random.uniform() < 0.05:
        open *= -1
