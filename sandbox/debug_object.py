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
lego = bullet.objects.lego()
# spam = bullet.objects.spam()
# box = bullet.objects.box()
# box_open_top = bullet.objects.long_box_open_top()
tray = bullet.objects.widow200_tray()
# drawer = bullet.objects.drawer()
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
