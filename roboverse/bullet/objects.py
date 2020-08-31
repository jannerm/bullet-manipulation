import os
import pdb

import pybullet as p
import pybullet_data as pdata
import math
import numpy as np

from roboverse.bullet.misc import (
  load_urdf,
  deg_to_quat,
)


def loader(*filepath, **defaults):
    filepath = os.path.join(*filepath)

    def fn(*args, **kwargs):
        defaults.update(kwargs)

        if 'deg' in defaults:
          assert 'quat' not in defaults
          defaults['quat'] = deg_to_quat(defaults['deg'])
          del defaults['deg']

        return load_urdf(filepath, **defaults)
    return fn


def widow_loader(name):
    assert name in ['wx200', 'wx250', 'wx250s']
    return loader(
        ROBOT_ASSET_PATH,
        'interbotix_descriptions/urdf/{}.urdf'.format(name),
        pos=[0.6, 0, -0.4],
        deg=[math.pi, math.pi, math.pi],
        scale=1
    )


cur_path = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(cur_path, '../envs/assets')
ROBOT_ASSET_PATH = ASSET_PATH # variable kept for debugging
PDATA_PATH = pdata.getDataPath()
obj_dir = "bullet-objects"

## robots

sawyer = loader(ROBOT_ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
sawyer_invisible = loader(ROBOT_ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro_invisible.urdf')
sawyer_finger_visual_only = loader(ROBOT_ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro_finger_visual_only.urdf')

widowx_200 = widow_loader("wx200")
widowx_250 = widow_loader("wx250")
widowx_250s = widow_loader("wx250s")

## pybullet_data objects

table = loader(PDATA_PATH, 'table/table.urdf',
               pos=[.75, -.2, -1],
               quat=[0, 0, 0.707107, 0.707107],
               scale=1.0)

duck = loader(PDATA_PATH, 'duck_vhacd.urdf',
              pos=[.75, -.4, -.3],
              deg=[0,0,0],
              scale=0.8)

lego = loader(PDATA_PATH, 'lego/lego.urdf',
              pos=[.8, -.1, -.0],
              quat=[0, 0, 1, 0],
              rgba=[1, 0, 0, 1],
              scale=1.2)


## custom objects

bowl = loader(ASSET_PATH, os.path.join(obj_dir, "bowl", "bowl.urdf"),
              pos=[.75, 0, -.3],
              scale=0.25)

lid = loader(ASSET_PATH, os.path.join(obj_dir, "bowl", "lid.urdf"),
              pos=[.75, 0, -.3],
              scale=0.25)

cube = loader(ASSET_PATH, os.path.join(obj_dir, "cube", "cube.urdf"),
              pos=[.75, -.4, -.3],
              scale=0.05)

spam = loader(ASSET_PATH, os.path.join(obj_dir, "spam", "spam.urdf"),
              pos=[.75, -.4, -.3],
              deg=[90,0,-90],
              scale=0.025)

## tray

tray = loader('', os.path.join("tray", "tray.urdf"),
              pos=[0.70, 0.15, -0.36],
              deg=[0, 0, 0],
              scale=0.75)

box = loader(ASSET_PATH, os.path.join(obj_dir, "box", "box.urdf"),
                # pos=[0.8, 0.075, -.35],
                pos=[0.8, 0.075, -.35],
                scale=0.125)

box_open_top = loader(ASSET_PATH, os.path.join(obj_dir, "box_open_top", "box_open_top.urdf"),
              pos=[0.8, 0.01, -.345],
              scale=0.1)

long_box_open_top = loader(ASSET_PATH, os.path.join(obj_dir, "box_open_top", "long_box_open_top.urdf"),
              pos=[0.8225, 0.01, -.345],
              scale=0.1)

lifted_long_box_open_top_center_pos = [0.6925, -0.25, -.345]

lifted_long_box_open_top = loader(ASSET_PATH, os.path.join(obj_dir, "box_open_top", "long_box_open_top.urdf"),
              pos=lifted_long_box_open_top_center_pos, # old: [0.8425, 0.05, -.295]
              scale=0.1)

small_tray_center_pos = [0.6425, -0.2, -.36]
small_object_tray = loader(ASSET_PATH, os.path.join(obj_dir, "tray", "small_tray.urdf"),
              pos=small_tray_center_pos,
              scale=0.2)

drawer_pos = [0.8425, 0.05, -.34]
drawer = loader(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer.urdf"),
              pos=drawer_pos,
              scale=0.1)

drawer_with_tray_inside = loader(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_with_tray_inside.urdf"),
              pos=drawer_pos,
              scale=0.1)

drawer_bottom_pos = list(np.array(drawer_pos) + np.array([0, 0.03, 0]))
drawer_bottom = loader(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_with_tray_inside.urdf"),
              pos=drawer_bottom_pos,
              scale=0.1)

drawer_top_pos = list(np.array(drawer_bottom_pos) + np.array([0, 0, 0.07]))
# drawer_top = loader(ASSET_PATH, os.path.join(obj_dir, "drawer", "left_side_handle_drawer.urdf"),
#               pos=drawer_top_pos,
#               scale=0.1)

drawer_no_handle = loader(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_no_handle.urdf"),
              pos=drawer_top_pos,
              scale=0.1)

# test_box = loader(ASSET_PATH, os.path.join(obj_dir, "box_open_top", "box_open_top.urdf"),
#               pos=[0.825, .05, -.33], #low: [0.775, -.03, -.345], #high: [0.825, .05, -.33]
#               scale=0.01)

widow200_tray = loader(ASSET_PATH, os.path.join(obj_dir, "tray", "tray.urdf"),
              pos=[0.8, -0.05, -0.36],
              deg=[0, 0, 0],
              scale=0.5)

widow200_hidden_tray = loader(ASSET_PATH, os.path.join(obj_dir, "tray", "tray.urdf"),
              pos=[0.82, -0.05, -0.5],
              deg=[0, 0, 0],
              scale=0.5)
