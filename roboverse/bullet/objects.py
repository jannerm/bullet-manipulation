import os
import pdb

import pybullet as p
import pybullet_data as pdata
import numpy as np
import math

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

cur_path = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(cur_path, '../envs/assets')
PDATA_PATH = pdata.getDataPath()


## robots

sawyer = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
sawyer_finger_visual_only = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro_finger_visual_only.urdf')
widow_downwards = loader(ASSET_PATH, 'widowx/widowxdownwards.urdf', pos=[0.7, 0, 0.1])
widow = loader(ASSET_PATH, 'widowx/widowx.urdf', pos=[0.7, 0, -0.4])
widowx_200 = loader(ASSET_PATH, 'interbotix_descriptions/urdf/wx200.urdf', pos=[0.4, 0, -0.4], quat=[0, 0, -0.707, -0.707])
#pos=[0.7, 0, 0.1]


## pybullet_data objects

table = loader(PDATA_PATH, 'table/table.urdf',
               pos=[.75, -.2, -1],
               quat=[0, 0, 0.707107, 0.707107],
               scale=1.0)

duck = loader(PDATA_PATH, 'duck_vhacd.urdf',
              pos=[.75, -.4, -.3],
              deg=[0,0,0],
              scale=0.8)

x = 1.5
lego = loader(PDATA_PATH, 'lego/lego.urdf',
              pos=np.array([0.6, -.3, -.3]) + np.array([math.cos(x), math.sin(x), 0]) * .15,
              quat=[0, 0, 1, 0],
              rgba=[1, 0, 0, 1],
              scale=0.8)


## custom objects

bowl = loader(ASSET_PATH, 'objects/bowl/bowl.urdf',
              pos=[.75, -0.1, -.3],
              scale=0.05)

lid = loader(ASSET_PATH, 'objects/bowl/lid.urdf',
              pos=[.75, 0, -.3],
              scale=0.25)

cube = loader(ASSET_PATH, 'objects/cube/cube.urdf',
              pos=[.65, .2, -.3],
              scale=0.04)

spam = loader(ASSET_PATH, 'objects/spam/spam.urdf',
              pos=[.75, -.4, -.3],
              deg=[90,0,-90],
              scale=0.025)

box = loader(ASSET_PATH, 'objects/box/box.urdf',
                # pos=[0.85, 0, -.35],
                pos=[0.8, 0, -.35],
                scale=0.15)

hinge = loader(ASSET_PATH, 'objects/hinge/hinge.urdf',
                pos=[.75, -0.1, -.3],
                scale=0.1)
