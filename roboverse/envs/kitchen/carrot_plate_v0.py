import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.kitchen.bridge_kitchen_v0 import BridgeKitchenVO
from roboverse.bullet.misc import load_obj, load_urdf, deg_to_quat, quat_to_deg
from bullet_objects import loader, metadata
from roboverse.bullet.drawer_utils import *
from roboverse.bullet.button_utils import *
import os.path as osp
import importlib.util
import random
import pickle
import gym

END_EFFECTOR_INDEX = 8
RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.036, -0.036]
RESET_JOINT_VALUES_GRIPPER_CLOSED = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.015, -0.015]
RESET_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 10, 11]
GUESS = 3.14
JOINT_LIMIT_LOWER = [-3.14, -1.88, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
                     -0.037]
JOINT_LIMIT_UPPER = [3.14, 1.99, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]
JOINT_RANGE = []
for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER):
    JOINT_RANGE.append(upper - lower)

GRIPPER_LIMITS_LOW = JOINT_LIMIT_LOWER[-2:]
GRIPPER_LIMITS_HIGH = JOINT_LIMIT_UPPER[-2:]

class CarrotPlateVO(BridgeKitchenVO):

    def __init__(self,
                 reward_type='shaped',
                 observation_mode='state',
                 obs_img_dim=128,
                 transpose_image=False,
                 use_bounding_box=True,
                 max_force=1000.,
                 random_color_p=0.5,
                 DoF=6,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param reward_type: one of 'shaped', 'sparse'
        :param observation_mode: state, pixels, pixels_debug
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        """
        super().__init__(*args, **kwargs)

        # Task Info #
        self.task_objects = ['carrot', 'yellow_plate']
        self.goal = np.array([2.25, -0.02, -0.54])

        self._pos_low = [2.1, -0.28, -0.5]
        self._pos_high = [2.35, 0.05, -0.25]
        self.robot_pos = np.array([2.45, -0.2, -0.55])
        self.robot_deg = np.array([0,0,180])

        # Camera Info #
        camera_adjustment = np.array([-0.08, 0.14, 0.22])
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=self.robot_pos + camera_adjustment,
            distance=0.12, yaw=self.robot_deg[2] - 58, pitch=-40, roll=0, up_axis_index=2)
    
    ### REWARD FUNCTIONS ###

    def get_info(self):
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.get_object_pos('carrot')
        aligned = np.linalg.norm(obj_pos[:2] - ee_pos[:2]) < 0.05
        enclosed = np.linalg.norm(obj_pos[2] - ee_pos[2]) < 0.05
        above_goal = np.linalg.norm(obj_pos[:2] - self.goal[:2]) < 0.05
        done = np.linalg.norm(obj_pos - self.goal) < 0.05
        return {
        'hand_object_aligned': aligned,
        'object_grasped': aligned * enclosed,
        'object_above_goal': above_goal,
        'task_achieved': done,
        }

    def get_reward(self, info):
        return info['task_achieved'] - 1


    ### SCRIPTED POLICY ###
    def demo_reset(self):
        reset_obs = self.reset()
        self.timestep = 0
        self.done = False
        self.grip = -1.
        self.default_height = self.get_end_effector_pos()[2] - 0.08
        return reset_obs

    def get_demo_action(self):

        # Get xyz action
        if self.done:
            xyz_action, rot_action = self.maintain_hand()
        else:
            xyz_action, rot_action = self.move_obj()

        # Get abc action
        curr_angle = np.array(bullet.get_link_state(self._robot_id, END_EFFECTOR_INDEX, 'theta'))
        abc_action = (self.default_angle - curr_angle) * 0.1
        abc_action[2] = rot_action

        action = np.concatenate((xyz_action, abc_action, [self.grip]))
        action = np.clip(action, a_min=-1, a_max=1)
        noisy_action = np.random.normal(action, 0.05)
        noisy_action = np.clip(noisy_action, a_min=-1, a_max=1)
        self.timestep += 1

        return action, noisy_action


    def move_obj(self):
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.get_object_pos('carrot')
        curr_rot = self.get_object_angle('robot')[2]
        obj_rot = self.get_object_angle('carrot')[2]
        target_pos = obj_pos + np.array([-0.02, 0.0, 0.02])
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.05
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.05
        above = ee_pos[2] >= self.default_height
        drop_object = np.linalg.norm(obj_pos[:2] - self.goal[:2]) < 0.05
        self.hand_pos = np.array(ee_pos)

        if not aligned and not above and not drop_object:
            #print('Stage 1')
            action = np.array([0.,0., 0.5])
            self.grip = -1.
        elif not aligned:
            #print('Stage 2')
            action = (target_pos - ee_pos) * 3.0
            action[2] = (self.default_height + 0.05) - ee_pos[2]
            action *= 2.0
            self.grip = -1.
        elif aligned and not enclosed and not drop_object:
            #print('Stage 3')
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 1.5
            self.grip = -1.
        elif enclosed and not drop_object and self.grip < 1.:
            #print('Stage 4')
            action = target_pos - ee_pos
            self.grip += 0.2
        elif not above and not drop_object:
            #print('Stage 5')
            action = np.array([0.,0., .5])
            self.grip = 1.
        elif not drop_object:
            #print('Stage 6')
            action = (self.goal - ee_pos) * 4.0
            action[2] = (self.default_height + 0.05) - ee_pos[2]
            self.grip = 1.
        elif self.grip > -1.:
            #print('Stage 7')
            action = (self.goal - ee_pos) * 3.0
            action[2] = (self.default_height + 0.05) - ee_pos[2]
            self.grip -= 0.2
        else:
            #print('Stage 8')
            action = (self.goal - ee_pos) * 3.0
            action[2] = (self.default_height + 0.05) - ee_pos[2]
            self.grip = -1.
            self.done = True

        if enclosed:
            rot_action = 0
        else:
            rot_action = (obj_rot - curr_rot) * 0.3

        return action, rot_action

    def maintain_hand(self):
        action = self.hand_pos - np.array(self.get_end_effector_pos())
        return action, 0

    ### ENV DESIGN ###

    def _reset_scene(self):
        # Reset Environment Variables
        self.lights_off = False
        objects_present = list(self._objects.keys())
        for key in objects_present:
            p.removeBody(self._objects[key])

        # Reset Sawyer
        p.removeBody(self._robot_id)
        self._robot_id = bullet.objects.widowx_250(
            pos=self.robot_pos,
            quat=deg_to_quat(self.robot_deg),
            )

        self._objects['drawer'] = bullet.objects.real_drawer(
            pos=[2.0075, -0.52, -0.497],
            quat=deg_to_quat([0, 0, 90]),
            scale=0.15,
            rgba=[0.9, 0.9, 0.9, 1.])

        ### POT OBJECTS ###
        self._objects['pot'] = load_obj(
            self.shapenet_func('Objects/cooking_pot/models/model_vhacd.obj'),
            self.shapenet_func('Objects/cooking_pot/models/model_normalized.obj'),
            pos=[2.035, -.235, -0.45],
            quat=deg_to_quat([90, 0, 0]),
            scale=0.22,
            )

        self._objects['lid'] = bullet.objects.lid(
            pos=[2.035, -.05, -0.5],
            scale=0.082,
            rgba=[.815, .84, .875, 1],
            )

        ### CUTTING BOARD OBJECTS ###
        self._objects['yellow_plate'] = load_obj(
            self.shapenet_func('Objects/yellow_plate/models/model_vhacd.obj'),
            self.shapenet_func('Objects/yellow_plate/models/model_normalized.obj'),
            pos=[2.238, -0.05, -0.55],
            quat=deg_to_quat([90, 0, 0]),
            scale=0.2,
            )

        self._objects['leaves'] = load_obj(
            self.shapenet_func('Objects/leaves/models/model_vhacd.obj'),
            self.shapenet_func('Objects/leaves/models/model_normalized.obj'),
            pos=[2.05, -0.28, -0.3],
            quat=deg_to_quat([90, 0, 80]),
            scale=0.2,
            )

        if np.random.uniform() < 0.5:
            plant_pos = [2.238, -0.28, -0.45]
            carrot_pos = [2.27, -0.17, -0.45]
            carrot_quat = deg_to_quat([0, 0, -50])
            self.pos_flag = True
        else:
            plant_pos = [2.24, -0.16, -0.45]
            carrot_pos = [2.245, -0.27, -0.45]
            carrot_quat = deg_to_quat([0, 0, -110])

        self._objects['plant'] = load_obj(
            self.shapenet_func('Objects/plant/models/model_vhacd.obj'),
            self.shapenet_func('Objects/plant/models/model_normalized.obj'),
            pos=plant_pos,
            quat=deg_to_quat([0, 0, 60]),
            scale=0.14,
            )

        self._objects['carrot'] = load_obj(
            self.shapenet_func('Objects/carrot/models/model_vhacd.obj'),
            self.shapenet_func('Objects/carrot/models/model_vhacd.obj'),
            pos=carrot_pos,
            quat=carrot_quat,
            rgba=[1., .5, 0., 1.],
            scale=0.15,
            )


        # Allow the objects to land softly in low gravity
        p.setGravity(0, 0, -1)
        for _ in range(100):
            bullet.step()
        # After landing, bring to stop
        p.setGravity(0, 0, -10)
        for _ in range(100):
            bullet.step()

    def _load_environment(self):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)


        self._objects = {}
        self._sensors = {}
        self._robot_id = bullet.objects.widowx_250()
        self._table = bullet.objects.table(scale=10., pos=[2.3, -.2, -7.2], rgba=[1., .97, .86,1])
        self._workspace = bullet.Sensor(self._robot_id,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])

        self._end_effector = bullet.get_index_by_attribute(
            self._robot_id, 'link_name', '/gripper_bar_link')

        ### WALLS ###
        tile_wall = load_urdf(self.shapenet_func('Furniture/tile_wall/tile_wall.urdf'),
            pos=[1.78, -0.3, -0.215],
            quat=deg_to_quat([180, 180, 90]),
            scale=2.5,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(tile_wall, -1, 1, 0)

        tile_wall = load_urdf(self.shapenet_func('Furniture/tile_wall/tile_wall.urdf'),
            pos=[2.55, -0.82, -0.215],
            quat=deg_to_quat([180, 180, 180]),
            scale=2.5,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(tile_wall, -1, 1, 0)

        tile_wall = load_urdf(self.shapenet_func('Furniture/tile_wall/tile_wall.urdf'),
            pos=[2.25, 0.82, -0.215],
            quat=deg_to_quat([180, 180, 0]),
            scale=2.5,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(tile_wall, -1, 1, 0)

        cabinets = load_urdf(self.shapenet_func('Furniture/cabinets/cabinets.urdf'),
            pos=[2.2, -0.395, -0.83],
            quat=deg_to_quat([90, 0, 0]),
            scale=1.4,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(cabinets, -1, 0, 0)

        ### MARBLE TABELS ###
        marble_table = load_urdf(self.shapenet_func('Furniture/marble_table/marble_table.urdf'),
            pos=[2.2, 0.745, -0.765],
            quat=deg_to_quat([90, 0, 180]),
            scale=1.75,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(marble_table, -1, 1, 0)

        marble_table = load_urdf(self.shapenet_func('Furniture/marble_table/marble_table.urdf'),
            pos=[2.2, -0.765, -0.765],
            quat=deg_to_quat([90, 0, 180]),
            scale=1.75,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(marble_table, -1, 1, 0)

        ### SINK ###
        stovetop = load_urdf(self.shapenet_func('Furniture/stovetop/stovetop.urdf'),
            pos=[2.035, -.15, -0.5514],
            quat=deg_to_quat([90, 0, 90]),
            scale=0.43,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(stovetop, -1, 1, 0)

        wood_table = load_urdf(self.shapenet_func('Furniture/wood_table/wood_table.urdf'),
            pos=[2.168, -.184, -0.583],
            quat=deg_to_quat([90, 0, 270]),
            scale=0.48,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(wood_table, -1, 1, 0)

        stove_base = load_urdf(self.shapenet_func('Furniture/stove_base/stove_base.urdf'),
            pos=[2.02, -.184, -0.5829],
            quat=deg_to_quat([90, 0, 270]),
            scale=0.48,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(stove_base, -1, 1, 0)

        table = load_urdf(self.shapenet_func('Furniture/table/table.urdf'),
            pos=[2.13, 0, -0.9],
            quat=deg_to_quat([90, 0, 270]),
            scale=1.0,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(table, -1, 1, 0)

        oven = load_urdf(self.shapenet_func('Furniture/oven/oven.urdf'),
            pos=[2.34, -.165, -0.68],
            quat=deg_to_quat([90, 0, 270]),
            scale=0.65,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(oven, -1, 0, 0)

        ### DRAWER ###
        drawer_cabinet = load_urdf(self.shapenet_func('Furniture/drawer_cabinet/drawer_cabinet.urdf'),
            pos=[2.0795, -0.52, -0.249],
            quat=deg_to_quat([90, 180, 270]),
            scale=0.637,
            useFixedBase=True,
            rgba=[0.9, 0.9, 0.9, 1.])
        p.setCollisionFilterGroupMask(drawer_cabinet, -1, 0, 0)
