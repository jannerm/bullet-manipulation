import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.sawyer_base import SawyerBaseEnv
from roboverse.bullet.misc import load_obj, load_urdf, deg_to_quat, quat_to_deg
from roboverse.bullet.drawer_utils import *
from roboverse.bullet.button_utils import *
import os.path as osp
import importlib.util
import random
import pickle
import gym

class BridgeKitchenBaseV0(SawyerBaseEnv):

    def __init__(self,
                 reward_type='shaped',
                 observation_mode='state',
                 obs_img_dim=48,
                 transpose_image=False,
                 invisible_robot=False,
                 use_bounding_box=True,
                 random_color_p=0.5,
                 task='goal_reaching',
                 DoF=3,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param reward_type: one of 'shaped', 'sparse'
        :param observation_mode: state, pixels, pixels_debug
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        :param invisible_robot: the robot arm is invisible when set to True
        """
        assert DoF in [3, 4, 6]
        assert task in ['goal_reaching', 'pickup']

        self._reward_type = reward_type
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self.random_color_p = random_color_p
        self.use_bounding_box = use_bounding_box
        self._ddeg_scale = 5
        self.task = task
        self.DoF = DoF

        self._object_position_low = (0.65,-0.15, -.3)
        self._object_position_high = (0.75,0.15, -.3)
        self._goal_low = np.array([0.65,-0.15,-.34])
        self._goal_high = np.array([0.75,0.15,-0.22])
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self.obs_img_dim = obs_img_dim
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.7, 0, -0.3], distance=0.3,
            yaw=90, pitch=-15, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        self.dt = 0.1
        super().__init__(*args, **kwargs)
        self._max_force = 100
        self._action_scale = 0.05
        self._load_environment()


    def _set_spaces(self):
        act_dim = self.DoF + 1
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        observation_dim = 11
        if self.DoF > 3:
            # Add wrist theta
            observation_dim += 4

        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)

        self.observation_space = Dict([
            ('observation', state_space),
            ('state_observation', state_space),
            ('desired_goal', state_space),
            ('state_desired_goal', state_space),
            ('achieved_goal', state_space),
            ('state_achieved_goal', state_space),
        ])


    def _reset_scene(self):
        # Reset Environment Variables
        self.lights_off = False
        objects_present = list(self._objects.keys())
        for key in objects_present:
            p.removeBody(self._objects[key])

        # Reset Sawyer
        p.removeBody(self._sawyer)
        self._sawyer = bullet.objects.sawyer_hand_visual_only()

        # Reset Button
        try: p.removeBody(self._button)
        except: pass
        self._button = bullet.objects.wall_button(pos=[1.95, 0.6, -0.32], quat=deg_to_quat([0, 90, 0]))
        self.init_button_depth = get_button_cylinder_pos(self._button)[0]

        # Reset Drawer
        try: p.removeBody(self._drawer)
        except: pass
        self._drawer = bullet.objects.real_drawer(
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
            pos=[2.035, -.277, -0.35],
            scale=0.082,
            rgba=[.815, .84, .875, 1],
            )

        ### SINK OBJECTS ###
        # self._objects['moving_faucet'] = load_urdf(
        #     self.shapenet_func('Furniture/moving_faucet/moving_faucet.urdf'),
        #     pos=[1.96, 0.17, -0.49],
        #     quat=deg_to_quat([0, 0, 180]),
        #     scale=0.15,
        #     useFixedBase=True)

        # self._objects['soap'] = load_obj(
        #     self.shapenet_func('Objects/dish_soap/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/dish_soap/models/model_normalized.obj'),
        #     pos=[1.95, 0.07, -0.5],
        #     quat=deg_to_quat([90, 0, 0]),
        #     scale=0.11,
        #     rgba=[0.2, 1., 0.6, 1]
        #     )

        # self._objects['bowl'] = load_obj(
        #     self.shapenet_func('Objects/bowl/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/bowl/models/model_normalized.obj'),
        #     pos=[2.05, 0.13, -0.65],
        #     quat=deg_to_quat([90, 0, 0]),
        #     scale=0.2,
        #     )


        ### DISH RACK OBJECTS ###
        # self._objects['mug'] = load_obj(
        #     self.shapenet_func('Objects/mug/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/mug/models/model_normalized.obj'),
        #     pos=[2.1, 0.45, -0.45],
        #     quat=deg_to_quat([90, 0, 0]),
        #     scale=0.11,
        #     rgba=[0.6, .6, 0., 1]
        #     )

        # self._objects['red_plate'] = load_obj(
        #     self.shapenet_func('Objects/red_plate/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/red_plate/models/model_normalized.obj'),
        #     pos=[2.25, 0.42, -0.35],
        #     quat=deg_to_quat([200, 0, 90]),
        #     scale=0.2,
        #     )

        # self._objects['knife'] = load_obj(
        #     self.shapenet_func('Objects/knife/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/knife/models/model_normalized.obj'),
        #     pos=[2.015, 0.34, -0.45],
        #     quat=deg_to_quat([90, -10, -10]),
        #     scale=0.18,
        #     rgba=[0.73, .77, .8, 1]
        #     )

        ### COLANDER OBJECTS ###
        # self._objects['colander'] = load_obj(
        #     self.shapenet_func('Objects/colander/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/colander/models/model_normalized.obj'),
        #     pos=[2.45, 0.42, -0.45],
        #     quat=deg_to_quat([90, 0, 0]),
        #     scale=0.23,
        #     )

        # x,y,z = 0.02, 0.05, 0.05
        # self._objects['fruit_basket'] = load_obj(
        #     self.shapenet_func('Objects/fruit_basket/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/fruit_basket/models/model_normalized.obj'),
        #     pos=[2.45 + y, 0.42 + z, -0.5],
        #     quat=deg_to_quat([90, 0, 0]),
        #     scale=0.25,
        #     )


        # self._objects['orange'] = load_obj(
        #     self.shapenet_func('Objects/orange/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/orange/models/model_normalized.obj'),
        #     pos=[2.45 + y, 0.41 + z, -0.42 + x],
        #     scale=0.09,
        #     )

        # self._objects['apple'] = load_obj(
        #     self.shapenet_func('Objects/apple/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/apple/models/model_normalized.obj'),
        #     pos=[2.45 + y, 0.43 + z, -0.45 + x],
        #     scale=0.07,
        #     )

        # self._objects['banana'] = load_obj(
        #     self.shapenet_func('Objects/banana/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/banana/models/model_normalized.obj'),
        #     pos=[2.45 + y, 0.42 + z, -0.38 + x],
        #     quat=deg_to_quat([90, 0, -20]),
        #     rgba=[1.,1.,0.,1.],
        #     scale=0.12,
        #     )

        ### TRASH OBJECTS ###
        # self._objects['swiffer'] = load_obj(
        #     self.shapenet_func('Objects/swiffer/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/swiffer/models/model_normalized.obj'),
        #     pos=[2.65, -0.45, -0.5],
        #     quat=deg_to_quat([180, 0, 0]),
        #     rgba=[.545, 0, .545, 1.],
        #     scale=0.2,
        #     )

        # self._objects['crushed_can'] = load_obj(
        #     self.shapenet_func('Objects/crushed_can/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/crushed_can/models/model_normalized.obj'),
        #     pos=[2.63, -0.53, -0.4],
        #     scale=0.1,
        #     rgba=[.78, .78, .78, 1],
        #     )

        # self._objects['crushed_foil'] = load_obj(
        #     self.shapenet_func('Objects/crushed_foil/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/crushed_foil/models/model_normalized.obj'),
        #     pos=[2.67, -0.53, -0.4],
        #     scale=0.1,
        #     )

        # self._objects['plastic_bottle'] = load_obj(
        #     self.shapenet_func('Objects/plastic_bottle/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/plastic_bottle/models/model_normalized.obj'),
        #     pos=[2.65, -0.28, -0.6],
        #     quat=deg_to_quat([90, 90, 0]),
        #     scale=0.13,
        #     )


        ### DRAWER OBJECTS ###
        # self._objects['spam'] = bullet.objects.spam(
        #     pos=[2.02, -0.52, -0.38],
        #     scale=0.02,
        #     )

        # self._objects['wine'] = load_obj(
            # self.shapenet_func('Objects/wine/models/model_vhacd.obj'),
            # self.shapenet_func('Objects/wine/models/model_normalized.obj'),
            # pos=[2.00, -0.52, -0.38],
            # quat=deg_to_quat([0, 0, 0]),
            # scale=0.125,
            # rgba=[.83, .83, .83, 1],
            # )

        ### CUTTING BOARD OBJECTS ###
        # self._objects['yellow_plate'] = load_obj(
        #     self.shapenet_func('Objects/yellow_plate/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/yellow_plate/models/model_normalized.obj'),
        #     pos=[2.238, -0.05, -0.55],
        #     quat=deg_to_quat([90, 0, 0]),
        #     scale=0.2,
        #     )

        # self._objects['plant'] = load_obj(
        #     self.shapenet_func('Objects/plant/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/plant/models/model_normalized.obj'),
        #     pos=[2.238, -0.17, -0.45],
        #     quat=deg_to_quat([0, 0, 60]),
        #     scale=0.14,
        #     )

        # self._objects['leaves'] = load_obj(
        #     self.shapenet_func('Objects/leaves/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/leaves/models/model_normalized.obj'),
        #     pos=[2.238, -0.28, -0.45],
        #     quat=deg_to_quat([0, 90, 80]),
        #     scale=0.18,
        #     )

        # self._objects['carrot'] = load_obj(
        #     self.shapenet_func('Objects/carrot/models/model_vhacd.obj'),
        #     self.shapenet_func('Objects/carrot/models/model_vhacd.obj'),
        #     pos=[2.243, -0.02, -0.45],
        #     quat=deg_to_quat([0, 0, -20]),
        #     rgba=[1., .5, 0., 1.],
        #     scale=0.15,
        #     )

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
        self._sawyer = bullet.objects.sawyer_hand_visual_only()
        self._table = bullet.objects.table(scale=10., pos=[2.3, -.2, -7.2], rgba=[1., .97, .86,1])
        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')

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

        cabinets = load_urdf(self.shapenet_func('Furniture/cabinets/cabinets.urdf'),
            pos=[2.2, 0.375, -0.83],
            quat=deg_to_quat([90, 0, 180]),
            scale=1.4,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(cabinets, -1, 0, 0)

        cabinets = load_urdf(self.shapenet_func('Furniture/cabinets/cabinets.urdf'),
            pos=[2.2, -0.395, -0.83],
            quat=deg_to_quat([90, 0, 0]),
            scale=1.4,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(cabinets, -1, 0, 0)

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

        faucet = load_urdf(self.shapenet_func('Furniture/faucet/faucet.urdf'),
            pos=[2.01, 0.175, -0.505],
            quat=deg_to_quat([90, 0, 270]),
            scale=0.2,
            rgba=[189 / 255, 195 / 255, 199 / 255, 1],
            useFixedBase=True)
        p.setCollisionFilterGroupMask(faucet, -1, 1, 0)

        oven = load_urdf(self.shapenet_func('Furniture/oven/oven.urdf'),
            pos=[2.34, -.165, -0.68],
            quat=deg_to_quat([90, 0, 270]),
            scale=0.65,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(oven, -1, 0, 0)

        sink_drawer = load_urdf(self.shapenet_func('Furniture/sink_drawer/sink_drawer.urdf'),
            pos=[2.3175, 0.175, -0.885],
            quat=deg_to_quat([90, 0, 270]),
            scale=0.71,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(sink_drawer, -1, 0, 0)


        ### DISH RACK ###
        dish_rack = load_urdf(self.shapenet_func('Furniture/dish_rack/dish_rack.urdf'),
            pos=[2.13, 0.425, -0.61],
            quat=deg_to_quat([90, 0, 270]),
            scale=0.4,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(dish_rack, -1, 1, 0)

        utensil_holder = load_urdf(self.shapenet_func('Furniture/utensil_holder/utensil_holder.urdf'),
            pos=[2.015, 0.34, -0.53],
            quat=deg_to_quat([90, 0, 90]),
            scale=0.1,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(utensil_holder, -1, 1, 0)

        ### DRAWER ###
        drawer_cabinet = load_urdf(self.shapenet_func('Furniture/drawer_cabinet/drawer_cabinet.urdf'),
            pos=[2.0795, -0.52, -0.249],
            quat=deg_to_quat([90, 180, 270]),
            scale=0.637,
            useFixedBase=True,
            rgba=[0.9, 0.9, 0.9, 1.])
        p.setCollisionFilterGroupMask(drawer_cabinet, -1, 1, 0)

        ### TRASH CAN ###
        trash_can = load_urdf(self.shapenet_func('Furniture/trash_can/trash_can.urdf'),
            pos=[2.65, -0.27, -0.7],
            quat=deg_to_quat([90, 0, 180]),
            scale=0.4,
            rgba=[.5,.8,.9,1],
            useFixedBase=True)
        p.setCollisionFilterGroupMask(trash_can, -1, 1, 0)

        ## DECOR ###
        microwave_wall = load_urdf(self.shapenet_func('Furniture/microwave_wall/microwave_wall.urdf'),
            pos=[2.08, 0.681, -0.55],
            quat=deg_to_quat([90, 0, 180]),
            scale=1.0,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(microwave_wall, -1, 0, 0)

        microwave = load_urdf(self.shapenet_func('Furniture/microwave/microwave.urdf'),
            pos=[2.16, 0.7, -0.4],
            quat=deg_to_quat([90, 0, 180]),
            scale=0.4,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(microwave, -1, 0, 0)

        painting = load_urdf(self.shapenet_func('Furniture/painting/painting.urdf'),
            pos=[2.7, -0.68, -0.175],
            quat=deg_to_quat([90, 0, 0]),
            scale=0.4,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(painting, -1, 0, 0)

        clock = load_urdf(self.shapenet_func('Furniture/clock/clock.urdf'),
            pos=[2.54, 0.68, -0.215],
            quat=deg_to_quat([90, 0, 0]),
            scale=0.25,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(clock, -1, 0, 0)

    def sample_object_location(self):
        if self._randomize:
            return np.random.uniform(
                low=self._object_position_low, high=self._object_position_high)
        return self._fixed_object_position

    def sample_object_color(self):
        if np.random.uniform() < self.random_color_p:
            return list(np.random.choice(range(256), size=3) / 255.0) + [1]
        return None

    def sample_quat(self, object_name):
        if object_name in self.quat_dict:
            return self.quat_dict[self.curr_object]
        return deg_to_quat(np.random.randint(0, 360, size=3))

    def add_object(self, obj_name, object_position=None, quat=None):
        # Generate random object position
        if object_position is None:
            object_position = self.sample_object_location()

        # Generate quaterion if none is given
        if quat is None:
            quat = self.sample_quat(self.curr_object)

        # Spawn object above table
        self._objects[obj_name] = loader.load_shapenet_object(
                self.curr_id,
                self.scaling,
                object_position,
                quat=quat,
                rgba=self.curr_color)

        # Allow the objects to land softly in low gravity
        p.setGravity(0, 0, -1)
        for _ in range(100):
            bullet.step()
        # After landing, bring to stop
        p.setGravity(0, 0, -10)
        for _ in range(100):
            bullet.step()


    def button_pressed(self):
        curr_depth = get_button_cylinder_pos(self._button)[0]
        return (self.init_button_depth - curr_depth) > 0.01

    def _format_action(self, *action):
        if self.DoF == 3:
            if len(action) == 1:
                delta_pos, gripper = action[0][:-1], action[0][-1]
            elif len(action) == 2:
                delta_pos, gripper = action[0], action[1]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), gripper
        elif self.DoF == 4:
            if len(action) == 1:
                delta_pos, delta_yaw, gripper = action[0][:3], action[0][3:4], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_yaw, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            
            delta_angle = [0, 0, delta_yaw[0]]
            return np.array(delta_pos), np.array(delta_angle), gripper
        else:
            if len(action) == 1:
                delta_pos, delta_angle, gripper = action[0][:3], action[0][3:6], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_angle, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), np.array(delta_angle), gripper


    def step(self, *action):
        # Get positional information
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        curr_angle = bullet.get_link_state(self._sawyer, self._end_effector, 'theta')
        default_angle = quat_to_deg(self.default_theta)
    
        # Keep necesary degrees of theta fixed
        if self.DoF == 3:
            angle = default_angle
        elif self.DoF == 4:
            angle = np.append(default_angle[:2], [curr_angle[2]])
        else:
            angle = curr_angle

        # If angle is part of action, use it
        if self.DoF == 3:
            delta_pos, gripper = self._format_action(*action)
        else:
            delta_pos, delta_angle, gripper = self._format_action(*action)
            angle += delta_angle * self._ddeg_scale

        # Update position and theta
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)
        theta = deg_to_quat(angle)
        self._simulate(pos, theta, gripper)

        # Turn off light if button is pressed
        self.lights_off = self.lights_off or self.button_pressed()

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        return observation, reward, done, info

    def get_info(self):
        return {}

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0, lights_off=self.lights_off)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def get_reward(self, info):
        return 0

    def reset(self, change_object=True):
        self._reset_scene()
        self._format_state_query()

        # Sample and load starting positions
        init_pos = np.array(self._pos_init)
        self.goal_pos = np.random.uniform(low=self._goal_low, high=self._goal_high)
        bullet.position_control(self._sawyer, self._end_effector, init_pos, self.default_theta)

        # Move to starting positions
        action = np.array([0 for i in range(self.DoF)] + [-1])
        for _ in range(3):
            self.step(action)

        return self.get_observation()

    def format_obs(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs
    
    def compute_reward_gr(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        obj_goal = self.format_obs(contexts['state_desired_goal'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
        object_goal_success = object_goal_distance < self._success_threshold
        return object_goal_success - 1

    def compute_reward(self, obs, actions, next_obs, contexts):
        return self.compute_reward_gr(obs, actions, next_obs, contexts)

    def get_object_deg(self):
        object_info = bullet.get_body_info(self._objects['obj'],
                                           quat_to_deg=True)
        return object_info['theta']

    def get_hand_deg(self):
        return bullet.get_link_state(self._sawyer, self._end_effector,
            'theta', quat_to_deg=True)

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)
        hand_theta = bullet.get_link_state(self._sawyer, self._end_effector,
            'theta', quat_to_deg=False)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        # object_info = bullet.get_body_info(self._objects['obj'],
        #                                    quat_to_deg=False)
        object_pos = [0,0,0]#object_info['pos']
        object_theta = [0,0,0,0]#object_info['theta']

        if self.DoF > 3:
            observation = np.concatenate((
                end_effector_pos, hand_theta, gripper_tips_distance,
                object_pos, object_theta))
            goal_pos = np.concatenate((
                self.goal_pos, hand_theta, gripper_tips_distance,
                self.goal_pos, object_theta))
        else:
            observation = np.concatenate((
                end_effector_pos, gripper_tips_distance,
                object_pos, object_theta))
            goal_pos = np.concatenate((
                end_effector_pos, gripper_tips_distance,
                self.goal_pos, object_theta))

        obs_dict = dict(
            observation=observation,
            state_observation=observation,
            desired_goal=goal_pos,
            state_desired_goal=goal_pos,
            achieved_goal=observation,
            state_achieved_goal=observation,
            )

        return obs_dict