import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.bullet.control import get_object_position
from roboverse.envs.sawyer_base import SawyerBaseEnv
from roboverse.bullet.misc import load_obj, deg_to_quat, get_bbox	
from roboverse.utils.misc import quat_to_deg, quat_to_deg_batch, first_nonzero
from bullet_objects import loader, metadata
import os.path as osp
import importlib.util
import random
import pickle
import gym
from roboverse.bullet.drawer_utils import *
from roboverse.bullet.button_utils import *
from PIL import Image
import pkgutil

# Constants
td_close_coeff = 0.15134
td_open_coeff = 0.2695
td_offset_coeff = 0.0125

gripper_bounding_x = [.5, .8]
gripper_bounding_y = [-.17, .17]
gripper_bounding_z = [-0.35, -0.1]

quadrants = [
    [.525, .1675],
    [.525, -.1675],
    [.775, -.1675],
    [.775, .1675],
]

slide_offset = .11
goal_slide_offset = .025
slide_quadrants = [
    [0.5 + slide_offset, 0.2 - slide_offset],
    [0.5 + slide_offset, slide_offset - 0.1989],
    [0.84 - slide_offset, slide_offset - 0.1989],
    [0.84 - slide_offset, 0.2 - slide_offset],
]
goal_slide_quadrants = [
    [0.5 + slide_offset - goal_slide_offset,
        0.2 - slide_offset + goal_slide_offset],
    [0.5 + slide_offset - goal_slide_offset,
        slide_offset - 0.1989 - goal_slide_offset],
    [0.84 - slide_offset + goal_slide_offset,
        slide_offset - 0.1989 - goal_slide_offset],
    [0.84 - slide_offset + goal_slide_offset,
        0.2 - slide_offset + goal_slide_offset],
]


class SawyerDiverseDrawerPnpPush(SawyerBaseEnv):

    def __init__(self,
                 obs_img_dim=48,
                 transpose_image=False,
                 test_env=False,
                 DoF=4,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        """
        assert DoF in [4]

        ## Constants
        self.obj_inverse_thresh = 0.05
        self.obj_thresh = 0.08
        self.drawer_thresh = 0.065
        self.gripper_pos_thresh = 0.08
        self.gripper_rot_thresh = 10
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self._ddeg_scale = 5
        self.lift_gripper_action = np.array([0, 0, 1, 0, -1])
        self.lower_gripper_action = np.array([0, 0, -1, 0, -1])

        ## Fixed Env Settings (Normally Shouldn't Change)
        self._transpose_image = transpose_image
        self.obs_img_dim = obs_img_dim
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(
            self.image_shape) * 3  # image has 3 channels
        self.DoF = DoF
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)

        self.drawer_z = -.34
        self.large_obj_z = -.3525
        self.table_z = -1.
        self.wall_z = -.3
        self.on_top_drawer_goal_z = -0.26951111
        self.in_drawer_goal_z = -0.3290406
        self.out_of_drawer_goal_z = -0.34

        ## Test Env
        self.test_env = test_env
        self.test_env_command = kwargs.pop('test_env_command', None)
        self.use_multiple_goals = kwargs.pop('use_multiple_goals', False)
        self.use_test_env_command_sequence = kwargs.pop('use_test_env_command_sequence', True)
        if self.test_env:
            assert self.test_env_command
            self.test_env_commands = kwargs.pop('test_env_commands', None)
            if self.test_env_commands is not None:
                self.test_env_seed = list(self.test_env_commands.keys())[0]
            else:
                self.test_env_seed = None
            self.use_inverse_pnp_metric = self.test_env_command.pop('use_inverse_pnp_metric', False)
        else:
            self.use_inverse_pnp_metric = False
        self.random_init_gripper_pos = kwargs.pop('random_init_gripper_pos', False)
        self.random_init_gripper_yaw = kwargs.pop('random_init_gripper_yaw', False)
        self.random_init_gripper_yaw_discrete = kwargs.pop('random_init_gripper_yaw_discrete', False)
        assert not (self.random_init_gripper_yaw and self.random_init_gripper_yaw_discrete)

        ## Env Config
        self.camera_yaw_low = kwargs.pop('camera_yaw_low', 60)
        self.camera_yaw_high = kwargs.pop('camera_yaw_high', 120)
        self.camera_pitch_low = kwargs.pop('camera_pitch_low', -44)
        self.camera_pitch_high = kwargs.pop('camera_pitch_high', -10)
        self.table_pos_offset_low = kwargs.pop('table_pos_offset_low', [-.05, -.05, -.035])
        self.table_pos_offset_high = kwargs.pop('table_pos_offset_high', [.01, .05, .1])
        self.use_target_config = self.test_env or kwargs.pop('use_target_config', False)
        self.fix_camera_yaw_pitch = kwargs.pop('fix_camera_yaw_pitch', False)
        self._load_new_env_config()

        self.fixed_drawer_yaw = kwargs.pop('fixed_drawer_yaw', None)
        self.fixed_drawer_quadrant = kwargs.pop('fixed_drawer_quadrant', None)
        if self.fixed_drawer_quadrant is not None:
            assert self.fixed_drawer_quadrant == 0 or self.fixed_drawer_quadrant == 1

        self.obj_pnp = None
        self.obj_slide = None
        self._large_obj = None
        self._small_obj = None
        self.top_drawer_handle_can_move = True

        ## Reset-free
        if self.test_env and self.use_test_env_command_sequence:
            kwargs.pop('reset_interval', 1)
            self.reset_interval = len(
                self.test_env_command['command_sequence'])
        else:
            self.reset_interval = kwargs.pop('reset_interval', 1)
        self.reset_counter = self.reset_interval-1
        self.expl = kwargs.pop('expl', False)
        self.reset_gripper_interval = kwargs.pop(
            'reset_gripper_interval', self.reset_interval)
        self.reset_gripper_counter = self.reset_gripper_interval - 1

        assert self.reset_gripper_interval <= self.reset_interval and self.reset_interval % self.reset_gripper_interval == 0

        # Demo
        self.demo_num_ts = kwargs.pop('demo_num_ts', None)
        self.expert_policy_std = kwargs.pop('expert_policy_std', 0.1)
        self.fixed_task = kwargs.pop('fixed_task', None)
        if self.fixed_task:
            assert self.reset_interval == 1

        self.random_pick_offset = 0
        self.gripper_in_right_position = False
        self.gripper_picked_object = False
        self.gripper_has_been_above = False
        self.trajectory_done = False
        self.curr_task = None

        # Rendering
        self.downsample = kwargs.pop('downsample', False)
        self.env_obs_img_dim = kwargs.pop('env_obs_img_dim', self.obs_img_dim)
        self.render_depth = kwargs.pop('render_depth', False)
        self.render_segmentation = kwargs.pop('render_segmentation', False)

        # Magic Grasp
        self.grasp_constraint = None

        super().__init__(*args, **kwargs)

        # Need to overwrite in some cases, registration isnt working
        self._max_force = 100
        self._action_scale = 0.05
        self._pos_init = [0.6, -0.15, -0.2]
        self._pos_low = [0.45, -0.25, -.36]
        self._pos_high = [0.85, 0.25, -0.1]
    
    def _set_spaces(self):
        act_dim = self.DoF + 1
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        observation_dim = 32
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

    def _load_new_env_config(self):
        if self.use_target_config:
            yaw = 90
            pitch = -27
            rgbs = np.array([
                [.93, .294, .169, 1.],
                [.5, 1., 0., 1],
                [0.0, .502, .502, 1.],
                [.1, .25, .6, 1.],
                [.68, .85, .90, 1.],
                [.5, .5, .5, 1.],
                [.59, .29, 0.0, 1.],
                [.92, .85, .7, 1.],
                [.67, .67, .67, 1.],
            ])
            table_pos_offset = np.zeros((3,))
        else:
            if self.fix_camera_yaw_pitch:
                yaw = 90
                pitch = -27
            else:
                yaw = np.random.uniform(self.camera_yaw_low, self.camera_yaw_high)
                pitch = np.random.uniform(self.camera_pitch_low, self.camera_pitch_high)
            rgbs = np.hstack((np.random.uniform(0, 1, (9, 3)), np.ones((9, 1))))
            table_pos_offset = np.random.uniform(self.table_pos_offset_low, self.table_pos_offset_high)

        ## For debugging
        # rgbs = np.array([
        #     [.93, .294, .169, 1.],
        #     [.5, 1., 0., 1],
        #     [0.0, .502, .502, 1.],
        #     [.1, .25, .6, 1.],
        #     [.68, .85, .90, 1.],
        #     [.5, .5, .5, 1.],
        #     [.59, .29, 0.0, 1.],
        #     [.92, .85, .7, 1],
        # ])
        # yaw = 0.5 * self.camera_yaw_low + 0.5 * self.camera_yaw_high
        # pitch = 0.5 * self.camera_pitch_low + 0.5 * self.camera_pitch_high
        # table_pos_offset = 0.5 * np.array(self.table_pos_offset_low) + 0.5 * np.array(self.table_pos_offset_high)
        
        # yaw = 90
        # pitch = -27
        # table_pos_offset = np.zeros((3,))

        self.configs = {
            'camera_angle': {
                'yaw': yaw,
                'pitch': pitch,
            },
            'object_rgbs': {
                'large_object': rgbs[0],
                'small_object': rgbs[1],
                'tray': rgbs[2],
                'drawer': {
                    'frame': rgbs[3],
                    'bottom_frame': rgbs[4],
                    'bottom': rgbs[5],
                    'handle': rgbs[6],
                },
            },
            'table_rgb': rgbs[7],
            'table_pos_offset': table_pos_offset,
            'wall_rgb': rgbs[8],
        }
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[0.7, 0, -0.25], distance=0.5,
            yaw=self.configs['camera_angle']['yaw'], pitch=self.configs['camera_angle']['pitch'], roll=0, up_axis_index=2)

    def _load_table(self):
        self._objects = {}
        self._sensors = {}

        self._sawyer = bullet.objects.drawer_sawyer(physicsClientId=self._uid)
        table_pos = np.array([.75, -.2, self.table_z]) + self.configs['table_pos_offset']
        self._table = bullet.objects.table(
            pos=table_pos, rgba=self.configs['table_rgb'], physicsClientId=self._uid)
        wall_pos = np.array([.68, 0, self.wall_z]) + self.configs['table_pos_offset']
        self._wall = bullet.objects.wall_narrow_r_rgba(
            pos=wall_pos, rgba=self.configs['wall_rgb'], scale=1.0, physicsClientId=self._uid)

        self.top_drawer_quadrant = random.choice([0, 1])
        if self.fixed_task == 'open_drawer':
            is_close_drawer = True
        elif self.fixed_task == 'close_drawer':
            is_close_drawer = False
        else:
            is_close_drawer = np.random.uniform() < .5

        ## Top Drawer
        if self.test_env:
            self.drawer_yaw = self.test_env_command['drawer_yaw']
            self.top_drawer_quadrant = self.test_env_command['drawer_quadrant']
            quadrant = quadrants[self.top_drawer_quadrant]
            drawer_frame_pos = np.array([quadrant[0], quadrant[1], self.drawer_z]) + self.configs['table_pos_offset']
        else:
            self.drawer_yaw = self.fixed_drawer_yaw if self.fixed_drawer_yaw else random.uniform(
                0, 180)
            if self.fixed_drawer_quadrant is not None:
                quadrant = quadrants[self.fixed_drawer_quadrant]
                drawer_frame_pos = np.array([quadrant[0], quadrant[1], self.drawer_z]) + self.configs['table_pos_offset']
            else:
                tries = 0
                quadrant = quadrants[self.top_drawer_quadrant]
                while(True):
                    drawer_frame_pos = np.array([quadrant[0], quadrant[1], self.drawer_z]) + self.configs['table_pos_offset']
                    drawer_handle_open_goal_pos = drawer_frame_pos + td_open_coeff * \
                        np.array([np.sin(self.drawer_yaw * np.pi / 180), -
                                 np.cos(self.drawer_yaw * np.pi / 180), 0])
                    if gripper_bounding_x[0] <= drawer_handle_open_goal_pos[0] <= gripper_bounding_x[1] \
                            and gripper_bounding_y[0] <= drawer_handle_open_goal_pos[1] <= gripper_bounding_y[1]:
                        break
                    tries += 1
                    if (tries > 25):
                        self.drawer_yaw = self.fixed_drawer_yaw if self.fixed_drawer_yaw else random.uniform(
                            0, 180)

        quat = deg_to_quat([0, 0, self.drawer_yaw], physicsClientId=self._uid)

        self._top_drawer = bullet.objects.drawer_lightblue_base_longhandle_rgba(
            quat=quat, pos=drawer_frame_pos, rgba=[
                self.configs['object_rgbs']['drawer'][k] for k in ['frame', 'bottom_frame', 'bottom', 'handle']
            ], physicsClientId=self._uid, scale=.11)
        self.top_drawer_handle_can_move = True

        open_drawer(self._top_drawer, 100, physicsClientId=self._uid)

        self.init_handle_pos = get_drawer_handle_pos(
            self._top_drawer, physicsClientId=self._uid)[1]

        ## Tray above top drawer
        top_drawer_tray_pos = drawer_frame_pos + np.array([0, 0, .059])
        self._top_drawer_tray = bullet.objects.tray_teal_rgba(
            quat=quat,
            rgba=self.configs['object_rgbs']['tray'],
            pos=top_drawer_tray_pos,
            scale=0.165,
            physicsClientId=self._uid
        )

        # Tray acts as stopper for drawer closing
        tray_pos = self.get_drawer_handle_future_pos(-.05)
        self._tray = bullet.objects.tray_heavy(
            quat=quat, pos=tray_pos, scale=0.001, physicsClientId=self._uid)

        if self.test_env:
            if not self.test_env_command['drawer_open']:
                close_drawer(self._top_drawer, 200, physicsClientId=self._uid)
        else:
            if is_close_drawer:
                close_drawer(self._top_drawer, 200, physicsClientId=self._uid)

        self._load_table_large_objs()
        self._load_table_small_objs(is_close_drawer)

        self._workspace = bullet.Sensor(self._sawyer,
                                        xyz_min=self._pos_low, xyz_max=self._pos_high,
                                        visualize=False, rgba=[0, 1, 0, .1], physicsClientId=self._uid)
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site', physicsClientId=self._uid)

        # Task 14/38: Remove collision physics between cylinder and drawer handle
        if self.test_env and self.test_env_command.get("no_collision_handle_and_cylinder", False):
            for idx in [2, 3, 4]:
                p.setCollisionFilterPair(
                    self._top_drawer,
                    self._large_obj,
                    idx,
                    -1,
                    enableCollision=False,
                    physicsClientId=self._uid)

        if self.test_env and self.test_env_command.get("no_collision_handle_and_small", False):
            for idx in [2, 3, 4]:
                p.setCollisionFilterPair(
                    self._top_drawer,
                    self._small_obj,
                    idx,
                    -1,
                    enableCollision=False,
                    physicsClientId=self._uid)

    def _load_table_small_objs(self, is_close_drawer=False):
        ## Small Object(s)
        self._init_objs_pos = []
        if self.test_env:
            low, high = np.array(self.test_env_command['small_object_pos_randomness']['low']), np.array(
                self.test_env_command['small_object_pos_randomness']['high'])
            random_position = self.test_env_command['small_object_pos'] + np.random.uniform(
                low=low, high=high)
            self._small_obj = self.spawn_small_object(
                object_position=random_position, rgba=self.configs['object_rgbs']['small_object'])
        else:
            objects_within_gripper_range = False
            tries = 0
            while(not objects_within_gripper_range):
                if self._small_obj:
                    p.removeBody(self._small_obj, physicsClientId=self._uid)

                self.get_obj_pnp_goals()
                possible_goals = [self.on_top_drawer_goal,
                                  self.in_drawer_goal, self.out_of_drawer_goal]
                if is_close_drawer:
                    possible_goals = [
                        self.on_top_drawer_goal, self.out_of_drawer_goal]
                pos = random.choice(possible_goals)
                self._init_objs_pos.append(pos)
                self._small_obj = self.spawn_small_object(
                    object_position=pos + np.array([0, 0, .1]), rgba=self.configs['object_rgbs']['small_object'])

                objects_within_gripper_range = True
                pos, _ = get_object_position(self._small_obj, physicsClientId=self._uid)
                if not (gripper_bounding_x[0] - .04 <= pos[0] and pos[0] <= gripper_bounding_x[1] + .04
                        and gripper_bounding_y[0] - .04 <= pos[1] and pos[1] <= gripper_bounding_y[1] + .04):
                    objects_within_gripper_range = False
                    break

                tries += 1
                if tries > 10:
                    break

    def _load_table_large_objs(self):
        ## Large Object
        if self.test_env:
            self.large_object_quadrant = self.test_env_command['large_object_quadrant']
            quadrant = slide_quadrants[self.large_object_quadrant]
            pos = np.array([quadrant[0], quadrant[1], self.large_obj_z]) + self.configs['table_pos_offset']
            self._large_obj = self.spawn_large_object(
                pos, self.configs['object_rgbs']['large_object'])
        else:
            large_object_within_gripper_range = False
            tries = 0
            while(not large_object_within_gripper_range):
                if tries > 0:
                    p.removeBody(self._large_obj, physicsClientId=self._uid)
                self._large_obj = None

                large_object_quadrant_opts = list(
                    set([0, 1, 2, 3]) - set([self.top_drawer_quadrant]))
                if self.handle_more_open_than_closed():
                    for opt in large_object_quadrant_opts:
                        # oc = np.linalg.norm(np.array(slide_quadrants[opt]) - self.get_drawer_handle_future_pos(td_open_coeff)[:2]) < .1
                        # cc = np.linalg.norm(np.array(slide_quadrants[opt]) - self.get_drawer_handle_future_pos(td_close_coeff)[:2]) < .1
                        df = np.linalg.norm(np.array(slide_quadrants[opt]) - get_drawer_frame_pos(
                            self._top_drawer, physicsClientId=self._uid)[:2]) < .15
                        bf = np.linalg.norm(np.array(slide_quadrants[opt]) - get_drawer_bottom_pos(
                            self._top_drawer, physicsClientId=self._uid)[:2]) < .168
                        #so = np.linalg.norm(np.array(slide_quadrants[opt]) - self.get_object_pos(self._small_obj)[:2]) < .1
                        if df or bf:
                            large_object_quadrant_opts.remove(opt)
                ## Bug where drawer in first quadrant pointing towards camera collides with large object in in fourth quadrant
                if self.top_drawer_quadrant == 0 and self.drawer_yaw > 60 and self.drawer_yaw < 90 and 3 in large_object_quadrant_opts:
                    large_object_quadrant_opts.remove(3)
                if len(large_object_quadrant_opts) == 0:
                    large_object_quadrant_opts = list(
                        set([0, 1, 2, 3]) - set([self.top_drawer_quadrant]))
                #print("OPTS: ", large_object_quadrant_opts)
                self.large_object_quadrant = random.choice(
                    large_object_quadrant_opts)

                quadrant = slide_quadrants[self.large_object_quadrant]
                pos = np.array([quadrant[0], quadrant[1], self.large_obj_z]) + self.configs['table_pos_offset']
                self._large_obj = self.spawn_large_object(
                    pos, self.configs['object_rgbs']['large_object'])

                large_object_within_gripper_range = True
                if not (gripper_bounding_x[0] - .1 <= pos[0] and pos[0] <= gripper_bounding_x[1] + .1
                        and gripper_bounding_y[0] - .1 <= pos[1] and pos[1] <= gripper_bounding_y[1] + .1):
                    objects_within_gripper_range = False
                    break

                tries += 1
                if tries > 10:
                    break

    def sample_quat(self):
        return deg_to_quat(np.array([random.randint(0, 360), random.randint(0, 360), random.randint(0, 360)]), physicsClientId=self._uid)

    def spawn_small_object(self, object_position=None, quat=None, rgba=[0, 1, 0, 1], scale=2):
        # Pick object if necessary and save information
        assert object_position is not None

        self.obj_yaw = random.uniform(0, 360)

        q = deg_to_quat([0, 0, self.obj_yaw], physicsClientId=self._uid)
        obj = bullet.objects.drawer_lego(
            pos=object_position, quat=q, rgba=rgba, scale=scale, physicsClientId=self._uid)

        # Allow the objects to land softly in low gravity
        p.setGravity(0, 0, -1, physicsClientId=self._uid)
        for _ in range(100):
            bullet.step(physicsClientId=self._uid)
        # After landing, bring to stop
        p.setGravity(0, 0, -10, physicsClientId=self._uid)
        for _ in range(100):
            bullet.step(physicsClientId=self._uid)

        return obj

    def spawn_large_object(self, object_position=None, rgba=[0, 1, 0, 1], scale=1.4):
        assert object_position is not None

        obj = bullet.objects.cylinder(
            pos=object_position,
            quat=deg_to_quat([0, 0, 0]),
            rgba=rgba,
            scale=scale,
            physicsClientId=self._uid
        )

        return obj

    def _format_action(self, *action):
        if len(action) == 1:
            action = np.clip(action[0], a_min=-1, a_max=1)
            delta_pos, delta_yaw, gripper = action[:3], action[3:4], action[-1]
        elif len(action) == 3:
            action[0] = np.clip(action[0], a_min=-1, a_max=1)
            action[1] = np.clip(action[1], a_min=-1, a_max=1)
            action[2] = np.clip(action[2], a_min=-1, a_max=1)
            delta_pos, delta_yaw, gripper = action[0], action[1], action[2]
        else:
            raise RuntimeError('Unrecognized action: {}'.format(action))

        # Don't rotate gripper if its turned too far
        # curr_angle = self.get_end_effector_theta()[2]
        # if 0 < curr_angle and curr_angle < 90:
        #     pass
        # elif -45 < curr_angle and curr_angle < 0:
        #     delta_yaw[0] = max(0, delta_yaw[0]) 
        # #elif -90 < curr_angle and curr_angle < -45
        # else:
        #     delta_yaw[0] = min(0, delta_yaw[0])


        # Don't move downwards if gripper too low
        # curr_pos = self.get_end_effector_pos()[2]
        # if curr_pos < -0.35 + self.configs['table_pos_offset'][2]:
        #     delta_pos[2] = max(0, delta_pos[2])

        delta_angle = [0, 0, delta_yaw[0]]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def step(self, *action):
        # Get positional information
        pos = bullet.get_link_state(
            self._sawyer, self._end_effector, 'pos', physicsClientId=self._uid)
        curr_angle = bullet.get_link_state(
            self._sawyer, self._end_effector, 'theta', physicsClientId=self._uid)
        default_angle = quat_to_deg(self.default_theta)

        # Keep necesary degrees of theta fixed
        angle = np.append(default_angle[:2], [curr_angle[2]])

        # If angle is part of action, use it
        delta_pos, delta_angle, gripper = self._format_action(*action)
        angle += delta_angle * self._ddeg_scale

        # Magic Grasp
        if gripper > 0 and self.grasp_constraint is None and np.linalg.norm(self.get_object_pos(self._small_obj)[:2] - self.get_end_effector_pos()[:2]) < 0.04:
            self.grasp_constraint = p.createConstraint(
                parentBodyUniqueId=self._sawyer,
                parentLinkIndex=24,
                childBodyUniqueId=self._small_obj,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                physicsClientId=self._uid,
            )
        elif gripper < 0 and self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint,
                               physicsClientId=self._uid)
            self.grasp_constraint = None

        # Task 31: Drawer can't open if can in front of it
        # if self.get_quadrant(self.get_object_pos(self._large_obj)) == 1:
        if self.test_env and self.test_env_command.get("drawer_hack", False):
            drawer_hack_quadrant = self.test_env_command.get("drawer_hack_quadrant", 1)
            if self.get_quadrant(self.get_object_pos(self._large_obj)) == drawer_hack_quadrant and self.top_drawer_handle_can_move:
                self.top_drawer_handle_can_move = False
                p.changeDynamics(
                    bodyUniqueId=self._top_drawer,
                    linkIndex=2,
                    mass=99999999,
                    physicsClientId=self._uid,
                )
            elif self.get_quadrant(self.get_object_pos(self._large_obj)) != drawer_hack_quadrant and not self.top_drawer_handle_can_move:
                self.top_drawer_handle_can_move = True
                p.changeDynamics(
                    bodyUniqueId=self._top_drawer,
                    linkIndex=2,
                    mass=.1,
                    physicsClientId=self._uid,
                )

        # Update position and theta
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)
        theta = deg_to_quat(angle, physicsClientId=self._uid)
        self._simulate(pos, theta, gripper)

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False

        return observation, reward, done, info

    def get_info(self):
        skill_id = None
        if self.curr_task == 'move_drawer':
            if self.drawer_skill == 'open':
                skill_id = 0
            else:
                skill_id = 1
        elif self.curr_task == 'move_obj_slide':
            skill_id = 5
        elif self.curr_task == 'move_obj_pnp':
            if self.obj_pnp_skill == 'top':
                skill_id = 2
            elif self.obj_pnp_skill == 'in':
                skill_id = 3
            elif self.obj_pnp_skill == 'out':
                skill_id = 4
            else:
                assert False, f"{self.obj_pnp_skill}"
        else:
            assert False, f"{self.curr_task}"
        return {
            'skill_id': skill_id
        }

    def get_success_metric(self, curr_state, goal_state, key=None):
        success = np.zeros((curr_state.shape[0], 1), dtype=int)
        if key == "overall":
            curr_pos = curr_state[:, 8:11]
            goal_pos = goal_state[:, 8:11]
            curr_pos_0 = curr_state[:, 11:14]
            goal_pos_0 = goal_state[:, 11:14]
            curr_pos_1 = curr_state[:, 14:17]
            goal_pos_1 = goal_state[:, 14:17]
            curr_pos_2 = curr_state[:, 17:20]
            goal_pos_2 = goal_state[:, 17:20]
            curr_pos_3 = curr_state[:, 20:23]
            goal_pos_3 = goal_state[:, 20:23]
            curr_pos_extra = curr_state[:, 23:32]
            goal_pos_extra = goal_state[:, 23:32]
            success = np.logical_and.reduce((
                self.drawer_done(curr_pos, goal_pos),
                self.obj_pnp_done(curr_pos_0, goal_pos_0, curr_pos_extra, goal_pos_extra),
                self.obj_pnp_done(curr_pos_1, goal_pos_1, curr_pos_extra, goal_pos_extra),
                self.obj_pnp_done(curr_pos_2, goal_pos_2, curr_pos_extra, goal_pos_extra),
                self.obj_slide_done(curr_pos_3, goal_pos_3))).astype(int)
        elif key == 'top_drawer':
            curr_pos = curr_state[:, 8:11]
            goal_pos = goal_state[:, 8:11]
            success = self.drawer_done(curr_pos, goal_pos).astype(int)
        elif key == 'obj_pnp':
            curr_pos_0 = curr_state[:, 11:14]
            goal_pos_0 = goal_state[:, 11:14]
            curr_pos_1 = curr_state[:, 14:17]
            goal_pos_1 = goal_state[:, 14:17]
            curr_pos_2 = curr_state[:, 17:20]
            goal_pos_2 = goal_state[:, 17:20]
            curr_pos_extra = curr_state[:, 23:32]
            goal_pos_extra = goal_state[:, 23:32]
            success = np.logical_and.reduce((
                self.obj_pnp_done(curr_pos_0, goal_pos_0, curr_pos_extra, goal_pos_extra),
                self.obj_pnp_done(curr_pos_1, goal_pos_1, curr_pos_extra, goal_pos_extra),
                self.obj_pnp_done(curr_pos_2, goal_pos_2, curr_pos_extra, goal_pos_extra))).astype(int)
        elif key == 'obj_pnp_0':
            curr_pos_0 = curr_state[:, 11:14]
            goal_pos_0 = goal_state[:, 11:14]
            curr_pos_extra = curr_state[:, 23:32]
            goal_pos_extra = goal_state[:, 23:32]
            success = self.obj_pnp_done(
                curr_pos_0, goal_pos_0, curr_pos_extra, goal_pos_extra).astype(int)
        elif key == 'obj_pnp_1':
            curr_pos_1 = curr_state[:, 14:17]
            goal_pos_1 = goal_state[:, 14:17]
            curr_pos_extra = curr_state[:, 23:32]
            goal_pos_extra = goal_state[:, 23:32]
            success = self.obj_pnp_done(
                curr_pos_1, goal_pos_1, curr_pos_extra, goal_pos_extra).astype(int)
        elif key == 'obj_pnp_2':
            curr_pos_2 = curr_state[:, 17:20]
            goal_pos_2 = goal_state[:, 17:20]
            curr_pos_extra = curr_state[:, 23:32]
            goal_pos_extra = goal_state[:, 23:32]
            success = self.obj_pnp_done(
                curr_pos_2, goal_pos_2, curr_pos_extra, goal_pos_extra).astype(int)
        elif key == 'obj_slide':
            curr_pos = curr_state[:, 20:23]
            goal_pos = goal_state[:, 20:23]
            success = self.obj_slide_done(curr_pos, goal_pos).astype(int)
        else:
            pos = curr_state[:, 0:3]
            goal_pos = goal_state[:, 0:3]

            deg = quat_to_deg_batch(curr_state[:, 3:7])
            goal_deg = quat_to_deg_batch(goal_state[:, 3:7])

            if key == 'gripper_position':
                success = (np.linalg.norm(pos - goal_pos, axis=1, keepdims=True)
                           < self.gripper_pos_thresh).astype(int)
            elif key == 'gripper_rotation_roll':
                success = (self.norm_deg(deg[:, [0]], goal_deg[:, [0]]) < self.gripper_rot_thresh).astype(int)
            elif key == 'gripper_rotation_pitch':
                success = (self.norm_deg(
                    deg[:, [1]], goal_deg[:, [1]]) < self.gripper_rot_thresh).astype(int)
            elif key == 'gripper_rotation_yaw':
                success = (self.norm_deg(
                    deg[:, [2]], goal_deg[:, [2]]) < self.gripper_rot_thresh).astype(int)
            elif key == 'gripper_rotation':
                success = (np.sqrt(self.norm_deg(deg[:, [0]], goal_deg[:, [0]])**2 + self.norm_deg(
                    deg[:, [1]], goal_deg[:, [1]])**2 + self.norm_deg(deg[:, [2]], goal_deg[:, [2]])**2) < self.gripper_rot_thresh).astype(int)
            elif key == 'gripper':
                success = np.logical_and(
                    np.linalg.norm(pos - goal_pos, axis=1, keepdims=True) < self.gripper_pos_thresh,
                    np.sqrt(self.norm_deg(deg[:, [0]], goal_deg[:, [0]])**2 + self.norm_deg(deg[:, [1]], goal_deg[:, [1]])**2 + self.norm_deg(deg[:, [2]], goal_deg[:, [2]])**2) < self.gripper_rot_thresh
                ).astype(int)
        # if success_list is not None:
        #     success_list.extend(success[:, 0].tolist())

        return success

    def get_distance_metric(self, curr_state, goal_state, key=None):
        distance = np.full((curr_state.shape[0], 1), np.inf, dtype=float)
        if key == 'top_drawer':
            curr_pos = curr_state[:, 8:11]
            goal_pos = goal_state[:, 8:11]
            distance = np.linalg.norm(curr_pos - goal_pos, axis=1, keepdims=True)
        elif key == 'obj_pnp':
            curr_pos_0 = curr_state[:, 11:14]
            goal_pos_0 = goal_state[:, 11:14]
            curr_pos_1 = curr_state[:, 14:17]
            goal_pos_1 = goal_state[:, 14:17]
            curr_pos_2 = curr_state[:, 17:20]
            goal_pos_2 = goal_state[:, 17:20]
            distance = np.linalg.norm(curr_pos_0 - goal_pos_0, axis=1, keepdims=True) + \
                       np.linalg.norm(curr_pos_1 - goal_pos_1, axis=1, keepdims=True) + \
                       np.linalg.norm(curr_pos_2 - goal_pos_2, axis=1, keepdims=True)
        elif key == 'obj_pnp_0':
            curr_pos_0 = curr_state[:, 11:14]
            goal_pos_0 = goal_state[:, 11:14]
            distance = np.linalg.norm(curr_pos_0 - goal_pos_0, axis=1, keepdims=True)
        elif key == 'obj_pnp_1':
            curr_pos_1 = curr_state[:, 14:17]
            goal_pos_1 = goal_state[:, 14:17]
            distance = np.linalg.norm(curr_pos_1 - goal_pos_1, axis=1, keepdims=True)
        elif key == 'obj_pnp_2':
            curr_pos_2 = curr_state[:, 17:20]
            goal_pos_2 = goal_state[:, 17:20]
            distance = np.linalg.norm(curr_pos_2 - goal_pos_2, axis=1, keepdims=True)
        elif key == 'obj_slide':
            curr_pos = curr_state[:, 20:23]
            goal_pos = goal_state[:, 20:23]
            distance = np.linalg.norm(curr_pos - goal_pos, axis=1, keepdims=True)
        else:
            pos = curr_state[:, 0:3]
            goal_pos = goal_state[:, 0:3]
            deg = quat_to_deg_batch(curr_state[:, 3:7])
            goal_deg = quat_to_deg_batch(goal_state[:, 3:7])

            if key == 'gripper_position':
                distance = np.linalg.norm(pos - goal_pos, axis=1, keepdims=True)
            elif key == 'gripper_rotation_roll':
                distance = self.norm_deg(deg[:, [0]], goal_deg[:, [0]])
            elif key == 'gripper_rotation_pitch':
                distance = self.norm_deg(deg[:, [1]], goal_deg[:, [1]])
            elif key == 'gripper_rotation_yaw':
                distance = self.norm_deg(deg[:, [2]], goal_deg[:, [2]])
            elif key == 'gripper_rotation':
                distance = np.sqrt(self.norm_deg(deg[:, [0]], goal_deg[:, [0]])**2 + self.norm_deg(
                    deg[:, [1]], goal_deg[:, [1]])**2 + self.norm_deg(deg[:, [2]], goal_deg[:, [2]])**2)
        # if distance_list is not None:
        #     distance_list.extend(distance[:, 0].tolist())

        return distance

    def norm_deg(self, deg1, deg2):
        return np.minimum(np.linalg.norm((360 + deg1 - deg2) % 360, axis=1, keepdims=True),
                          np.linalg.norm((360 + deg2 - deg1) % 360, axis=1, keepdims=True))

    def get_gripper_deg(self, curr_state):
        quat = curr_state[:, 3:7]
        deg = quat_to_deg_batch(quat)
        # if roll_list is not None:
        #     roll_list.extend(deg[:, 0].tolist())
        # if pitch_list is not None:
        #     pitch_list.extend(deg[:, 1].tolist())
        # if yaw_list is not None:
        #     yaw_list.append(deg[:, 2].tolist())

        return deg

    def get_contextual_diagnostics(self, paths, contexts):
        # from roboverse.utils.diagnostics import create_stats_ordered_dict
        from multiworld.envs.env_util import create_stats_ordered_dict
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"

        success_keys = ["overall",
                        "top_drawer",
                        "obj_pnp",
                        "obj_pnp_0",
                        "obj_pnp_1",
                        "obj_pnp_2",
                        "obj_slide",
                        "gripper_position",
                        "gripper_rotation_roll",
                        "gripper_rotation_pitch",
                        "gripper_rotation_yaw",
                        "gripper_rotation", "gripper"]
        distance_keys = ["top_drawer",
                         "obj_pnp",
                         "obj_pnp_0",
                         "obj_pnp_1",
                         "obj_pnp_2",
                         "obj_slide",
                         "gripper_position",
                         "gripper_rotation_roll",
                         "gripper_rotation_pitch",
                         "gripper_rotation_yaw",
                         "gripper_rotation"]

        dict_of_success_arrays = {}
        # for k in success_keys:
        #     dict_of_success_arrays[k] = []

        dict_of_distance_arrays = {}
        # for k in distance_keys:
        #     dict_of_distance_lists[k] = []

        dict_of_length_arrays_1 = {}
        # for k in success_keys:
        #     dict_of_length_lists_1[k] = []

        dict_of_length_arrays_2 = {}
        # for k in success_keys:
        #     dict_of_length_lists_2[k] = []

        # ---------------------------------------------------------

        # import time
        # start_time = time.time()
        # for i in range(len(paths)):
        #     for t in range(len(paths[i]["observations"])):
        #         curr_obs = self.process(paths[i]["observations"][t][state_key])
        #         goal_obs = self.process(contexts[i][goal_key])
        #         for k in success_keys:
        #             self.get_success_metric(
        #                 self.process(curr_obs),
        #                 self.process(goal_obs),
        #                 key=k)
        #         for k in distance_keys:
        #             self.get_distance_metric(
        #                 curr_obs,
        #                 goal_obs,
        #                 key=k)
        # end_time = time.time()
        # print("Time of original implementation: {} sec".format(end_time - start_time))

        num_paths = len(paths)
        path_length = len(paths[0]["observations"])
        # start_time = time.time()
        curr_obses, goal_obses = [], []
        for i in range(len(paths)):
            path_curr_obses, path_goal_obses = [], []
            for t in range(len(paths[i]["observations"])):
                curr_obs = paths[i]["observations"][t][state_key]
                goal_obs = contexts[i][goal_key]
                path_curr_obses.append(curr_obs)
                path_goal_obses.append(goal_obs)
            curr_obses.append(path_curr_obses)
            goal_obses.append(path_goal_obses)
        curr_obses, goal_obses = \
            np.array(curr_obses).reshape([-1, curr_obs.shape[0]]), \
            np.array(goal_obses).reshape([-1, goal_obs.shape[0]])
        for k in success_keys:
            dict_of_success_array_k = self.get_success_metric(
                curr_obses, goal_obses, key=k)
            dict_of_success_arrays[k] = dict_of_success_array_k.reshape([num_paths, path_length])
        for k in distance_keys:
            dict_of_distance_array_k = self.get_distance_metric(
                curr_obses, goal_obses, key=k)
            dict_of_distance_arrays[k] = dict_of_distance_array_k.reshape([num_paths, path_length])
        # end_time = time.time()
        # print("Time of batch implementation: {} sec".format(end_time - start_time))

        for k in success_keys:
            diagnostics.update(create_stats_ordered_dict(
                goal_key + f"/final/{k}_success", dict_of_success_arrays[k][:, -1]))
        for k in distance_keys:
            diagnostics.update(create_stats_ordered_dict(
                goal_key + f"/final/{k}_distance", dict_of_distance_arrays[k][:, -1]))

        # ---------------------------------------------------------
        # for i in range(len(paths)):
        #     for t in range(len(paths[i]["observations"])):
        #         curr_obs = self.process(paths[i]["observations"][t][state_key])
        #         goal_obs = self.process(contexts[i][goal_key])
        #         for k in success_keys:
        #             self.get_success_metric(
        #                 self.process(curr_obs),
        #                 self.process(goal_obs),
        #                 success_list=dict_of_success_lists[k],
        #                 key=k)
        #         for k in distance_keys:
        #             self.get_distance_metric(
        #                 curr_obs,
        #                 goal_obs,
        #                 distance_list=dict_of_distance_lists[k],
        #                 key=k)

        for k in success_keys:
            diagnostics.update(create_stats_ordered_dict(
                goal_key + f"/{k}_success", dict_of_success_arrays[k]))
        for k in distance_keys:
            diagnostics.update(create_stats_ordered_dict(
                goal_key + f"/{k}_distance", dict_of_distance_arrays[k]))

        # ---------------------------------------------------------
        # gripper_rotation_roll_list = []
        # gripper_rotation_pitch_list = []
        # gripper_rotation_yaw_list = []
        # for i in range(len(paths)):
        #     for t in range(len(paths[i]["observations"])):
        #         curr_obs = paths[i]["observations"][t][state_key]
        #         self.get_gripper_deg(curr_obs, roll_list=gripper_rotation_roll_list,
        #                              pitch_list=gripper_rotation_pitch_list, yaw_list=gripper_rotation_yaw_list)

        degs = self.get_gripper_deg(curr_obses)
        gripper_rotation_roll_array, gripper_rotation_pitch_array, gripper_rotation_yaw_array = \
            degs[:, 0].reshape([num_paths, path_length]), \
            degs[:, 1].reshape([num_paths, path_length]), \
            degs[:, 2].reshape([num_paths, path_length])

        diagnostics.update(create_stats_ordered_dict(
            state_key + "/gripper_rotation_roll", gripper_rotation_roll_array))
        diagnostics.update(create_stats_ordered_dict(
            state_key + "/gripper_rotation_pitch", gripper_rotation_pitch_array))
        diagnostics.update(create_stats_ordered_dict(
            state_key + "/gripper_rotation_yaw", gripper_rotation_yaw_array))

        # ---------------------------------------------------------
        # for k in success_keys:
        #     for i in range(len(paths)):
        #         success = False
        #         goal_obs = self.process(contexts[i][goal_key])
        #         for t in range(len(paths[i]["observations"])):
        #             curr_obs = self.process(paths[i]["observations"][t][state_key])
        #             success = self.get_success_metric(
        #                 curr_obs,
        #                 goal_obs,
        #                 success_list=None,
        #                 key=k)
        #             if success:
        #                 break
        #
        #         dict_of_length_lists_1[k].append(t)
        #
        #         if success:
        #             dict_of_length_lists_2[k].append(t)
        #
        #     if len(dict_of_length_lists_2[k]) == 0:
        #         dict_of_length_lists_2[k] = [int(1e3 - 1)]

        # Reference: https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
        for k in success_keys:
            first_success_t = first_nonzero(
                dict_of_success_arrays[k], axis=1, invalid_val=-1)
            dict_of_length_arrays_1[k] = np.where(
                first_success_t != -1, first_success_t, path_length - 1)
            dict_of_length_arrays_2[k] = np.where(
                first_success_t != -1, first_success_t, int(1e3 - 1))

            diagnostics.update(create_stats_ordered_dict(
                goal_key + f"/{k}_length_inclusive",
                dict_of_length_arrays_1[k]))
            diagnostics.update(create_stats_ordered_dict(
                goal_key + f"/{k}_length_exclusive",
                dict_of_length_arrays_2[k]))

        return diagnostics

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.env_obs_img_dim, self.env_obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0, physicsClientId=self._uid)

        if self.downsample:
            img = Image.fromarray(np.uint8(img), 'RGB').resize(
                self.image_shape, resample=Image.ANTIALIAS)
            img = np.uint8(np.array(img))

            if self.render_depth:
                depth = Image.fromarray(depth, 'F').resize(
                    self.image_shape, resample=Image.ANTIALIAS)
                depth = np.array(depth)
                depth = np.clip(depth, a_min=0, a_max=1)

            if self.render_segmentation:
                segmentation = Image.fromarray(segmentation, 'I').resize(
                    self.image_shape, resample=Image.ANTIALIAS)
                segmentation = np.array(segmentation)

        render_list = [np.uint8(img)]
        if self.render_depth:
            depth = np.uint8(depth * 255.0)[:, :, None]
            render_list.append(depth)
        if self.render_segmentation:
            ## TODO(patrick): we can only do this because we have 9 objects and (9-1) * 31 < 255.
            segmentation = np.uint8((segmentation + 1) * 31)[:, :, None]
            render_list.append(segmentation)

        obs = np.concatenate(render_list, axis=2)
        if self._transpose_image:
            obs = np.transpose(obs, (2, 0, 1))

        return obs

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def get_reward(self, info=None, print_stats=False):
        curr_state = self.process(self.get_observation()['state_achieved_goal'])
        goal_state = self.process(self.goal_state)
        td_success = float(np.squeeze(self.get_success_metric(
            curr_state, goal_state, key='top_drawer')))
        obj_pnp_success = float(np.squeeze(self.get_success_metric(
            curr_state, goal_state, key='obj_pnp')))
        obj_slide_success = float(np.squeeze(self.get_success_metric(
            curr_state, goal_state, key='obj_slide')))
        if print_stats:
            print('-----------------')
            print('Top Drawer: ', td_success)
            print('Obj Pnp: ', obj_pnp_success)
            print('Obj Slide: ', obj_slide_success)
        reward = td_success + obj_pnp_success + obj_slide_success
        return reward

    def process(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def compute_reward(self, states, actions, next_states, contexts):
        state_observation = self.process(next_states['state_observation'])
        state_desired_goal = self.process(contexts['state_desired_goal'])
        # B = state_observation.shape[0]
        # rewards = np.zeros((B, 1))
        # for i in range(B):
        #     curr_state = state_observation[i]
        #     goal_state = state_desired_goal[i]
        #     td_success = self.get_success_metric(
        #         curr_state, goal_state, key='top_drawer')
        #     obj_pnp_success = self.get_success_metric(
        #         curr_state, goal_state, key='obj_pnp')
        #     obj_slide_success = self.get_success_metric(
        #         curr_state, goal_state, key='obj_slide')
        #     success = td_success and obj_pnp_success and obj_slide_success
        #     rewards[i] = success - 1
        td_success = self.get_success_metric(
            state_observation, state_desired_goal, key='top_drawer')
        obj_pnp_success = self.get_success_metric(
            state_observation, state_desired_goal, key='obj_pnp')
        obj_slide_success = self.get_success_metric(
            state_observation, state_desired_goal, key='obj_slide')
        success = np.logical_and.reduce((
            td_success, obj_pnp_success, obj_slide_success))
        rewards = (success - 1).astype(float)

        return rewards

    def sample_goals(self):
        if self.test_env and self.use_test_env_command_sequence:
            task, task_info = self.test_env_command['command_sequence'][self.reset_counter]
            if task == 'move_drawer':
                self.update_obj_pnp_goal()
                self.update_drawer_goal(task_info)
                self.update_obj_slide_goal()
            elif task == 'move_obj_pnp':
                self.update_obj_pnp_goal(task_info)
                self.update_drawer_goal()
                self.update_obj_slide_goal()
            elif task == 'move_obj_slide':
                self.update_obj_pnp_goal()
                self.update_drawer_goal()
                self.update_obj_slide_goal(task_info)
            else:
                assert False, 'not a valid task'
        else:
            self.update_obj_pnp_goal()
            self.update_drawer_goal()
            self.update_obj_slide_goal()

            self.get_obj_pnp_goals()
            obj_in_drawer = self.get_drawer_objs()[0]
            opts = ['move_obj_slide', 'move_obj_pnp', 'move_drawer']
            if np.linalg.norm(self.obj_slide_goal - self.get_object_pos(self._large_obj)) < .0001:
                opts.remove('move_obj_slide')
                
            # # Object in drawer
            # if obj_in_drawer is not None:
            #     opts.remove('move_drawer')
            # # Object blocks drawer
            # obj_pos, _ = get_object_position(
            #     self._small_obj, physicsClientId=self._uid)
            # large_obj_pos = self.get_object_pos(self.obj_slide)
            # for base_pos in [self.get_drawer_handle_future_pos(td_open_coeff), self.get_td_handle_pos()]:
            #     for offset in [i * self.obj_thresh / 4 for i in range(4+1)]:
            #         no_obj_center = base_pos - offset * \
            #             np.array([np.sin((self.drawer_yaw+180) * np.pi / 180), -
            #                       np.cos((self.drawer_yaw+180) * np.pi / 180), 0])
            #         if np.linalg.norm(obj_pos[:2] - no_obj_center[:2]) < self.obj_thresh + .02 or np.linalg.norm(large_obj_pos[:2] - no_obj_center[:2]) < self.obj_thresh + .02:
            #             if 'move_drawer' in opts:
            #                 opts.remove('move_drawer')
            #             break
            #     else:
            #         continue
            #     break
            # # Open drawer blocks obj
            # if np.linalg.norm(self.obj_slide_goal[:2] - self.get_td_handle_pos()[:2]) < self.obj_thresh:
            #     if 'move_obj_slide' in opts:
            #         opts.remove('move_obj_slide')

            if self.fixed_task:
                fixed_task = 'move_drawer' if self.fixed_task in [
                    'open_drawer', 'close_drawer'] else self.fixed_task
                if self.fixed_task == 'move_obj_slide' and 'move_obj_slide' not in opts:
                    target_quadrant = random.choice([(self.large_object_quadrant - 1) %
                                                     4, (self.large_object_quadrant + 1) % 4])
                    self.update_obj_slide_goal({
                        'target_quadrant': target_quadrant,
                    })
                task = fixed_task
            else:
                if len(opts) == 0:
                    opts = ['move_drawer']

                task = random.choice(opts)

        self.update_goal_state()
        return task

    def reset_gripper(self):
        action = np.array([0, 0, 1, 0, -1])
        for _ in range(10):
            self.step(action)

        # Sample and load starting positions
        if self.test_env and 'init_pos' in self.test_env_command:
            init_pos = self.test_env_command['init_pos']
        else:
            init_pos = np.array(self._pos_init)

        if self.test_env and 'init_theta' in self.test_env_command:
            init_theta = bullet.deg_to_quat(
                self.test_env_command['init_theta'])
        else:
            init_theta = self.default_theta
        
        if self.random_init_gripper_pos:
            init_pos = [
                np.random.uniform(gripper_bounding_x[0], gripper_bounding_x[1]),
                np.random.uniform(gripper_bounding_y[0], gripper_bounding_y[1]),
                np.random.uniform(-.25, gripper_bounding_z[1]),
            ]
        if self.random_init_gripper_yaw:
            init_theta = bullet.deg_to_quat([
                180,
                0,
                np.random.uniform(-180, 180)
            ])
        if self.random_init_gripper_yaw_discrete:
            init_theta = bullet.deg_to_quat([
                180,
                0,
                random.choice([0, 90])
            ])

        bullet.position_control(self._sawyer,
                                self._end_effector,
                                init_pos,
                                init_theta,
                                physicsClientId=self._uid)

    def reset(self):
        if self.use_multiple_goals:
            self.test_env_seed = np.random.choice(
                list(self.test_env_commands.keys()))
            self.test_env_command = self.test_env_commands[self.test_env_seed]

        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint,
                               physicsClientId=self._uid)
            self.grasp_constraint = None

        if self.expl:
            self.reset_counter += 1
            self.reset_gripper_counter += 1
            if self.reset_interval == self.reset_counter:
                self.reset_counter = 0
                self.reset_gripper_counter = 0
                self.curr_task = None
            else:
                self.curr_task = self.sample_goals()
                if self.reset_gripper_interval == self.reset_gripper_counter:
                    self.reset_gripper_counter = 0
                    self.reset_gripper()
                return self.get_observation()

        else:
            self.trajectory_done = False

        ## Null objects
        self._small_obj = None
        self._large_obj = None

        # Load Environment
        self._load_new_env_config()
        bullet.reset(physicsClientId=self._uid)
        bullet.setup_headless(
            self._timestep, solver_iterations=self._solver_iterations, physicsClientId=self._uid)
        self._load_table()
        self._format_state_query()

        if self.curr_task == None:
            self.curr_task = self.sample_goals()

        self.reset_gripper()

        # Move to starting positions
        action = np.array([0 for i in range(self.DoF)] + [-1])
        for _ in range(3):
            self.step(action)

        return self.get_observation()

    def format_obs(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos', physicsClientId=self._uid)
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos', physicsClientId=self._uid)
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)
        hand_theta = bullet.get_link_state(self._sawyer, self._end_effector,
                                           'theta', quat_to_deg=False, physicsClientId=self._uid)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        top_drawer_pos = self.get_td_handle_pos()

        obj0_pos = [0, 0, 0]
        obj1_pos = self.get_object_pos(self._small_obj)
        obj2_pos = [0, 0, 0]

        large_obj_pos = self.get_object_pos(self._large_obj)

        #(hand_pos, hand_theta, gripper, td_pos, obj0_pos, obj1_pos, obj2_pos, large_obj_pos, on_top_drawer_goal, in_drawer_goal, out_of_drawer_goal)
        #(3, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3)
        observation = np.concatenate((
            end_effector_pos, hand_theta, gripper_tips_distance,
            top_drawer_pos,
            obj0_pos, obj1_pos, obj2_pos, large_obj_pos,
            self.on_top_drawer_goal, self.in_drawer_goal, self.out_of_drawer_goal,
        ))

        obs_dict = dict(
            observation=observation,
            state_observation=observation,
            desired_goal=self.goal_state.copy(),
            state_desired_goal=self.goal_state.copy(),
            achieved_goal=observation,
            state_achieved_goal=observation,
        )

        return obs_dict

    ### Helper Functions
    def get_drawer_handle_future_pos(self, coeff):
        drawer_frame_pos = get_drawer_frame_pos(
            self._top_drawer, physicsClientId=self._uid)
        return drawer_frame_pos + coeff * np.array([np.sin(self.drawer_yaw * np.pi / 180), -np.cos(self.drawer_yaw * np.pi / 180), 0])

    def handle_more_open_than_closed(self):
        drawer_handle_close_pos = self.get_drawer_handle_future_pos(
            td_close_coeff)
        drawer_handle_open_pos = self.get_drawer_handle_future_pos(
            td_open_coeff)
        drawer_handle_pos = self.get_td_handle_pos()
        return np.linalg.norm(drawer_handle_open_pos - drawer_handle_pos) < np.linalg.norm(drawer_handle_close_pos - drawer_handle_pos)

    def get_td_handle_pos(self):
        return np.array(get_drawer_handle_pos(self._top_drawer, physicsClientId=self._uid))

    def drawer_done(self, curr_pos, goal_pos):
        # if curr_pos.size == 0 or goal_pos.size == 0:
        #     return np.empty((curr_pos.shape[0], 1))
        # else:
        #     return np.linalg.norm(curr_pos - goal_pos,
        #                           axis=1, keepdims=True) < self.drawer_thresh
        return np.linalg.norm(curr_pos - goal_pos,
                              axis=1, keepdims=True) < self.drawer_thresh

    def obj_pnp_done(self, curr_pos, goal_pos, curr_pos_extra, goal_pos_extra):
        on_top_drawer_goal_pos = goal_pos_extra[:, 0:3]
        in_drawer_goal_pos = goal_pos_extra[:, 3:6]
        out_of_drawer_goal_pos = goal_pos_extra[:, 6:9]

        if self.use_inverse_pnp_metric:
            success = np.linalg.norm(curr_pos - on_top_drawer_goal_pos, axis=1, keepdims=True) > self.obj_inverse_thresh
        else:
            goal_not_on_top = np.linalg.norm(
                goal_pos - on_top_drawer_goal_pos, axis=1, keepdims=True) > self.obj_thresh
            goal_not_in = np.linalg.norm(
                goal_pos - in_drawer_goal_pos, axis=1, keepdims=True) > self.obj_thresh

            not_on_top = np.linalg.norm(
                curr_pos - on_top_drawer_goal_pos, axis=1, keepdims=True) > self.obj_thresh
            not_in = np.linalg.norm(
                curr_pos - in_drawer_goal_pos, axis=1, keepdims=True) > self.obj_thresh
            goal_not_on_top_not_in_done = np.logical_and.reduce((
                not_on_top, not_in, np.linalg.norm(curr_pos[:, [2]] - goal_pos[:, [2]], axis=1, keepdims=True) < 0.01))
            other_done = np.logical_and(
                np.linalg.norm(curr_pos - goal_pos, axis=1, keepdims=True) < self.obj_thresh,
                np.linalg.norm(curr_pos[:, [2]] - goal_pos[:, [2]], axis=1, keepdims=True) < 0.01)

            success = np.where(np.logical_and(goal_not_on_top, goal_not_in),
                            goal_not_on_top_not_in_done,
                            other_done)

        return success 

    def obj_pnp_not_on_drawer_thresh(self, curr_pos, goal_pos, curr_pos_extra, goal_pos_extra, thresh):
        on_top_drawer_goal_pos = goal_pos_extra[:, 0:3]
        in_drawer_goal_pos = goal_pos_extra[:, 3:6]
        out_of_drawer_goal_pos = goal_pos_extra[:, 6:9]

        return np.linalg.norm(curr_pos - on_top_drawer_goal_pos, axis=1, keepdims=True) > thresh

    def obj_pnp_done_thresh(self, curr_pos, goal_pos, curr_pos_extra, goal_pos_extra, thresh):
        on_top_drawer_goal_pos = goal_pos_extra[:, 0:3]
        in_drawer_goal_pos = goal_pos_extra[:, 3:6]
        out_of_drawer_goal_pos = goal_pos_extra[:, 6:9]

        return np.linalg.norm(curr_pos - goal_pos, axis=1, keepdims=True) < thresh

    def obj_slide_done(self, curr_pos, goal_pos):
        # if curr_pos.shape[1] == 0 or goal_pos.shape[1] == 0:
        #     return np.zeros((curr_pos.shape[1], 1))
        # else:
        #     return self.get_quadrant(curr_pos) == self.get_quadrant(goal_pos)
        return self.get_quadrant(curr_pos) == self.get_quadrant(goal_pos)

    def get_obj_pnp_goals(self, task_info=None):
        ## Top Drawer Goal ##
        self.on_top_drawer_goal = np.array(
            list(get_drawer_frame_pos(self._top_drawer, physicsClientId=self._uid)))
        self.on_top_drawer_goal[2] = self.on_top_drawer_goal_z
        self.on_top_drawer_goal += self.configs['table_pos_offset']
        ## Randomly shift goal a little
        self.on_top_drawer_goal = self.on_top_drawer_goal

        ## In Drawer Goal ##
        self.in_drawer_goal = np.array(list(get_drawer_bottom_pos(self._top_drawer, physicsClientId=self._uid))) \
            - .025 * np.array([np.sin((self.drawer_yaw+180) * np.pi / 180), -
                              np.cos((self.drawer_yaw+180) * np.pi / 180), 0])
        self.in_drawer_goal[2] = self.in_drawer_goal_z
        self.in_drawer_goal += self.configs['table_pos_offset']

        ## Out of Drawer Goal ##
        if task_info is not None and task_info.get('target_position', None) is not None and task_info.get("target_location", None) == 'out':
            self.out_of_drawer_goal = np.array(task_info['target_position'])
        else:
            self.out_of_drawer_goal = None
            tries = 0
            while self.out_of_drawer_goal is None:
                offset = 0.0
                out_of_drawer_goal = np.array([random.uniform(gripper_bounding_x[0] + offset, gripper_bounding_x[1] - offset),
                                              random.uniform(gripper_bounding_y[0] + offset, gripper_bounding_y[1] - offset), self.out_of_drawer_goal_z])
                out_of_drawer_goal += self.configs['table_pos_offset']
                drawer_frame_far = np.linalg.norm(
                    out_of_drawer_goal[:2] - self.on_top_drawer_goal[:2]) > 0.1
                drawer_base_far = np.linalg.norm(
                    out_of_drawer_goal[:2] - self.in_drawer_goal[:2]) > 0.2
                if self._large_obj is not None:
                    large_obj_far = np.linalg.norm(
                        out_of_drawer_goal[:2] - self.get_object_pos(self._large_obj)[:2]) > 0.15
                else:
                    large_obj_far = True
                if drawer_frame_far and drawer_base_far and large_obj_far:
                    self.out_of_drawer_goal = out_of_drawer_goal
                else:
                    tries += 1
                    if tries > 25:
                        self.out_of_drawer_goal = np.array([random.uniform(gripper_bounding_x[0] + offset, gripper_bounding_x[1] - offset), random.uniform(
                            gripper_bounding_y[0] + offset, gripper_bounding_y[1] - offset), -0.34])
        self.out_of_drawer_goal[2] = -0.35201056
        #print(self.out_of_drawer_goal)

    def get_drawer_objs(self):
        obj_in_drawer = None
        obj_pos = self.get_object_pos(self._small_obj)
        if np.linalg.norm(self.in_drawer_goal - obj_pos) < self.obj_thresh:
            obj_in_drawer = self._small_obj

        obj_on_drawer = None
        obj_pos = self.get_object_pos(self._small_obj)
        if np.linalg.norm(self.on_top_drawer_goal - obj_pos) < self.obj_thresh:
            obj_on_drawer = self._small_obj

        return obj_in_drawer, obj_on_drawer

    def update_obj_slide_goal(self, task_info=None):
        if task_info is None:
            self.large_object_quadrant = self.get_quadrant(
                self.get_object_pos(self._large_obj))
            opts = [(self.large_object_quadrant - 1) %
                    4, (self.large_object_quadrant + 1) % 4]
            opts = [opt for opt in opts if self.top_drawer_quadrant != opt]
            if self.handle_more_open_than_closed():
                if (self.drawer_yaw >= 22 and self.drawer_yaw <= 68) or (self.drawer_yaw >= 112 and self.drawer_yaw <= 158):
                    opts = []
                for opt in opts:
                    #print(opt, np.linalg.norm(goal_slide_quadrants[opt][:2] - self.get_td_handle_pos()[:2]))
                    if np.linalg.norm(goal_slide_quadrants[opt][:2] - self.get_td_handle_pos()[:2]) < .14:
                        opts.remove(opt)
            #print("GOALS: ", opts)
            if len(opts) == 0:
                # opts = [(self.large_object_quadrant - 1) % 4, (self.large_object_quadrant + 1) % 4]
                # opts = [opt for opt in opts if self.top_drawer_quadrant != opt]
                goal_quadrant = None
            else:
                goal_quadrant = goal_slide_quadrants[random.choice(opts)]
        else:
            goal_quadrant = goal_slide_quadrants[task_info['target_quadrant']]
        self.obj_slide = self._large_obj
        self.obj_slide_goal = np.array(
            [goal_quadrant[0], goal_quadrant[1], self.large_obj_z]) + self.configs['table_pos_offset'] if goal_quadrant else self.get_object_pos(self._large_obj)

    def get_quadrant(self, pos):
        # if np.linalg.norm(slide_quadrants[0][0] - pos[0]) < np.linalg.norm(slide_quadrants[2][0] - pos[0]):
        #     if np.linalg.norm(slide_quadrants[0][1] - pos[1]) < np.linalg.norm(slide_quadrants[1][1] - pos[1]):
        #         return 0
        #     else:
        #         return 1
        # else:
        #     if np.linalg.norm(.17 - pos[1]) < np.linalg.norm(-.17 - pos[1]):
        #         return 3
        #     else:
        #         return 2
        squeeze = False
        if len(pos.shape) == 1:
            pos = pos.reshape(1, -1)
            squeeze = True

        cond0 = np.linalg.norm(slide_quadrants[0][0] - pos[:, [0]], axis=1, keepdims=True) < \
                np.linalg.norm(slide_quadrants[2][0] - pos[:, [0]], axis=1, keepdims=True)
        cond1 = np.linalg.norm(slide_quadrants[0][1] - pos[:, [1]], axis=1, keepdims=True) < \
                np.linalg.norm(slide_quadrants[1][1] - pos[:, [1]], axis=1, keepdims=True)
        cond2 = np.linalg.norm(.17 - pos[:, [1]], axis=1, keepdims=True) < \
                np.linalg.norm(-.17 - pos[:, [1]], axis=1, keepdims=True)

        cond1_quadrant = np.where(
            cond1,
            np.full((pos.shape[0], 1), 0, dtype=int),
            np.full((pos.shape[0], 1), 1, dtype=int)
        )
        cond2_quadrant = np.where(
            cond2,
            np.full((pos.shape[0], 1), 3, dtype=int),
            np.full((pos.shape[0], 1), 2, dtype=int)
        )
        cond0_quadrant = np.where(cond0, cond1_quadrant, cond2_quadrant)

        if squeeze:
            cond0_quadrant = int(np.squeeze(cond0_quadrant))

        return cond0_quadrant

    def update_obj_pnp_goal(self, task_info=None):
        self.get_obj_pnp_goals(task_info)
        obj_in_drawer, obj_on_drawer = self.get_drawer_objs()

        if task_info is None:
            obj_to_be_in_drawer = set()
            obj_to_be_on_drawer = set()
            obj_to_be_out_of_drawer = set()

            if obj_on_drawer:
                obj_to_be_out_of_drawer.add(obj_on_drawer)
            else:
                obj_to_be_on_drawer = set([self._small_obj])

            if obj_in_drawer:
                obj_to_be_out_of_drawer.add(obj_in_drawer)
            elif self.handle_more_open_than_closed():
                obj_to_be_in_drawer = set([self._small_obj])

            possible_goals = [
                (self.in_drawer_goal, list(obj_to_be_in_drawer), "in"),
                (self.on_top_drawer_goal, list(obj_to_be_on_drawer), "top"),
                (self.out_of_drawer_goal, list(obj_to_be_out_of_drawer), "out")
            ]

            random.shuffle(possible_goals)

            for (goal, can_interact_objs, skill) in possible_goals:
                if len(can_interact_objs) != 0:
                    self.obj_pnp = random.choice(can_interact_objs)
                    self.obj_pnp_goal = goal
                    self.obj_pnp_skill = skill

            if self.obj_pnp is None:
                self.obj_pnp = self._small_obj
                self.obj_pnp_goal = self.on_top_drawer_goal
                self.obj_pnp_skill = "top"
        else:
            target_location_to_goal = {
                "top": self.on_top_drawer_goal,
                "in": self.in_drawer_goal,
                "out": self.out_of_drawer_goal,
            }
            self.obj_pnp = self._small_obj
            self.obj_pnp_goal = target_location_to_goal[task_info['target_location']]
            self.obj_pnp_skill = task_info['target_location']

    def update_drawer_goal(self, task_info=None):
        if self.handle_more_open_than_closed():
            td_goal_coeff = td_close_coeff
            self.drawer_skill = 'close'
        else:
            td_goal_coeff = td_open_coeff
            self.drawer_skill = 'open'

        drawer_handle_goal_pos = self.get_drawer_handle_future_pos(
            td_goal_coeff)

        self.td_goal_coeff = td_goal_coeff
        self.td_goal = drawer_handle_goal_pos

    def update_goal_state(self):
        ## We save a bunch of zeros at the end to remain consistent with previous environments with had up to 3 objects
        obj_goal_state = [0 for _ in range(3)] + list(self.obj_pnp_goal) + [0 for _ in range(3)]
        self.goal_state = np.concatenate(
            [
                [0 for _ in range(8)],
                self.td_goal,
                obj_goal_state,
                self.obj_slide_goal,
                self.on_top_drawer_goal, self.in_drawer_goal, self.out_of_drawer_goal,
            ]
        )

    def get_object_pos(self, obj):
        return np.array(bullet.get_body_info(obj, quat_to_deg=False, physicsClientId=self._uid)['pos'])
    
    def get_random_large_action(self):
        action = np.concatenate((np.random.randint(2, size=3)*2-1, np.zeros(2,)))
        return action

    ### DEMO COLLECTING FUNCTIONS BEYOND THIS POINT ###
    def demo_reset(self):
        self.timestep = 0
        self.grip = -1.
        reset_obs = self.reset()

        #print('----Initial----')
        #self.get_reward(print_stats=True)
        return reset_obs

    def get_demo_action(self, first_timestep=False, return_done=False):
        if self.demo_num_ts:
            offset = self.demo_num_ts - self.timestep
        else:
            offset = 999
        task_dict = {
            'move_drawer': lambda: self.move_drawer(),
            'move_obj_pnp': lambda: self.move_obj_pnp(),
            'move_obj_slide': lambda: self.move_obj_slide(),
        }
        action, done = task_dict[self.curr_task]()

        if first_timestep:
            self.trajectory_done = False
            self.gripper_has_been_above = False
            self.gripper_in_right_position = False
            self.gripper_picked_object = False
            self.random_pick_offset = random.uniform(-.04, .04)
            action = np.array([0, 0, 1, 0])
        if done:
            if self.trajectory_done == False:
                self.trajectory_done = True

        if self.trajectory_done:
            action = np.array([0, 0, 1, 0, -1])
        else:
            action = np.append(action, [self.grip])
            action = np.random.normal(action, self.expert_policy_std)

        action = np.clip(action, a_min=-1, a_max=1)
        self.timestep += 1

        if return_done:
            return action, done
        return action

    def move_drawer(self, print_stages=False):
        self.grip = -1
        ee_pos = self.get_end_effector_pos()
        ee_yaw = self.get_end_effector_theta()[2]

        drawer_handle_pos = self.get_td_handle_pos()
        #drawer_frame_pos = get_drawer_frame_pos(self._top_drawer, physicsClientId=self._uid)
        #print((drawer_handle_pos - drawer_frame_pos)/np.array([np.sin((self.drawer_yaw+180) * np.pi / 180) , -np.cos((self.drawer_yaw+180) * np.pi / 180), 0]))
        ee_early_stage_goal_pos = drawer_handle_pos - td_offset_coeff * \
            np.array([np.sin((self.drawer_yaw+180) * np.pi / 180), -
                     np.cos((self.drawer_yaw+180) * np.pi / 180), 0])

        if 0 <= self.drawer_yaw < 90:
            goal_ee_yaw = self.drawer_yaw
        elif 90 <= self.drawer_yaw < 270:
            goal_ee_yaw = self.drawer_yaw - 180
        else:
            goal_ee_yaw = self.drawer_yaw - 360

        gripper_yaw_aligned = np.linalg.norm(goal_ee_yaw - ee_yaw) > 5
        gripper_pos_xy_aligned = np.linalg.norm(
            ee_early_stage_goal_pos[:2] - ee_pos[:2]) < .01
        gripper_pos_z_aligned = np.linalg.norm(
            ee_early_stage_goal_pos[2] - ee_pos[2]) < .0175
        gripper_above = ee_pos[2] >= -0.105
        if not self.gripper_has_been_above and gripper_above:
            self.gripper_has_been_above = True
        done = np.linalg.norm(self.td_goal - drawer_handle_pos) < 0.01

        # Stage 1: if gripper is too low, raise it
        if not self.gripper_in_right_position and not self.gripper_has_been_above:
            if print_stages:
                print('Stage 1')
            action = np.array([0, 0, 1, 0])
        # Do stage 2 and 3 at the same time
        elif not self.gripper_in_right_position and (gripper_yaw_aligned or not gripper_pos_xy_aligned):
            # Stage 2: align gripper yaw
            action = np.zeros((4,))
            if gripper_yaw_aligned:
                if print_stages:
                    print('Stage 2')
                if goal_ee_yaw > ee_yaw:
                    action[3] = 1
                else:
                    action[3] = -1
            # Stage 3: align gripper position with handle position
            if not gripper_pos_xy_aligned:
                if print_stages:
                    print('Stage 3')
                xy_action = (ee_early_stage_goal_pos - ee_pos) * 6 * 2
                action[0] = xy_action[0]
                action[1] = xy_action[1]
        # Stage 4: lower gripper around handle
        elif not self.gripper_in_right_position and (gripper_pos_xy_aligned and not gripper_pos_z_aligned):
            if print_stages:
                print('Stage 4')
            xy_action = (ee_early_stage_goal_pos - ee_pos) * 6 * 2
            action = np.array([xy_action[0], xy_action[1], xy_action[2]*3, 0])
        # Stage 5: open/close drawer
        else:
            if print_stages:
                print('Stage 5')
            if not self.gripper_in_right_position:
                self.gripper_in_right_position = True
            xy_action = self.td_goal - drawer_handle_pos
            s = 12  # 12 if self.drawer_skill == 'open' else 12
            action = s*np.array([xy_action[0], xy_action[1], 0, 0])
            # if self.drawer_skill == 'open':
            #     action *= 1.0

        if done:
            action = np.array([0, 0, 1, 0])

        if print_stages:
            print("drawer_yaw: ", self.drawer_yaw, ", drawer_frame_pos: ",
                  get_drawer_frame_pos(self._top_drawer, physicsClientId=self._uid))
        return action, done

    def move_obj_pnp(self, print_stages=False):
        self.goal_ee_yaw = 0
        align_cutoff = .035 if np.linalg.norm(self.get_object_pos(
            self._small_obj)[:2] - self.in_drawer_goal[:2]) < .02 else .05
        enclose_cutoff = .05
        cutoff = .025

        ee_pos = self.get_end_effector_pos()
        target_pos = self.get_object_pos(self.obj_pnp)
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < align_cutoff
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < enclose_cutoff
        done_xy = np.linalg.norm(
            target_pos[:2] - self.obj_pnp_goal[:2]) < cutoff
        done = done_xy
        above = ee_pos[2] >= -0.125

        target_pos += np.array([self.random_pick_offset, 0, 0])

        ee_yaw = self.get_end_effector_theta()[2]
        self.goal_ee_yaw = self.goal_ee_yaw % 360
        if 0 <= self.goal_ee_yaw < 90:
            goal_ee_yaw = self.goal_ee_yaw
        elif 90 <= self.goal_ee_yaw < 270:
            goal_ee_yaw = self.goal_ee_yaw - 180
        else:
            goal_ee_yaw = self.goal_ee_yaw - 360
        # goal_ee_yaw_opts = [self.goal_ee_yaw, self.goal_ee_yaw - 180, self.goal_ee_yaw + 180]
        # goal_ee_yaw = min(goal_ee_yaw_opts, key=lambda x : np.linalg.norm(x - ee_yaw))
        # if np.linalg.norm(self.goal_ee_yaw - ee_yaw) < np.linalg.norm(self.goal_ee_yaw - 180 + ee_yaw):
        #     goal_ee_yaw = self.goal_ee_yaw
        # else:
        #     goal_ee_yaw = self.goal_ee_yaw - 180
        # if 0 <= self.goal_ee_yaw < 90:
        #     goal_ee_yaw = self.goal_ee_yaw - 90
        # elif 90 <= self.goal_ee_yaw < 270:
        #     goal_ee_yaw = self.goal_ee_yaw - 90
        # else:
        #     goal_ee_yaw = self.goal_ee_yaw - 360 + 90
        gripper_yaw_aligned = np.linalg.norm(goal_ee_yaw - ee_yaw) > 10

        if not aligned and not above:
            if print_stages:
                print('Stage 1')
            action = np.array([0., 0., 1., 0.])
            self.grip = -1.

            if goal_ee_yaw > ee_yaw:
                action[3] = 1
            else:
                action[3] = -1
        elif (not self.gripper_picked_object and gripper_yaw_aligned) or not aligned:
            action = np.zeros((4,))
            if not self.gripper_picked_object and gripper_yaw_aligned:
                if print_stages:
                    print('Stage 2')
                if goal_ee_yaw > ee_yaw:
                    action[3] = 1
                else:
                    action[3] = -1
            if not aligned:
                if print_stages:
                    print('Stage 3')
                diff = (target_pos - ee_pos) * 3.0 * 2.0
                action[0] = diff[0]
                action[1] = diff[1]
                self.grip = -1.
        elif aligned and not enclosed and self.grip < 1:
            if print_stages:
                print('Stage 4')
            diff = target_pos - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 1.5
            self.grip = -1.
        elif enclosed and self.grip < 1:
            if print_stages:
                print('Stage 5')
            if not self.gripper_picked_object:
                self.gripper_picked_object = True
            diff = target_pos - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
            self.grip += 0.5
        elif not self.gripper_in_right_position and not above:
            if print_stages:
                print('Stage 6')
            action = np.array([0., 0., 1., 0.])
            self.grip = 1.
        elif not done_xy:
            if not self.gripper_in_right_position:
                self.gripper_in_right_position = True
            if print_stages:
                print('Stage 7')
            diff = self.obj_pnp_goal - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] = 0
            action *= 3.0
            self.grip = 1.
        else:
            if print_stages:
                print('Stage 9')
            action = np.array([0., 0., 0., 0.])
            self.grip = -1

        # print(aligned, above, done, enclosed, self.grip)
        # print(target_pos, ee_pos, goal)

        return action, done

    def move_obj_slide(self, print_stages=False):
        ee_pos = self.get_end_effector_pos()
        ee_yaw = self.get_end_effector_theta()[2]
        obj_pos = self.get_object_pos(self.obj_slide)
        goal_pos = self.obj_slide_goal

        vec = goal_pos[:2] - obj_pos[:2]
        direction = (np.arctan2(vec[1], vec[0]) * 180 / np.pi + 360 + 90) % 360
        goal_ee_yaw_opts = [direction, direction - 180,
                            direction + 180, direction - 360, direction + 360]
        goal_ee_yaw_opts = [o + 90 for o in goal_ee_yaw_opts]
        goal_ee_yaw = min(goal_ee_yaw_opts,
                          key=lambda x: np.linalg.norm(x - ee_yaw))

        # if 0 <= direction < 90:
        #     goal_ee_yaw = direction
        # elif 90 <= direction < 270:
        #     goal_ee_yaw = direction - 180
        # else:
        #     goal_ee_yaw = direction - 360
        ee_early_stage_goal_pos = obj_pos - 0.11 * \
            np.array([np.sin(direction * np.pi / 180), -
                     np.cos(direction * np.pi / 180), 0])

        gripper_yaw_aligned = np.linalg.norm(goal_ee_yaw - ee_yaw) > 5
        gripper_pos_xy_aligned = np.linalg.norm(
            ee_early_stage_goal_pos[:2] - ee_pos[:2]) < .005
        gripper_pos_z_aligned = np.linalg.norm(
            ee_early_stage_goal_pos[2] - ee_pos[2]) < .0375
        gripper_above = ee_pos[2] >= -0.105
        if not self.gripper_has_been_above and gripper_above:
            self.gripper_has_been_above = True

        done_xy = np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.05
        done = done_xy and np.linalg.norm(obj_pos[2] - goal_pos[2]) < 0.03

        # Stage 1: if gripper is too low, raise it
        if not self.gripper_has_been_above:
            if print_stages:
                print('Stage 1')
            action = np.array([0, 0, 1, 0])

            if goal_ee_yaw > ee_yaw:
                action[3] = 1
            else:
                action[3] = -1
        elif (not self.gripper_in_right_position and gripper_yaw_aligned) or (not self.gripper_in_right_position and not gripper_pos_xy_aligned):
            # Stage 2: align gripper yaw
            action = np.zeros((4,))
            if gripper_yaw_aligned:  # not self.gripper_in_right_position and gripper_yaw_aligned:
                if print_stages:
                    print('Stage 2')
                if goal_ee_yaw > ee_yaw:
                    action[3] = 1
                else:
                    action[3] = -1
            # Stage 3: align gripper position with handle position
            if not self.gripper_in_right_position and not gripper_pos_xy_aligned:  # not self.gripper_in_right_position and
                if print_stages:
                    print('Stage 3')
                xy_action = (ee_early_stage_goal_pos - ee_pos) * 6 * 2
                action[0] = xy_action[0]
                action[1] = xy_action[1]
                #print(action, np.linalg.norm(ee_early_stage_goal_pos[:2] - ee_pos[:2]))
        # Stage 4: lower gripper around handle
        elif gripper_pos_xy_aligned and not gripper_pos_z_aligned:  # not self.gripper_in_right_position and
            if print_stages:
                print('Stage 4')
            xy_action = (ee_early_stage_goal_pos - ee_pos) * 6 * 2
            action = np.array([xy_action[0], xy_action[1], xy_action[2]*3, 0])
        # Stage 5: open/close drawer
        else:
            if not self.gripper_in_right_position:
                self.gripper_in_right_position = True
            if print_stages:
                print('Stage 5')
            xy_action = goal_pos - obj_pos
            xy_action *= 6
            action = np.array([xy_action[0], xy_action[1], 0, 0])

        if done:
            action = np.array([0, 0, 1, 0])

        return action, done