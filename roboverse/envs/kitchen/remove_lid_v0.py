# import roboverse.bullet as bullet
# import numpy as np
# import pybullet as p
# from gym.spaces import Box, Dict
# from collections import OrderedDict
# from roboverse.envs.kitchen.bridge_kitchen_v0 import BridgeKitchenVO
# from roboverse.bullet.misc import load_obj, load_urdf, deg_to_quat, quat_to_deg
# from roboverse.bullet.drawer_utils import *
# from roboverse.bullet.button_utils import *
# import os.path as osp
# import importlib.util
# import random
# import pickle
# import gym

# END_EFFECTOR_INDEX = 8
# RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.036, -0.036]
# RESET_JOINT_VALUES_GRIPPER_CLOSED = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.015, -0.015]
# RESET_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 10, 11]
# GUESS = 3.14
# JOINT_LIMIT_LOWER = [-3.14, -1.88, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
#                      -0.037]
# JOINT_LIMIT_UPPER = [3.14, 1.99, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]
# JOINT_RANGE = []
# for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER):
#     JOINT_RANGE.append(upper - lower)

# GRIPPER_LIMITS_LOW = JOINT_LIMIT_LOWER[-2:]
# GRIPPER_LIMITS_HIGH = JOINT_LIMIT_UPPER[-2:]


# # THINGS TO DEFINE: REWARD, POS LIM, OBJECT RESET LIM

# class PutLidVO(BridgeKitchenVO):

#     def __init__(self,
#                  reward_type='shaped',
#                  observation_mode='state',
#                  obs_img_dim=128,
#                  transpose_image=False,
#                  use_bounding_box=True,
#                  max_force=1000.,
#                  random_color_p=0.5,
#                  DoF=6,
#                  *args,
#                  **kwargs
#                  ):
#         """
#         Grasping env with a single object
#         :param reward_type: one of 'shaped', 'sparse'
#         :param observation_mode: state, pixels, pixels_debug
#         :param obs_img_dim: image dimensions for the observations
#         :param transpose_image: first dimension is channel when true
#         """
#         super().__init__(*args, **kwargs)

#         # Task Info #
#         self.task_objects = ['lid']
#         self.goal = np.array([2.035, -.28, -0.568])

#         self._pos_low = [1.95, -0.32, -0.55]
#         self._pos_high = [2.35, 0.05, -0.25]
#         self.robot_pos = np.array([2.45, -0.2, -0.55])
#         self.robot_deg = np.array([0,0,180])

#         # Camera Info #
#         camera_adjustment = np.array([-0.08, 0.14, 0.22])
#         self._view_matrix_obs = bullet.get_view_matrix(
#             target_pos=self.robot_pos + camera_adjustment,
#             distance=0.12, yaw=self.robot_deg[2] - 58, pitch=-40, roll=0, up_axis_index=2)
#     ### REWARD FUNCTIONS ###


#     def get_info(self):
#         delta_pos = [-3.38739228e-05, -4.18773970e-02, 1.62979934e-02]
#         goal_pos = self.get_object_pos('pot') + delta_pos

#         ee_pos = self.get_end_effector_pos()
#         obj_pos = self.get_object_pos('lid')
#         aligned = np.linalg.norm(obj_pos[:2] - ee_pos[:2]) < 0.05
#         enclosed = np.linalg.norm(obj_pos[2] - ee_pos[2]) < 0.08
#         above_goal = np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.05
#         done = np.linalg.norm(obj_pos - goal_pos) < 0.05
#         return {
#         'hand_object_aligned': aligned,
#         'object_grasped': enclosed * enclosed,
#         'object_above_goal': above_goal,
#         'task_achieved': done,
#         }

#     def get_reward(self, info):
#         return info['task_achieved'] - 1


#     ### SCRIPTED POLICY ###
#     def demo_reset(self):
#         reset_obs = self.reset()
#         self.timestep = 0
#         self.done = False
#         self.grip = -1.
#         self.default_height = self.get_end_effector_pos()[2] + 0.03
#         return reset_obs

#     def get_demo_action(self):

#         # Get xyz action
#         if self.done:
#             xyz_action, rot_action = self.maintain_hand()
#         else:
#             xyz_action, rot_action = self.move_lid()

#         # Get abc action
#         curr_angle = np.array(bullet.get_link_state(self._robot_id, END_EFFECTOR_INDEX, 'theta'))
#         abc_action = (self.default_angle - curr_angle) * 0.1
#         abc_action[2] = rot_action

#         action = np.concatenate((xyz_action, abc_action, [self.grip]))
#         action = np.clip(action, a_min=-1, a_max=1)
#         noisy_action = np.random.normal(action, 0.05)
#         noisy_action = np.clip(noisy_action, a_min=-1, a_max=1)
#         self.timestep += 1

#         return action, noisy_action


#     def move_lid(self):
#         delta_pos = [-3.38739228e-05, -4.18773970e-02, 0]#1.62979934e-02]
#         ee_pos = self.get_end_effector_pos()
#         obj_pos = self.get_object_pos('lid')

#         goal_pos = self.get_object_pos('pot') + delta_pos
#         curr_rot = self.get_object_angle('robot')[2]
#         obj_rot = self.get_object_angle('lid')[2]
#         target_pos = obj_pos + np.array([0.01, 0, 0.072]) #np.array([0.01, 0.025, 0.075])
#         aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.05
#         enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.05
#         above = ee_pos[2] >= self.default_height
#         drop_object = np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.05
#         self.hand_pos = np.array(ee_pos)

#         # if not aligned and not above and not drop_object:
#         #     print('Stage 1')
#         #     action = np.array([0.,0., 1.])
#         #     self.grip = -1.
#         if not aligned and not drop_object:
#             print('Stage 2')
#             action = (target_pos - ee_pos) * 3.0
#             action[2] = (self.default_height + 0.05) - ee_pos[2]
#             action *= 2.0
#             self.grip = -1.
#         elif aligned and not enclosed and not drop_object:
#             print('Stage 3')
#             action = target_pos - ee_pos
#             action[2] -= 0.03
#             action *= 3.0
#             action[2] *= 1.5
#             self.grip = -1.
#         elif enclosed and self.grip < 1. and not drop_object:
#             print('Stage 4')
#             action = target_pos - ee_pos
#             #action[2] *= 1.5
#             self.grip += 0.2
#         elif not above and not drop_object:
#             print('Stage 5')
#             action = np.array([0.,0., .18])
#             self.grip = 1.
#         elif not drop_object:
#             print('Stage 6')
#             action = (goal_pos - ee_pos) * 3.0
#             action[2] = (self.default_height + 0.05) - ee_pos[2]
#             self.grip = 1.
#         elif self.grip > -1.:
#             print('Stage 7')
#             action = (goal_pos - ee_pos) * 3.0
#             action[2] = (self.default_height + 0.05) - ee_pos[2]
#             self.grip -= 0.2
#         else:
#             print('Stage 8')
#             action = (goal_pos - ee_pos) * 3.0
#             action[2] = (self.default_height + 0.05) - ee_pos[2]
#             self.grip = -1.
#             self.done = True

#         if enclosed:
#             rot_action = 0
#         else:
#             rot_action = (obj_rot - curr_rot - 90) * 0.1

#         return action, rot_action

#     def maintain_hand(self):
#         action = self.hand_pos - np.array(self.get_end_effector_pos())
#         return action, 0

#     ### ENV DESIGN ###

#     def _reset_scene(self):
#         # Reset Environment Variables
#         self.lights_off = False
#         objects_present = list(self._objects.keys())
#         for key in objects_present:
#             p.removeBody(self._objects[key])

#         # Reset Sawyer
#         p.removeBody(self._robot_id)
#         self._robot_id = bullet.objects.widowx_250(
#             pos=self.robot_pos,
#             quat=deg_to_quat(self.robot_deg),
#             )

#         ### POT OBJECTS ###
#         self._objects['pot'] = load_obj(
#             self.shapenet_func('Objects/cooking_pot/models/model_vhacd.obj'),
#             self.shapenet_func('Objects/cooking_pot/models/model_normalized.obj'),
#             pos=[2.035, -.035, -0.45],
#             quat=deg_to_quat([90, 0, 0]),
#             scale=0.22,
#             )

#         2.035, -.035, -0.45
#         -.05

#         lid_pos = np.array([np.random.uniform(2.233, 2.243),
#                 np.random.uniform(-.135, -.135), -0.5])

#         self._objects['lid'] = bullet.objects.lid(
#             pos=lid_pos,
#             scale=0.082,
#             rgba=[.815, .84, .875, 1],
#             )

#         ### SINK OBJECTS ###
#         self._objects['soap'] = load_obj(
#             self.shapenet_func('Objects/dish_soap/models/model_vhacd.obj'),
#             self.shapenet_func('Objects/dish_soap/models/model_normalized.obj'),
#             pos=[1.95, 0.07, -0.5],
#             quat=deg_to_quat([90, 0, 0]),
#             scale=0.11,
#             rgba=[0.2, 1., 0.6, 1]
#             )

#         ### FRUIT BASKET OBJECTS ###
#         x,y,z = 0.02, -0.45, -0.9
#         self._objects['fruit_basket'] = load_obj(
#             self.shapenet_func('Objects/fruit_basket/models/model_vhacd.obj'),
#             self.shapenet_func('Objects/fruit_basket/models/model_normalized.obj'),
#             pos=[2.45 + y, 0.42 + z, -0.5],
#             quat=deg_to_quat([90, 0, 0]),
#             scale=0.25,
#             )


#         self._objects['orange'] = load_obj(
#             self.shapenet_func('Objects/orange/models/model_vhacd.obj'),
#             self.shapenet_func('Objects/orange/models/model_normalized.obj'),
#             pos=[2.45 + y, 0.41 + z, -0.42 + x],
#             scale=0.09,
#             )

#         self._objects['apple'] = load_obj(
#             self.shapenet_func('Objects/apple/models/model_vhacd.obj'),
#             self.shapenet_func('Objects/apple/models/model_normalized.obj'),
#             pos=[2.45 + y, 0.43 + z, -0.45 + x],
#             scale=0.07,
#             )

#         self._objects['banana'] = load_obj(
#             self.shapenet_func('Objects/banana/models/model_vhacd.obj'),
#             self.shapenet_func('Objects/banana/models/model_normalized.obj'),
#             pos=[2.45 + y, 0.42 + z, -0.38 + x],
#             quat=deg_to_quat([90, 0, -20]),
#             rgba=[1.,1.,0.,1.],
#             scale=0.12,
#             )

#         # Allow the objects to land softly in low gravity
#         p.setGravity(0, 0, -1)
#         for _ in range(100):
#             bullet.step()
#         # After landing, bring to stop
#         p.setGravity(0, 0, -10)
#         for _ in range(100):
#             bullet.step()

#     def _load_environment(self):
#         bullet.reset()
#         bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)

#         self._objects = {}
#         self._sensors = {}
#         self._robot_id = bullet.objects.widowx_250()
#         self._table = bullet.objects.table(scale=10., pos=[2.3, -.2, -7.2], rgba=[1., .97, .86,1])
#         self._workspace = bullet.Sensor(self._robot_id,
#             xyz_min=self._pos_low, xyz_max=self._pos_high,
#             visualize=False, rgba=[0,1,0,.1])

#         ### WALLS ###
#         tile_wall = load_urdf(self.shapenet_func('Furniture/tile_wall/tile_wall.urdf'),
#             pos=[1.78, -0.3, -0.215],
#             quat=deg_to_quat([180, 180, 90]),
#             scale=2.5,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(tile_wall, -1, 1, 0)

#         tile_wall = load_urdf(self.shapenet_func('Furniture/tile_wall/tile_wall.urdf'),
#             pos=[2.55, -0.82, -0.215],
#             quat=deg_to_quat([180, 180, 180]),
#             scale=2.5,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(tile_wall, -1, 1, 0)

#         tile_wall = load_urdf(self.shapenet_func('Furniture/tile_wall/tile_wall.urdf'),
#             pos=[2.25, 0.82, -0.215],
#             quat=deg_to_quat([180, 180, 0]),
#             scale=2.5,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(tile_wall, -1, 1, 0)

#         ### MARBLE TABELS ###
#         marble_table = load_urdf(self.shapenet_func('Furniture/marble_table/marble_table.urdf'),
#             pos=[2.2, 0.745, -0.765],
#             quat=deg_to_quat([90, 0, 180]),
#             scale=1.75,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(marble_table, -1, 1, 0)

#         marble_table = load_urdf(self.shapenet_func('Furniture/marble_table/marble_table.urdf'),
#             pos=[2.2, -0.765, -0.765],
#             quat=deg_to_quat([90, 0, 180]),
#             scale=1.75,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(marble_table, -1, 1, 0)

#         ### SINK ###
#         stovetop = load_urdf(self.shapenet_func('Furniture/stovetop/stovetop.urdf'),
#             pos=[2.035, -.15, -0.5514],
#             quat=deg_to_quat([90, 0, 90]),
#             scale=0.43,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(stovetop, -1, 1, 0)

#         wood_table = load_urdf(self.shapenet_func('Furniture/wood_table/wood_table.urdf'),
#             pos=[2.168, -.184, -0.583],
#             quat=deg_to_quat([90, 0, 270]),
#             scale=0.48,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(wood_table, -1, 1, 0)

#         stove_base = load_urdf(self.shapenet_func('Furniture/stove_base/stove_base.urdf'),
#             pos=[2.02, -.184, -0.5829],
#             quat=deg_to_quat([90, 0, 270]),
#             scale=0.48,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(stove_base, -1, 1, 0)

#         table = load_urdf(self.shapenet_func('Furniture/table/table.urdf'),
#             pos=[2.13, 0, -0.9],
#             quat=deg_to_quat([90, 0, 270]),
#             scale=1.0,
#             useFixedBase=True)
#         p.setCollisionFilterGroupMask(table, -1, 1, 0)




import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.kitchen.bridge_kitchen_v0 import BridgeKitchenVO
from roboverse.bullet.misc import load_obj, load_urdf, deg_to_quat, quat_to_deg
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


# THINGS TO DEFINE: REWARD, POS LIM, OBJECT RESET LIM

class RemoveLidVO(BridgeKitchenVO):

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
        self.task_objects = ['lid']
        self.goal = np.array([2.035, -.28, -0.568])

        self._pos_low = [1.95, -0.32, -0.55]
        self._pos_high = [2.1, 0.02, -0.2]
        self.robot_pos = np.array([2.2, -0.2, -0.55])
        self.robot_deg = np.array([0,0,180])

        # Camera Info #
        camera_adjustment = np.array([-0.08, 0.14, 0.22])
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=self.robot_pos + camera_adjustment,
            distance=0.12, yaw=self.robot_deg[2] - 58, pitch=-40, roll=0, up_axis_index=2)
    ### REWARD FUNCTIONS ###


    def get_info(self):
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.get_object_pos('lid')
        aligned = np.linalg.norm(obj_pos[:2] - ee_pos[:2]) < 0.05
        enclosed = np.linalg.norm(obj_pos[2] - ee_pos[2]) < 0.05
        above_goal = np.linalg.norm(obj_pos[:2] - self.goal[:2]) < 0.05
        picked_up = obj_pos[2] > -0.44
        done = np.linalg.norm(obj_pos - self.goal) < 0.05
        return {
        'hand_object_aligned': aligned,
        'object_grasped': enclosed * enclosed,
        'object_above_goal': above_goal,
        'picked_up': picked_up,
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
        self.default_height = self.get_end_effector_pos()[2] + 0.03
        return reset_obs

    # def get_demo_action(self):
    #     if self.done:
    #         xyz_action = self.maintain_hand()
    #     else:
    #         xyz_action = self.move_lid()

    #     curr_angle = np.array(bullet.get_link_state(self._robot_id, END_EFFECTOR_INDEX, 'theta'))
    #     abc_action = (self.default_angle - curr_angle) * 0.1
    #     print(abc_action)
    #     action = np.concatenate((xyz_action, abc_action, [self.grip]))
    #     action = np.random.normal(action, 0.05)
    #     action = np.clip(action, a_min=-1, a_max=1)
    #     self.timestep += 1

    #     return action

    def get_demo_action(self):

        # Get xyz action
        if self.done:
            xyz_action, rot_action = self.maintain_hand()
        else:
            xyz_action, rot_action = self.move_lid()

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


    def move_lid(self):
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.get_object_pos('lid')
        curr_rot = self.get_object_angle('robot')[2]
        obj_rot = self.get_object_angle('lid')[2]
        target_pos = obj_pos + np.array([0.01, 0.025, 0.075])
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.05
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.05
        above = ee_pos[2] >= self.default_height
        drop_object = np.linalg.norm(obj_pos[:2] - self.goal[:2]) < 0.05
        self.hand_pos = np.array(ee_pos)

        if not aligned and not above and not drop_object:
            #print('Stage 1')
            action = np.array([0.,0., 1.])
            self.grip = -1.
        elif not aligned and not drop_object:
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
        elif enclosed and self.grip < 1. and not drop_object:
            #print('Stage 4')
            action = target_pos - ee_pos
            self.grip += 0.2
        elif not above and not drop_object:
            #print('Stage 5')
            action = np.array([0.,0., .2])
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
            rot_action = (obj_rot - curr_rot - 90) * 0.1

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

        ### POT OBJECTS ###
        self._objects['pot'] = load_obj(
            self.shapenet_func('Objects/cooking_pot/models/model_vhacd.obj'),
            self.shapenet_func('Objects/cooking_pot/models/model_normalized.obj'),
            pos=[2.035, -.035, -0.45],
            quat=deg_to_quat([90, 0, 0]),
            scale=0.22,
            )

        self._objects['lid'] = bullet.objects.lid(
            pos=[2.035, -.077, -0.35],
            scale=0.082,
            rgba=[.815, .84, .875, 1],
            )

        ### SINK OBJECTS ###
        self._objects['soap'] = load_obj(
            self.shapenet_func('Objects/dish_soap/models/model_vhacd.obj'),
            self.shapenet_func('Objects/dish_soap/models/model_normalized.obj'),
            pos=[1.95, 0.07, -0.5],
            quat=deg_to_quat([90, 0, 0]),
            scale=0.11,
            rgba=[0.2, 1., 0.6, 1]
            )

        ### FRUIT BASKET OBJECTS ###
        x,y,z = 0.02, -0.45, -0.9
        self._objects['fruit_basket'] = load_obj(
            self.shapenet_func('Objects/fruit_basket/models/model_vhacd.obj'),
            self.shapenet_func('Objects/fruit_basket/models/model_normalized.obj'),
            pos=[2.45 + y, 0.42 + z, -0.5],
            quat=deg_to_quat([90, 0, 0]),
            scale=0.25,
            )


        self._objects['orange'] = load_obj(
            self.shapenet_func('Objects/orange/models/model_vhacd.obj'),
            self.shapenet_func('Objects/orange/models/model_normalized.obj'),
            pos=[2.45 + y, 0.41 + z, -0.42 + x],
            scale=0.09,
            )

        self._objects['apple'] = load_obj(
            self.shapenet_func('Objects/apple/models/model_vhacd.obj'),
            self.shapenet_func('Objects/apple/models/model_normalized.obj'),
            pos=[2.45 + y, 0.43 + z, -0.45 + x],
            scale=0.07,
            )

        self._objects['banana'] = load_obj(
            self.shapenet_func('Objects/banana/models/model_vhacd.obj'),
            self.shapenet_func('Objects/banana/models/model_normalized.obj'),
            pos=[2.45 + y, 0.42 + z, -0.38 + x],
            quat=deg_to_quat([90, 0, -20]),
            rgba=[1.,1.,0.,1.],
            scale=0.12,
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
