import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.widow_base import WidowBaseEnv
from roboverse.utils.shapenet_utils import load_single_object


class WidowGraspUpwardsOneEnv(WidowBaseEnv):

    def __init__(self, goal_pos=(.7, 0.0,-0.26), *args, **kwargs):
        kwargs['downwards'] = False
        super().__init__(*args, **kwargs)
        self._goal_pos = goal_pos

    def _load_meshes(self):
        super()._load_meshes()
        self._objects = {
            'lego': bullet.objects.lego(),
            'bowl':   load_single_object('36ca3b684dbb9c159599371049c32d38',
                                         [.9, 0, -.28], quat=[0, 0, 0, 1],scale=0.5)[0]

        }

    def get_reward(self, observation):
        object_pos = self.get_object_midpoint('lego')
        if object_pos[2] > -0.1:
            reward = 1
        else:
            reward = 0
        return reward

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        return observation, reward, done, info

    def get_info(self):
        object_pos = self.get_object_midpoint('lego')
        object_goal_distance = np.linalg.norm(object_pos - self._goal_pos)
        end_effector_pos = self.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(
            object_pos - end_effector_pos)
        info = {
            'object_goal_distance': object_goal_distance,
            'object_gripper_distance': object_gripper_distance
        }
        return info

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[0], keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[1], keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        object_info = bullet.get_body_info(self._objects['lego'],
                                           quat_to_deg=False)
        object_pos = object_info['pos']
        object_theta = object_info['theta']

        return np.concatenate((end_effector_pos, gripper_tips_distance,
                               object_pos, object_theta))
