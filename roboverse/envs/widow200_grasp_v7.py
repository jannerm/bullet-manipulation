import numpy as np
from roboverse.envs.widow200_grasp_v6 import Widow200GraspV6Env
import roboverse.bullet as bullet

REWARD_FAIL = 0.0
REWARD_SUCCESS = 1.0


class Widow200GraspV7Env(Widow200GraspV6Env):
    def __init__(self,
                 *args,
                 **kwargs):
        self.use_goal_location_for_reward = True
        super().__init__(*args, **kwargs)

    def get_reward(self, info):
        object_list = self._objects.keys()
        reward = REWARD_FAIL
        end_effector_pos = np.asarray(self.get_end_effector_pos())
        for object_name in object_list:
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            object_gripper_distance = np.linalg.norm(
                object_pos - end_effector_pos)
            gripper_goal_distance= np.linalg.norm(
                end_effector_pos - self.gripper_goal_location)
            if object_gripper_distance < 0.07 and gripper_goal_distance < 0.03:
                reward = REWARD_SUCCESS
        reward = self.adjust_rew_if_use_positive(reward)
        return reward