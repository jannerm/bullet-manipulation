import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.widow_grasp_downwards import WidowGraspDownwardsOneEnv
from roboverse.utils.shapenet_utils import load_single_object


class WidowBoxPackingOneEnv(WidowGraspDownwardsOneEnv):
    OPENED_DOOR_ANGLE = 1.2

    def get_door_angle(self):
        handle_id = bullet.get_index_by_attribute(self._objects['box'], 'link_name', 'handle_r')
        handle_pos = np.array(bullet.get_link_state(self._objects['box'], handle_id, 'pos'))
        lid_joint_id = bullet.get_index_by_attribute(self._objects['box'], 'joint_name', 'lid_joint')
        lid_joint_pos = np.array(bullet.get_link_state(self._objects['box'], lid_joint_id, 'world_link_pos'))
        _, x, y = (handle_pos - lid_joint_pos) # y -> x, z -> y
        door_angle = np.arctan2(y, -x)
        return door_angle

    def get_reward(self, info):
        lego_box_dist = np.linalg.norm(self.get_object_midpoint("lego") - self.get_object_midpoint("box"))
        door_angle = self.get_door_angle()
        ee_pos = np.array(self.get_end_effector_pos())
        handle_id = bullet.get_index_by_attribute(self._objects['box'], 'link_name', 'handle_r')
        handle_pos = np.array(bullet.get_link_state(self._objects['box'], handle_id, 'pos'))
        if self._reward_type == 'sparse':
            # reward = 0.5 * int(lego_box_dist < 0.1) + 0.5 * int(door_angle > 1.2)
            reward = int(door_angle > self.OPENED_DOOR_ANGLE)
        elif self._reward_type == 'shaped':
            # reward = np.clip(door_angle, 0, 1.2) - lego_box_dist
            # reward = np.clip(reward, 0, 1)
            door_angle_reward = np.clip(door_angle, 0, self.OPENED_DOOR_ANGLE) / self.OPENED_DOOR_ANGLE
            gripper_handle_dist = np.clip(np.linalg.norm(ee_pos - handle_pos), 0, 1)
            print("reward gripper_handle_dist", gripper_handle_dist)
            print("door_angle_reward - gripper_handle_dist", door_angle_reward - gripper_handle_dist)
            reward = np.clip(door_angle_reward - gripper_handle_dist, 0, 1)
        else:
            raise NotImplementedError

        return reward
