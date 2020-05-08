import roboverse.bullet as bullet
import numpy as np
# from roboverse.envs.widow_grasp_downwards import WidowGraspDownwardsOneEnv
from roboverse.envs.widow200_grasp_v5 import Widow200GraspV5Env
from roboverse.utils.shapenet_utils import load_single_object


class WidowBoxPackingV2Env(Widow200GraspV5Env):
    OPENED_DOOR_ANGLE = 1.2
    PARTIAL_OPENED_DOOR_ANGLE = 0.4
    SUCCESS_THRESH = 0.065

    def get_handle_pos(self):
        handle_id = bullet.get_index_by_attribute(self._objects['box'], 'link_name', 'handle_r')
        handle_pos = np.array(bullet.get_link_state(self._objects['box'], handle_id, 'pos'))
        return handle_pos

    def get_door_angle(self):
        handle_pos = self.get_handle_pos()
        lid_joint_id = bullet.get_index_by_attribute(self._objects['box'], 'joint_name', 'lid_joint')
        lid_joint_pos = np.array(bullet.get_link_state(self._objects['box'], lid_joint_id, 'world_link_pos'))
        _, x, y = (handle_pos - lid_joint_pos) # y -> x, z -> y
        door_angle = np.arctan2(y, -x)
        return door_angle

    def get_info(self):
        object_pos = self.get_object_midpoint('lego')
        object_goal_distance = np.linalg.norm(object_pos - self._goal_pos)
        end_effector_pos = self.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(object_pos - end_effector_pos)
        gripper_handle_distance = np.linalg.norm(end_effector_pos - self.get_handle_pos())
        info = {
            'object_goal_distance': object_goal_distance,
            'object_gripper_distance': object_gripper_distance,
            'gripper_handle_distance': gripper_handle_distance,
            'door_angle': self.get_door_angle(),
            'handle_grasping_success': int(gripper_handle_distance < self.SUCCESS_THRESH),
            'door_opening_success': int(self.get_door_angle() > self.OPENED_DOOR_ANGLE),
            'partial_door_opening_success': int(self.get_door_angle() > self.PARTIAL_OPENED_DOOR_ANGLE),
        }
        return info

    def get_reward(self, info):
        lego_box_dist = np.linalg.norm(self.get_object_midpoint("lego") - self.get_object_midpoint("box"))
        door_angle = self.get_door_angle()
        ee_pos = np.array(self.get_end_effector_pos())
        handle_pos = self.get_handle_pos()
        gripper_handle_dist = np.clip(np.linalg.norm(ee_pos - handle_pos), 0, 1)
        if self._reward_type == 'sparse':
            # reward = int(door_angle > self.OPENED_DOOR_ANGLE)
            reward = (
                0.5*int(gripper_handle_dist < self.SUCCESS_THRESH) +
                0.5*int(door_angle > self.OPENED_DOOR_ANGLE)
            )
        elif self._reward_type == 'shaped':
            # door_angle_reward = np.clip(door_angle, 0, self.OPENED_DOOR_ANGLE) / self.OPENED_DOOR_ANGLE
            door_angle_reward = door_angle
            # reward = np.clip(door_angle_reward - gripper_handle_dist, 0, 1)
            reward = door_angle_reward - (5 * gripper_handle_dist)
        else:
            raise NotImplementedError

        return reward


if __name__ == "__main__":
    import roboverse
    import time
    env = roboverse.make("Widow200BoxPackingV2-v0",
                         gui=True,
                         observation_mode='state')

    object_ind = 0
    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        for _ in range(25):
            ee_pos = obs[:3]
            object_pos = obs[object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.
            # theta_action = np.random.uniform()
            # print(object_gripper_dist)
            if object_gripper_dist > dist_thresh and env._gripper_open:
                # print('approaching')
                action = (object_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.02:
                    action[2] = 0.0
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif env._gripper_open:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            else:
                # print('terminating')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.7])))

            action[:3] += np.random.normal(scale=0.1, size=(3,))
            # print(action)
            obs, rew, done, info = env.step(action)
            time.sleep(0.05)
            if done:
                print('reward: {}'.format(rew))
                break
