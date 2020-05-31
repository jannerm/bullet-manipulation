from roboverse.envs.widow200_grasp_v5 import Widow200GraspV5Env
import roboverse.bullet as bullet
import numpy as np
import gym

REWARD_FAIL = 0.0
REWARD_SUCCESS = 1.0


class Widow200GraspV6Env(Widow200GraspV5Env):

    def __init__(self,
                 *args,
                 scaling_local_list=[0.5],
                 **kwargs):
        super().__init__(*args,
            scaling_local_list=scaling_local_list,
            **kwargs)
        self.terminates = False
        self.scripted_traj_len = 25

    def step(self, action):
        action = np.asarray(action)
        pos = list(bullet.get_link_state(self._robot_id, self._end_effector, 'pos'))
        delta_pos = action[:3]
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        theta = list(bullet.get_link_state(self._robot_id, self._end_effector,
                                           'theta'))
        target_theta = theta
        delta_theta = action[3]
        target_theta = np.clip(target_theta, [0, 85, 137], [180, 85, 137])
        target_theta = bullet.deg_to_quat(target_theta)

        gripper_action = action[4]
        self._gripper_simulate(pos, target_theta, delta_theta, gripper_action)

        reward = self.get_reward({})
        info = {'grasp_success': 1.0 if reward > 0 else 0.0}

        observation = self.get_observation()
        self._prev_pos = bullet.get_link_state(self._robot_id, self._end_effector,
                                               'pos')
        done = False
        return observation, reward, done, info


if __name__ == "__main__":
    import roboverse
    import time
    env = roboverse.make("Widow200GraspV6-v0",
                         gui=True,
                         observation_mode='state',)

    object_ind = 0
    EPSILON = 0.05
    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        for _ in range(env.scripted_traj_len):
            ee_pos = obs[:3]
            object_pos = obs[object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            object_lifted = object_pos[2] > env._reward_height_thresh
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
            elif not object_lifted:
                # print('raise object upward')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            else:
                action = np.zeros((6,))

            action[:3] += np.random.normal(scale=0.1, size=(3,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print(action)
            obs, rew, done, info = env.step(action)
            time.sleep(0.05)
            if done:
                print('reward: {}'.format(rew))
                break

