from roboverse.envs.widow200_grasp_v5 import Widow200GraspV5Env
import roboverse.bullet as bullet
import numpy as np


class Widow200GraspV5AndPlaceV0Env(Widow200GraspV5Env):

    def __init__(self,
                 *args,
                 goal_position=(0.78, -0.12, -0.22),
                 scaling_local_list=[0.5],
                 success_dist_threshold=0.04,
                 **kwargs):
        self._goal_position = np.asarray(goal_position)
        self._success_dist_threshold = success_dist_threshold
        super().__init__(*args,
            scaling_local_list=scaling_local_list,
            **kwargs)

    def get_reward(self, info):
        if self._reward_type == 'dense':
            reward = -1.0*info['object_goal_dist']
        elif self._reward_type == 'sparse':
            reward = float(info['object_goal_success'])
        else:
            print(self._reward_type)
            raise NotImplementedError
        return reward

    def get_info(self):
        assert self._num_objects == 1
        object_name = list(self._objects.keys())[0]
        object_info = bullet.get_body_info(self._objects[object_name],
                                           quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])
        object_goal_dist = np.linalg.norm(object_pos - self._goal_position)
        object_goal_success = object_goal_dist < self._success_dist_threshold
        info = dict(
            object_goal_dist=object_goal_dist,
            object_goal_success=object_goal_success)
        return info

    def step(self, action):
        action = np.asarray(action)
        pos = list(
            bullet.get_link_state(self._robot_id, self._end_effector, 'pos'))
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

        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        self._prev_pos = bullet.get_link_state(self._robot_id,
                                               self._end_effector,
                                               'pos')
        done = False
        return observation, reward, done, info


if __name__ == "__main__":
    import roboverse
    import time
    env = roboverse.make("Widow200GraspV5PlaceV0Env-v0",
                         gui=True,
                         reward_type='sparse',
                         observation_mode='pixels_debug',)

    object_ind = 0
    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        for _ in range(25):
            if isinstance(obs, dict):
                obs = obs['state']

            ee_pos = obs[:3]
            object_pos = obs[object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.
            object_goal_dist = np.linalg.norm(object_pos - env._goal_position)
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
            elif object_goal_dist > env._success_dist_threshold:
                print(object_goal_dist)
                action = (env._goal_position - object_pos)*7.0
                # action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            else:
                action = np.zeros((6,))

            action[:3] += np.random.normal(scale=0.1, size=(3,))
            # print(action)
            obs, rew, done, info = env.step(action)
            time.sleep(0.05)

        print('object pos: {}'.format(object_pos))
        print('reward: {}'.format(rew))
        print('distance: {}'.format(info['object_goal_dist']))
        print('--------------------')
