from roboverse.envs.widow200_grasp_v5 import Widow200GraspV5Env
from roboverse.envs.rand_obj import RandObjEnv
import roboverse.bullet as bullet
import numpy as np
import gym

REWARD_FAIL = 0.0
REWARD_SUCCESS = 1.0


class Widow200GraspV6Env(Widow200GraspV5Env):

    def __init__(self,
                 *args,
                 object_names=('beer_bottle',),
                 scaling_local_list=[0.5],
                 use_tray=True,
                 **kwargs):
        self.object_names = object_names
        self.reward_height_threshold = -0.275
        kwargs['env_name'] = "Widow200GraspEnv"
        self._env_name = kwargs['env_name']
        self.use_tray = use_tray
        super().__init__(*args,
            object_names=self.object_names,
            scaling_local_list=scaling_local_list,
            **kwargs)
        self.terminates = False
        self.scripted_traj_len = 25

    def get_info(self):
        # assert self._num_objects == 1
        object_name = list(self._objects.keys())[0]
        object_info = bullet.get_body_info(self._objects[object_name],
                                           quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])
        ee_pos = np.array(self.get_end_effector_pos())

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)

        assert len(self._objects.keys()) == 2  # for now
        object_pos_list = []
        for object_name in self._objects.keys():
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            object_pos_list.append(object_pos)
        dist_btw_objects = np.linalg.norm(object_pos_list[0] - object_pos_list[1])

        info = dict(object_gripper_dist=object_gripper_dist,
                    dist_btw_objects=dist_btw_objects)
        return info

    def step(self, action):
        action = np.asarray(action)
        # In Grasp V6, the 6th action dim corresponds to a binary
        # move to reset command.
        if action[5] > 0.5:
            # currently nothing happens for large negative action[5] commands.
            bullet.move_to_neutral(self.RESET_JOINTS, self._robot_id)
        else:
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

        info = self.get_info()
        reward = self.get_reward(info)

        info['grasp_success'] = 0.0
        object_list = self._objects.keys()
        for object_name in object_list:
            if self.is_object_grasped(object_name):
                info['grasp_success'] = 1.0

        if self.target_object is not None:
            if self.is_object_grasped(self.target_object):
                info['grasp_success_target'] = 1.0
            else:
                info['grasp_success_target'] = 0.0

        info['push_success'] = 0.0
        object_list = self._objects.keys()
        if self.is_objects_close(object_list):
            info['push_success'] = 1.0

        observation = self.get_observation()
        self._prev_pos = bullet.get_link_state(self._robot_id, self._end_effector,
                                               'pos')
        done = False

        return observation, reward, done, info

class Widow200GraspV6RandObjEnv(RandObjEnv, Widow200GraspV6Env):
    """Grasping Env but with a random object each time."""

if __name__ == "__main__":
    import roboverse
    import time
    env = roboverse.make("Widow200GraspV6RandObj-v0",
                         gui=True,
                         observation_mode='pixels_debug',)

    object_ind = 0
    EPSILON = 0.05
    margin = 0.025

    def strint(x):
        return str(int(x))

    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)
        rewards = []

        for _ in range(env.scripted_traj_len):
            if isinstance(obs, dict):
                state_obs = obs[env.fc_input_key]
                obj_obs = obs[env.object_obs_key]

            ee_pos = state_obs[:3]
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]

            object_lifted = object_pos[2] > env._reward_height_thresh
            object_lifted_with_margin = object_pos[2] > (env._reward_height_thresh + margin)
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
            elif not object_lifted_with_margin:
                # print('raise object upward')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            else:
                tray_info = bullet.get_body_info(env._tray, quat_to_deg=False)
                tray_center = np.asarray(tray_info['pos'])
                action = (tray_center - ee_pos)[:2]
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0., 0.])))
                # action = np.zeros((6,))

            action[:3] += np.random.normal(scale=0.1, size=(3,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            obs, rew, done, info = env.step(action)
            time.sleep(0.05)

            print("info", info)
            rewards.append(rew)

        # print("="*10)
        print("".join(list(map(strint, rewards))))
