from roboverse.envs.widow200_grasp_v5_and_place_v0 import Widow200GraspV5AndPlaceV0Env
import roboverse.bullet as bullet
import numpy as np


class Widow200GraspV5BoxV0Env(Widow200GraspV5AndPlaceV0Env):

    def __init__(self,
                 *args,
                 success_dist_threshold=0.04,
                 **kwargs):
        kwargs['object_names'] = ('jar',)
        super().__init__(*args, **kwargs)
        self._object_position_high = (.82, -.04, -.20)
        self._object_position_low = (.78, -.125, -.20)
        self._success_dist_threshold = success_dist_threshold
        self._scaling_local_list = [0.3] # converted into dict below.
        self.set_scaling_dicts()
        self.set_box_pos_as_goal_pos()

    def _load_meshes(self):
        super()._load_meshes()
        self._box = bullet.objects.box_open_top()

    def get_reward(self, info):
        if self._reward_type == 'dense':
            reward = -1.0*info['object_goal_dist']
        elif self._reward_type == 'sparse':
            reward = float(info['object_goal_success'])
        else:
            print(self._reward_type)
            raise NotImplementedError
        return reward

    def set_box_pos_as_goal_pos(self):
        box_open_top_info = bullet.get_body_info(self._box, quat_to_deg=False)
        self._goal_position = np.asarray(box_open_top_info['pos'])

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
        return super().step(action)


if __name__ == "__main__":
    import roboverse
    import time
    env = roboverse.make("Widow200GraspV5BoxV0Env-v0",
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
