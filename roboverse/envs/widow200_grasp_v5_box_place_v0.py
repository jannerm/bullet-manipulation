from roboverse.envs.widow200_grasp_v5_and_place_v0 import Widow200GraspV5AndPlaceV0Env
import roboverse.bullet as bullet
import roboverse.utils as utils
import numpy as np


class Widow200GraspV5BoxPlaceV0Env(Widow200GraspV5AndPlaceV0Env):

    def __init__(self,
                 *args,
                 object_names=('jar',),
                 scaling_local_list=[0.3],
                 success_dist_threshold=0.04,
                 **kwargs):
        super().__init__(*args,
            object_names=object_names,
            scaling_local_list=scaling_local_list,
            **kwargs)
        self._object_position_high = (.82, -.07, -.20)
        self._object_position_low = (.78, -.125, -.20)
        self._success_dist_threshold = success_dist_threshold
        # self._scaling_local_list = scaling_local_list
        # self.set_scaling_dicts()
        self.set_box_pos_as_goal_pos()
        # self.obs_img_dim = 228
        self.box_high = np.array([0.83, .05, -.32])
        self.box_low = np.array([0.77, -.03, -.345])

        # Params used for combine_railrl_pools.py
        self.terminates = False
        self.scripted_traj_len = 30

    def _load_meshes(self):
        super()._load_meshes()
        self._box = bullet.objects.box_open_top()
        # self._test_box = bullet.objects.test_box()

    def get_reward(self, info):
        if self._reward_type in ['dense', 'shaped']:
            reward = -1.0*info['object_goal_dist']
        elif self._reward_type == 'sparse':
            reward = float(info['object_in_box_success'])
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
        ee_pos = self.get_end_effector_pos()

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        object_gripper_success = int(
            object_gripper_dist < self._success_dist_threshold)

        object_goal_dist = np.linalg.norm(object_pos - self._goal_position)
        object_dist_success = int(object_goal_dist < self._success_dist_threshold)

        object_within_box_bounds = ((self.box_low <= object_pos)
            & (object_pos <= self.box_high))
        object_in_box_success = int(np.all(object_within_box_bounds))

        object_xy_in_box_xy = int(np.all(object_within_box_bounds[:2]))
        object_z_above_box_z = int(object_pos[2] >= self.box_low[2])
        object_above_box_sucess = object_xy_in_box_xy and object_z_above_box_z
        info = dict(
            object_goal_dist=object_goal_dist,
            object_dist_success=object_dist_success,
            object_in_box_success=object_in_box_success,
            object_above_box_success=object_above_box_sucess,
            object_gripper_dist=object_gripper_dist,
            object_gripper_success=object_gripper_success)
        return info

class Widow200GraspV5BoxPlaceV0RandObjEnv(Widow200GraspV5BoxPlaceV0Env):
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 *args,
                 in_eval=False,
                 success_dist_threshold=0.04,
                 scaling_local_list=[0.3]*10,
                 **kwargs):
        self.in_eval = in_eval # True when doing evaluation
        # so that we use novel test_objects.

        self.possible_train_objects = [
            'smushed_dumbbell',
            'jar',
            'beer_bottle',
            'mug',
            'square_prism_bin',
            'conic_bin',
            'ball',
            'shed',
            'sack_vase',
            'conic_cup'
        ]

        self.possible_test_objects = [
        ]

        if self.in_eval:
            self.possible_objects = self.possible_test_objects
        else:
            self.possible_objects = self.possible_train_objects

        super().__init__(*args,
            object_names=self.possible_objects,
            success_dist_threshold=success_dist_threshold,
            scaling_local_list=scaling_local_list,
            **kwargs)

    def reset(self):
        """Currently only implemented for selecting 1 object at random"""
        self.object_names = list([np.random.choice(self.possible_objects)])
        print("self.object_names", self.object_names)
        return super().reset()


if __name__ == "__main__":
    import roboverse
    import time

    EPSILON = 0.05
    save_video = True

    env = roboverse.make("Widow200GraspV5BoxPlaceV0RandObj-v0",
                         gui=True,
                         reward_type='sparse',
                         observation_mode='pixels_debug',)

    object_ind = 0
    for i in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        images = [] # new video at the start of each trajectory.

        for _ in range(env.scripted_traj_len):
            if isinstance(obs, dict):
                state_obs = obs[env.fc_input_key]
                obj_obs = obs[env.object_obs_key]

            ee_pos = state_obs[:3]
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.
            object_goal_dist = np.linalg.norm(object_pos - env._goal_position)

            info = env.get_info()
            # theta_action = np.random.uniform()
            # print(object_gripper_dist)
            if (object_gripper_dist > dist_thresh and
                env._gripper_open and not info['object_above_box_success']):
                # print('approaching')
                action = (object_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.02:
                    action[2] = 0.0
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif env._gripper_open and not info['object_above_box_success']:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif not info['object_above_box_success']:
                print(object_goal_dist)
                action = (env._goal_position - object_pos)*7.0
                # action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif not info['object_in_box_success']:
                # object is now above the box.
                action = (env._goal_position - object_pos)*7.0
                action = np.concatenate(
                    (action, np.asarray([0., 0.7, 0.])))
            else:
                action = np.zeros((6,))

            action[:3] += np.random.normal(scale=0.1, size=(3,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print(action)
            obs, rew, done, info = env.step(action)
            print("obs", obs)

            img = env.render_obs()
            if save_video:
                images.append(img)

            time.sleep(0.05)

        print('object pos: {}'.format(object_pos))
        print('reward: {}'.format(rew))
        print('distance: {}'.format(info['object_goal_dist']))
        print('--------------------')

        if save_video:
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)
