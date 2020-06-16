from roboverse.envs.widow200_grasp_v6 import Widow200GraspV6Env
import roboverse.bullet as bullet
import roboverse.utils as utils
import numpy as np


class Widow200GraspV6BoxV0Env(Widow200GraspV6Env):
    """
    Deterministic object, non-terminating grasping env with
    a box in the tray. Like GraspV6, the goal is still only
    to grasp the object and lift it above a certain height.
    """

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
        self.box_high = np.array([0.83, .05, -.32])
        self.box_low = np.array([0.77, -.03, -.345])

    def _load_meshes(self):
        super()._load_meshes()
        self._box = bullet.objects.box_open_top()

class Widow200GraspV6BoxV0RandObjEnv(Widow200GraspV6BoxV0Env):
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
            self.possible_objects = self.possible_train_objects
        else:
            self.possible_objects = self.possible_test_objects

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

    env = roboverse.make("Widow200GraspV6BoxV0RandObj-v0",
                         gui=True,
                         reward_type='sparse',
                         observation_mode='state',)

    EPSILON = 0.05
    object_ind = 0
    margin = 0.025

    def strint(x):
        return str(int(x))

    for _ in range(50):
        obs = env.reset()

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

            # print('reward: {}'.format(rew))
            rewards.append(rew)

        # print("="*10)
        print("".join(list(map(strint, rewards))))
