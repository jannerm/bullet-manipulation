from roboverse.envs.widow200_grasp_v6_box_place_v0 import (
    Widow200GraspV6BoxPlaceV0Env)
from roboverse.envs.rand_obj import RandObjEnv
import roboverse.bullet as bullet
import roboverse.utils as utils
import numpy as np
import time

class Widow200GraspV6DrawerPlaceV0Env(Widow200GraspV6BoxPlaceV0Env):
    """Task is to grasp object from open drawer and place on top of the drawer."""
    def __init__(self,
                 *args,
                 object_names=('aero_cylinder',),
                 scaling_local_list=[0.2],
                 success_dist_threshold=0.04,
                 **kwargs):
        super().__init__(*args,
            object_names=object_names,
            scaling_local_list=scaling_local_list,
            **kwargs)
        self._env_name = "Widow200GraspV6DrawerPlaceV0Env"
        self._object_position_high = (.82, -.07, -.2)
        self._object_position_low = (.78, -.125, -.2)
        self._success_dist_threshold = success_dist_threshold
        # self._scaling_local_list = scaling_local_list
        # self.set_scaling_dicts()
        self.set_box_pos_as_goal_pos()
        # self.obs_img_dim = 228
        self.box_high = np.array([0.8825, .05, -.21])
        self.box_low = np.array([0.7775, -0.03, -.255])
        
        self.scripted_traj_len = 30

    def _load_meshes(self):
        super()._load_meshes()
        self._box = bullet.objects.lifted_long_box_open_top()

    def reset(self):
        obs = super().reset()
        return obs

if __name__ == "__main__":
    import roboverse
    import time

    EPSILON = 0.05
    save_video = True

    env = roboverse.make("Widow200GraspV6DrawerPlaceV0-v0",
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
                    action[2] = 0.1
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
            print("rew", rew)

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
