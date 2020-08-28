from roboverse.envs.widow200_grasp_v6 import Widow200GraspV6Env
import roboverse
import numpy as np

class Widow250GraspV6Env(Widow200GraspV6Env):

    def __init__(self,
                 *args,
                 object_names=('beer_bottle',),
                 scaling_local_list=[0.5],
                 **kwargs):
        self.object_names = object_names
        kwargs['env_name'] = "Widow250GraspEnv"
        super().__init__(*args,
            object_names=self.object_names,
            scaling_local_list=scaling_local_list,
            **kwargs)
        self._env_name = kwargs['env_name']

if __name__ == "__main__":
    EPSILON = 0.05
    save_video = True

    env = roboverse.make("Widow250GraspV6-v0",
                         gui=True,
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
            object_goal_dist = 0.1 # dummy value

            info = env.get_info()
            # theta_action = np.random.uniform()
            # print(object_gripper_dist)
            if (object_gripper_dist > dist_thresh) and env._gripper_open:
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
                action = np.zeros((6,))

            action[:3] += np.random.normal(scale=0.1, size=(3,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print(action)
            obs, rew, done, info = env.step(action)
            print("rew", rew)

            img = env.render_obs()
            if save_video:
                images.append(img)

        print('object pos: {}'.format(object_pos))
        print('reward: {}'.format(rew))
        print('--------------------')
