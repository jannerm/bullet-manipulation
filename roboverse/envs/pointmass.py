import numpy as np
import gym

import roboverse.bullet as bullet
from roboverse.envs.serializable import Serializable


class PointmassBaseEnv(gym.Env, Serializable):

    def __init__(self,
                 img_dim=48,
                 gui=False,
                 action_scale=0.1,
                 timestep=1. / 120,
                 solver_iterations=150,
                 init_pos=(0.75, -0.4, -0.22),
                 goal_pos=(1.0, 0.2, -0.22),
                 observation_mode='state',  # or pixels
                 transpose_image=False,
                 ):

        self._img_dim = img_dim
        self.image_shape = (img_dim, img_dim)
        self._gui = gui
        self._action_scale = action_scale
        self._timestep = timestep
        self._solver_iterations = solver_iterations
        self._init_pos = np.asarray(init_pos)
        self._goal_pos = np.asarray(goal_pos)
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image

        bullet.connect_headless(self._gui)

        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[0.8, -.2, -0.3], distance=1.2,
            yaw=90, pitch=-45, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self._img_dim, self._img_dim)

        self.image_length = img_dim*img_dim*3
        self.xy_max = [1.0, 0.2]
        self.xy_min = [0.75, -0.4]

        self._set_spaces()

    def _set_action_space(self):
        act_dim = 2
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_spaces(self):
        self._set_action_space()
        # obs = self.reset()
        if self._observation_mode == 'state':
            observation_dim = 3
            obs_bound = 100
            obs_high = np.ones(observation_dim) * obs_bound
            self.observation_space = gym.spaces.Box(-obs_high, obs_high)
        elif self._observation_mode == 'pixels':
            img_space = gym.spaces.Box(0, 1, (self.image_length,), dtype=np.float32)
            spaces = {'image': img_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def _load_meshes(self):
        self._plane = bullet.objects.table(scale=1.0)
        self._agent = bullet.objects.cube(pos=self._init_pos, scale=0.04)

    def reset(self):
        bullet.reset()
        self._load_meshes()
        bullet.set_pointmass_control(self._agent)
        return self.get_observation()

    def compute_reward(self, info):
        return -1.0*info['distance']

    def get_info(self):
        object_info = bullet.get_body_info(self._agent, quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])
        info = dict(
            distance=np.linalg.norm(object_pos - self._goal_pos),
            object_pos=object_pos
        )
        return info

    def step(self, action):

        info = self.get_info()
        if info['object_pos'][0] >= self.xy_max[0]:
            action[0] = min(0, action[0])
        if info['object_pos'][1] >= self.xy_max[1]:
            action[1] = min(0, action[1])
        if info['object_pos'][0] <= self.xy_min[0]:
            action[0] = max(0, action[0])
        if info['object_pos'][1] <= self.xy_min[1]:
            action[1] = max(0, action[1])

        bullet.pointmass_position_step_simulation(self._agent, action)
        reward = self.compute_reward(info)
        obs = self.get_observation()
        done = False
        return obs, reward, done, info

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self._img_dim, self._img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def get_observation(self):
        if self._observation_mode == 'state':
            object_info = bullet.get_body_info(self._agent, quat_to_deg=False)
            object_pos = object_info['pos']
            return np.asarray(object_pos)
        elif self._observation_mode == 'pixels':
            img = self.render_obs()
            img = np.float32(img.flatten()) / 255.0
            return {'image': img}
        else:
            raise NotImplementedError