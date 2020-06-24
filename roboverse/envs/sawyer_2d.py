import copy
import numpy as np
import gym
import pdb

import roboverse as rv
from roboverse import bullet

class Sawyer2dEnv(gym.Env):

    def __init__(self, env='SawyerSoup-v0',
                       pos_init=[.75, 0, 0],
                       pos_high=[.75,.4,.25],
                       pos_low=[.75,-.6,-.36],
                       soft_clip=False,
                       **kwargs):

        self._soft_clip = soft_clip
        self._env = gym.make(env, pos_init=pos_init, pos_high=pos_high, pos_low=pos_low, **kwargs)
        self._objects = self._env._objects
        self._init_states = self._get_body_states()
        self._set_spaces()

    def _set_spaces(self):
        self.action_space = self._env.action_space
        obs_low = self._env.observation_space.low[0]
        obs_high = self._env.observation_space.high[0]
        obs_type = type(self._env.observation_space)
        obs_shape = self.get_observation().shape
        self.observation_space = obs_type(low=obs_low, high=obs_high, shape=obs_shape)

    ## @TODO : rewrite this so that it uses self._state_query
    def get_observation(self):
        ee_pos = bullet.get_link_state(self._env._sawyer, self._env._end_effector, 'pos')
        # bodies = sorted([v for k, v in self._objects.items() if not bullet.has_fixed_root(v)])
        # obj_pos = [bullet.get_body_info(body, 'pos') for body in bodies]
        obj_pos = self._env.get_object_positions()
        ee_pos = np.array(ee_pos[1:])
        obj_pos = np.array([pos[1:] for pos in obj_pos])
        observation = np.concatenate((ee_pos, obj_pos.flatten()))
        return observation

    def reset(self, *args, **kwargs):
        self._env.reset(*args, **kwargs)
        return self.get_observation()

    def load_state(self, *args, **kwargs):
        self._env.load_state(*args, **kwargs)
        return self.get_observation()

    def step(self, act, *args, **kwargs):
        act[0] = 0
        out = self._env.step(act, *args, **kwargs)

        # for name, body in self._objects.items():
        #     if 'bowl' in name:
        #         continue
        #     link = bullet.get_index_by_attribute(body, 'link_name', 'spam')
        #     self._format_state_query()
        #     state = bullet.get_link_state(body, link, ['pos', 'theta'])
        #     theta = state['theta']
        #     pos = state['pos']
        #     print(pos)
        #     bullet.set_body_state(body, pos, theta)
        # print()

        if self._env._clip_obj_pos:
            current_states = self._get_body_states()
            for name, body in self._objects.items():
                current = current_states[name]

                if self._soft_clip:
                    x = current['pos'][0:1]
                    center_x = self._env._pos_init[0:1]
                    if np.abs(x[0] - center_x[0]) > 0.015:
                        x = center_x
                else:
                    x = self._env._pos_init[0:1] #init['pos'][0:1]

                yz = current['pos'][1:3]
                theta = current['theta']
                pos = list(x) + list(yz)
                bullet.set_body_state(body, pos, theta)
        obs = self.get_observation()
        rew = self._env.get_reward(obs)
        term = self._env.get_termination(obs)
        info = {}
        return obs, rew, term, info

    def _get_body_states(self):
        states = {}
        for name, body in self._objects.items():
            state = bullet.get_body_info(body, ['pos', 'theta'])
            states[name] = state
        return states

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._env, attr)

