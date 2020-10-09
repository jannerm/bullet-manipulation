import gym
import numpy as np
import ipdb


def change_obs(obs):
    new_obs = obs.reshape(-1,48,48,3).T
    new_obs = np.rot90(new_obs,3,axes=(-3,-2))
    new_obs = np.rollaxis(new_obs,3)
    new_obs = new_obs.reshape(new_obs.shape[0], -1)
    return new_obs

class ReshapeObsEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space

    def step(self, act, *args, **kwargs):
        obs, rew, term, info =  self.env.step(act, *args, **kwargs)
        obs['image'] = change_obs(obs['image'])
        return obs, rew, term, info

    def step_slow(self, act, *args, **kwargs):
        obs, rew, term, info = self.env.step(act, *args, **kwargs)
        obs['image'] = change_obs(obs['image'])
        return change_obs(obs), rew, term, info, imgs

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.env, attr)