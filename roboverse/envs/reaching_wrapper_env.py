import gym
import numpy as np
import ipdb

class ReachingWrapperEnv(gym.Env):
    def __init__(self, env, goal, eps = 0.1, red=False):
        self.env = env
        self.action_space = self.env.action_space
        self.action_dim_curr = int(np.prod(self.action_space.shape))
        self.goal = goal
        self.eps = eps
        self.red = red

    def step(self, act, *args, **kwargs):
        if self.red and act.shape[-1] != self.action_dim_curr:
            act = np.concatenate((act, np.zeros(self.action_dim_curr-3)))
        obs, _, term, info =  self.env.step(act, *args, **kwargs)
        return obs, self.get_reward(obs), term, info

    def step_slow(self, act, *args, **kwargs):
        if self.red: 
            act = np.concatenate((act, np.zeros(self.action_dim_curr-3)))
        obs, _, term, info, imgs =  self.env.step_slow(act, *args, **kwargs)
        return obs, self.get_reward(obs), term, info, imgs

    def get_reward(self, obs):
        state = obs['state'] if 'state' in obs else obs['robot_state']
        dist = np.linalg.norm((state[:3]-self.goal[:3])) 
        return -dist if dist > self.eps else 5
    
    def set_goal(self, goal):
        self.goal = goal

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.env, attr)