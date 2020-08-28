import gym
import numpy as np
import ipdb

class NormWrapperEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.p_low = np.array(self.env._pos_low )
        self.p_hi = np.array(self.env._pos_high)
        print(self.p_low, self.p_hi)
        self.observation_space = self.env.observation_space

    def invert_act(self, act):
        # print('action', act, np.concatenate((act[:3]*(self.p_hi-self.p_low), act[3:])))
        return np.concatenate((act[:3]*(self.p_hi-self.p_low), act[3:]))
    
    def transform_actions(self, act):
        return np.concatenate((act[:,:3]*(self.p_hi-self.p_low), act[:,3:]), 1)

    def transform_xyz(self, xyz):
        # print('obs', xyz, (xyz-self.p_low)/(self.p_hi-self.p_low))
        return (xyz-self.p_low)/(self.p_hi-self.p_low)

    def step(self, act, *args, **kwargs):
        act = self.invert_act(act)
        obs, rew, term, info =  self.env.step(act, *args, **kwargs)
        state = obs['robot_state'] if 'robot_state' in obs else obs['state']
        xyz, rest = state[:3], state[3:]
        xyz = self.transform_xyz(xyz)
        if 'state' in obs:
            obs['state'] = np.concatenate((xyz,rest))
        else:
            obs['robot_state'] = np.concatenate((xyz,rest))
        return obs, rew, term, info

    def step_slow(self, act, *args, **kwargs):
        act = self.invert_act(act)
        obs, rew, term, info, imgs =  self.env.step_slow(act, *args, **kwargs)
        state = obs['robot_state'] if 'robot_state' in obs else obs['state']
        xyz, rest = state[:3], state[3:]
        xyz = self.transform_xyz(xyz)
        if 'state' in obs:
            obs['state'] = np.concatenate((xyz,rest))
        else:
            obs['robot_state'] = np.concatenate((xyz,rest))
        return obs, rew, term, info, imgs

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def __getattr__(self, attr):
        if attr == 'env': return None
        return getattr(self.env, attr)