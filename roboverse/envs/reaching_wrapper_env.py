import gym
import ipdb

class ReachingWrapperEnv(gym.Env):
    def __init__(self, env, goal):
        self.env = env
        self.goal = goal

    def step(self, act, *args, **kwargs):
        obs, _, term, info =  self.env.step(act, *args, **kwargs)
        return obs, self.get_reward(obs), term, info

    def get_reward(self, obs):
        ipdb.set_trace()
        return -np.linalg.norm((obs[:3]-self.goal[:3]))
    
    def set_goal(self, goal):
        self.goal = goal

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)