import gym
import numpy as np
import ipdb
import roboverse.bullet as bullet
from roboverse.envs.widow200_grasp_v6 import Widow200GraspV6Env

class PushingWrapperEnv(gym.Env):
    def __init__(self, env, goal = None, eps = 0.03):
        self.env = env
        self.action_space = self.env.action_space
        if not goal:
            tray_info = bullet.get_body_info(env._tray, quat_to_deg=False)
            goal = np.asarray(tray_info['pos'])
        self.goal = goal
        self.eps = eps

    def step(self, act, *args, **kwargs):
        obs, _, term, info =  self.env.step(act, *args, **kwargs)
        return obs, self.get_reward(obs), term, info

    def step_slow(self, act, *args, **kwargs):
        obs, _, term, info, imgs =  self.env.step_slow(act, *args, **kwargs)
        return obs, self.get_reward(obs), term, info, imgs

    def get_reward(self, obs):
        object_grasp = self.env._objects.keys()
        object_grasp = list(object_grasp)[0]
        object_info = bullet.get_body_info(self._objects[object_grasp],quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])
        
        #calculated in case want to include current state as reward
        state = obs['state'] if 'state' in obs else obs['robot_state']

        dist = np.linalg.norm((object_pos[:3]-self.goal[:3])) 
        return -dist if dist > self.eps else 5
    
    def set_goal(self, goal):
        self.goal = goal

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

if __name__ == "__main__":
    env1 = Widow200GraspV6Env()
    push_env = PushingWrapperEnv(env1)
    obs = dict(state=push_env.reset())
    rew = push_env.get_reward(obs)
    import ipdb; ipdb.set_trace()