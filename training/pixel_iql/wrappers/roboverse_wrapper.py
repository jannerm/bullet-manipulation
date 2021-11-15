import gym
import numpy as np


class Roboverse(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.shape = (96, 128, 3)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=self.shape,
                                                dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self.unwrapped.render_obs()


class SuccessWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['success'] = (reward == 0)
        return obs, reward, done, info
