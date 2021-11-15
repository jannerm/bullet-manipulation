import gym
import numpy as np


class Roboverse(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.shape = (48, 48, 3)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=self.shape,
                                                dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation['image'] = observation['image'].reshape(3, 48,
                                                            48).transpose(
                                                                (1, 2, 0))
        return (observation['image'] * 255).astype(np.uint8)
