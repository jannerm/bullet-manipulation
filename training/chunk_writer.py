import os

import numpy as np


class ChunkWriter(object):
    def __init__(self, datadir: str = 'dataset', max_size: int = 1000):
        self.max_size = max_size

        self.observations = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.dones = []
        self.dummys = []

        self.datadir = datadir
        self.chunk_id = 0

    def add_transiton(self,
                      obs: np.ndarray,
                      action: np.ndarray,
                      reward: float,
                      mask: float,
                      done: bool,
                      dummy: bool = False):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.dones.append(done)
        self.dummys.append(dummy)

        if len(self.observations) == self.max_size:
            observations = np.stack(self.observations)
            actions = np.stack(self.actions)
            rewards = np.stack(self.rewards)
            masks = np.stack(self.masks)
            dones = np.stack(self.dones)
            dummys = np.stack(self.dummys)

            chunk_filename = os.path.join(self.datadir, f'{self.chunk_id}.npz')
            with open(chunk_filename, 'wb') as f:
                np.savez_compressed(f,
                                    observations=observations,
                                    actions=actions,
                                    rewards=rewards,
                                    masks=masks,
                                    dones=dones,
                                    dummys=dummys)

            self.observations.clear()
            self.actions.clear()
            self.rewards.clear()
            self.masks.clear()
            self.dones.clear()
            self.dummys.clear()

            self.chunk_id += 1
