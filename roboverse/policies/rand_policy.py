import numpy as np
import ipdb

class RandPolicy:
    def __init__(self):
        super().__init__()
    
    def get_action(self, obs):
        std = [0.035, 0.035, 0.05, np.pi / 8]
        mean = np.zeros_like(std)
        action = np.random.normal(mean, std)
        ipdb.set_trace()
        return action

if __name__ == "__main__":
    rp = RandPolicy()
    ipdb.set_trace()
    rp.get_action(None)