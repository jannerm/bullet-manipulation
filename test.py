import numpy as np
import roboverse as rv
import pdb


render = False
env = rv.make('SawyerLiftGC-v0', goal_mult=4, action_scale=.1, action_repeat=10,
              timestep=1./120, gui=render)

obs = env.reset()
while True:
    action = env.action_space.sample()
    import pdb; pdb.set_trace()
    obs, reward, done, info = env.step(action)
