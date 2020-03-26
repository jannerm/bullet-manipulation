import roboverse
import time

env = roboverse.make('WidowGraspJointDownwards-v0', gui=True)
env.reset()
for i in range(100000):
    env.step([0.1, 0, 0, 0.1, 0, 0, 1])
   
