import roboverse
import time

env = roboverse.make('WidowX200Grasp-v0', gui=True)
time.sleep(100000)
'''
env.reset()
for i in range(100000):
    env.step([0, ((i//10)%2) * 2 - 1, 0, (i//10)%2])
time.sleep(200000)
'''
