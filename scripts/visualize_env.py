import roboverse as rv
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg 
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from PIL import Image
import argparse
import time
images = []
plt.ion()

#change to done 2 seperate
#spacemouse = rv.devices.SpaceMouse(DoF=6)
#env = rv.make('CarrotPlate-v0', gui=False)
#env = rv.make('RemoveLid-v0', gui=True)
env = rv.make('MugDishRack-v0', gui=True)
#env = rv.make('FlipPot-v0', gui=True)


start = time.time()
num_traj = 5
for j in range(num_traj):
	#env.reset()
	env.demo_reset()
	if j > 0: print(returns)
	returns = 0
	for i in range(50):
		#img = Image.fromarray(np.uint8(env.render_obs()))
		#images.append(img)
		#human_action = spacemouse.get_action()
		action, noisy_action = env.get_demo_action()

		# clear_output(wait=True)
		# img = env.render_obs()
		# plt.imshow(img)
		# plt.show()
		# plt.pause(0.01)
		next_observation, reward, done, info = env.step(noisy_action)
		returns += reward
		#print(returns)

print('Simulation Time:', (time.time() - start) / num_traj)

# images[0].save('/Users/sasha/Desktop/rollout.gif',
#                        format='GIF', append_images=images[1:],
#                        save_all=True, duration=100, loop=0)
