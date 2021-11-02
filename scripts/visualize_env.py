import roboverse as rv
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg
import matplotlib.pyplot as plt
import os
from PIL import Image
from moviepy.editor import *
import argparse
import time
images = []
plt.ion()

from torchvision.transforms import ColorJitter, RandomResizedCrop

jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
cropper = RandomResizedCrop((96, 128), (0.8, 1.0), (0.8, 1.2))

#change to done 2 seperate
#spacemouse = rv.devices.SpaceMouse(DoF=6)
env = rv.make('RemoveLid-v0', gui=True)
env.reset()
#env = rv.make('RemoveLid-v0', gui=False)
#env = rv.make('MugDishRack-v0', gui=False)
#env = rv.make('FlipPot-v0', gui=True)


start = time.time()
num_traj = 1
for j in range(num_traj):
	env.demo_reset()
	if j > 0: print(returns)
	returns = 0
	for i in range(50):
		img = Image.fromarray(np.uint8(env.render_obs()))
		images.append(np.array(img))
		#human_action = spacemouse.get_action()
		action, noisy_action = env.get_demo_action()

		# clear_output(wait=True)
		# img = env.render_obs()
		# plt.imshow(img)
		# plt.show()
		# plt.pause(0.01)
		next_observation, reward, done, info = env.step(noisy_action)
		returns += reward
		print(info)

print('Simulation Time:', (time.time() - start) / num_traj)

path = '/Users/sasha/Desktop/rollout.mp4'
#path = '/iris/u/khazatsky/bridge_codebase/data/visualizations/rollout.gif'

video = ImageSequenceClip(images, fps=24)
video.write_videofile(path)
