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
from gym.wrappers import TimeLimit

images = []
plt.ion()

from torchvision.transforms import ColorJitter, RandomResizedCrop

jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
cropper = RandomResizedCrop((96, 128), (0.8, 1.0), (0.8, 1.2))

#change to done 2 seperate
#spacemouse = rv.devices.SpaceMouse(DoF=6)
env = rv.make('RemoveLid-v0', gui=True)
#env = rv.make('RemoveLid-v0', gui=False)
# env = rv.make('MugDishRack-v0', gui=False)
# env = rv.make('FlipPot-v0', gui=True)
env = TimeLimit(env, max_episode_steps=50)


start = time.time()
num_traj = 5
for j in range(num_traj):
	env.reset()
	env.demo_reset()
	done = False
	returns = 0
	while not done:
		img = Image.fromarray(np.uint8(env.render_obs()))
		images.append(np.array(img))
		#human_action = spacemouse.get_action()
		action, noisy_action = env.get_demo_action()

		next_observation, reward, done, info = env.step(noisy_action)
		returns += reward

print('Simulation Time:', (time.time() - start) / num_traj)

path = './rollout.mp4'

video = ImageSequenceClip(images, fps=24)
video.write_videofile(path)
