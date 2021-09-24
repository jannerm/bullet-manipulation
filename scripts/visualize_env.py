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
images = []
plt.ion()


spacemouse = rv.devices.SpaceMouse(DoF=6)

#env = rv.make('WidowBase-v0', gui=True)
env = rv.make('BridgeKitchen-v0', DoF=6, use_bounding_box=False, gui=True)
#env = rv.make('SawyerRigAffordances-v0', DoF=4, use_bounding_box=False, gui=True)

for j in range(5):
	env.reset()

	for i in range(1000):
		img = Image.fromarray(np.uint8(env.render_obs()))
		images.append(img)
		action = spacemouse.get_action()

		# clear_output(wait=True)
		# img = env.render_obs()
		# plt.imshow(img)
		# plt.show()
		# plt.pause(0.01)
		#action = env.get_demo_action()
		next_observation, reward, done, info = env.step(action)