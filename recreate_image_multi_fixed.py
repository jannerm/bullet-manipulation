import roboverse
import numpy as np
#import skvideo.io
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PIL import Image
import sys
import numpy as np
import pdb 
import pybullet as p
import roboverse
import roboverse.bullet as bullet
import math
from roboverse.bullet.misc import load_urdf, load_obj
import os

from tqdm import tqdm
from PIL import Image
import argparse
import time

fig = plt.figure()
ax = plt.axes(projection='3d')
data = np.load(sys.argv[1], allow_pickle=True)

for i in range(len(data)):
	traj = data[i]
	#actions = traj["actions"] #np.array([traj["observations"][j]["state"] for j in range(len(traj["observations"]))])
	#ax.plot3D(observations[:50, 0], observations[:50, 1], observations[:50, 2], label="{}".format(i))
	actions = traj["actions"]
	observations = traj["observations"]

	for index in range(len(observations)):
		data[i]["observations"][index] = {"state": data[i]["observations"][index], "image": None}

	first_observation = traj["observations"][0]["state"]

	reward_type = "shaped"
	env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type, 
						 randomize=False, observation_mode='pixels_debug')
	#lego_pos = first_observation[-7:-4]
	#env._fixed_object_position = lego_pos
	env.reset()
	print(env._trimodal_positions)

	images = []
	
	image = env.render()
	data[i]["observations"][0]["image"] = image

	for index in range(len(actions)):
		image = env.render_obs()
		data[i]["observations"][index + 1]["image"] = image
		a = actions[index]
		env.step(a)
		images.append(Image.fromarray(np.uint8(image)))
		
	
	#video_save_path = os.path.join(".", "videos")
	images[0].save('{}/{}.gif'.format("videos_multi", i),
				format='GIF', append_images=images[1:],
				save_all=True, duration=100, loop=0)
	
plt.legend()
plt.show()

path = os.path.join(__file__, "..", "{}_multi_fixed".format(sys.argv[1][:-4]))
np.save(path, data)