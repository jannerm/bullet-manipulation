import roboverse
import numpy as np
#import skvideo.io
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PIL import Image

fig = plt.figure()
ax = plt.axes(projection='3d')
data = np.load('vr_demos_success_2020-04-27T13-26-29.npy', allow_pickle=True)
for i in range(100):
	traj = data[i]
	observations = np.array(traj["observations"])
	#ax.plot3D(observations[:50, 0], observations[:50, 1], observations[:50, 2], 'gray')
		
	
	actions = traj["actions"]
	first_observation = observations[0]
	reward_type = "shaped"
	env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type, randomize=False)
	#lego_pos = first_observation[-7:-4]
	env._trimodal_positions = first_observation["all_obj_pos"]
	env.reset()
	print(env.get_observation())
	


	"""
	filename = 'replay_demo_{}.mp4'.format(i)
	writer = skvideo.io.FFmpegWriter(
		filename,
		inputdict={"-r": "10"},
		outputdict={
			'-vcodec': 'libx264',
		})
	"""

	images = []
	
	for a in actions:
		env.step(a)
		image = env.render()
		images.append(Image.fromarray(np.uint8(image)))

	#video_save_path = os.path.join(".", "videos")
	images[0].save('{}/{}.gif'.format("videos", i),
				format='GIF', append_images=images[1:],
				save_all=True, duration=100, loop=0)
	

plt.show()