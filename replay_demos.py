import roboverse
import numpy as np
import skvideo.io
import os
data = np.load('vr_demos_success.npy', allow_pickle=True)
for i in [2, 45, 6, 90, 114]:
	traj = data[i]
	observations = traj["observations"]
	actions = traj["actions"]
	first_observation = observations[0]
	reward_type = "shaped"
	env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type, randomize=False)
	lego_pos = first_observation[-7:-4]
	env._fixed_object_position = list(lego_pos)
	env.reset()
	print(env.get_observation())
	filename = 'replay_demo_{}.mp4'.format(i)
	writer = skvideo.io.FFmpegWriter(
	    filename,
	    inputdict={"-r": "10"},
	    outputdict={
	        '-vcodec': 'libx264',
	    })
	images = []
	for a in actions:
		env.step(a)
		image = env.render()
		images.append(image)
	for i in range(len(images)):
	    writer.writeFrame(images[i])
	writer.close()