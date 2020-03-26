import roboverse
import numpy as np
import skvideo.io
import os

data = np.load('reaching_success_fixed_2020-02-17T11-06-03.npy', allow_pickle=True)
for i in range(data.shape[0]):
	traj = data[i]
	observations = traj["observations"]
	actions = traj["actions"]
	first_observation = observations[0]
	reward_type = "grasp_only"
	env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type, randomize=False)
	#lego_pos = first_observation[-7:-4]
	#env._fixed_object_position = list(lego_pos)
	env.reset()
	filename = 'replay_demo_{}_new.mp4'.format(i)
	writer = skvideo.io.FFmpegWriter(
	    filename,
	    inputdict={"-r": "10"},
	    outputdict={
	        '-vcodec': 'libx264',
	    })
	images = []

	for k in range(len(actions)):
		a = actions[k]
		next_state, reward, done, info = env.step(a)
		image = env.render()
		images.append(image)

	for j in range(len(images)):
	    writer.writeFrame(images[j])
	writer.close()

