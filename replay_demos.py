print("here")
import roboverse
import numpy as np
import skvideo.io
import os

print("reach here")
#%matplotlib inline
data = np.load('vr_demos_success_2020-01-09T21-06-36.npy', allow_pickle=True)
for i in range(5):
	traj = data[i]
	observations = traj["observations"]
	actions = traj["actions"]
	#for a in actions:/home/huihanl/Downloads/vr_demos_success_2020-01-09T21-06-36.npy
	#	a[3] = a[3] / 3
	#print(actions)
	first_observation = observations[0]
	print("first_observation is : ", first_observation)
	reward_type = "grasp_only"
	env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type, randomize=False)
	lego_pos = first_observation[-7:-4]
	#print(lego_pos)
	env._fixed_object_position = list(lego_pos)
	env.reset()
	print("observation from the env is : ", env.get_observation())
	#print(env.get_observation())
	filename = 'replay_demo_{}_new.mp4'.format(i)
	writer = skvideo.io.FFmpegWriter(
	    filename,
	    inputdict={"-r": "10"},
	    outputdict={
	        '-vcodec': 'libx264',
	    })
	images = []
	print("reach here ")
	obs_0 = []
	obs_1 = []
	obs_2 = []
	#obs_4 = []
	#obs_5 = []
	#obs_6 = []

	obs_0_r = []
	obs_1_r = []
	obs_2_r = []
	#obs_4_r = []
	#obs_5_r = []
	#obs_6_r = []

	for k in range(len(actions)):
		a = actions[k]
		
		next_state, reward, done, info = env.step(a)
		print("true next state: ", traj["next_observations"][k])
		print("actual next state: ", next_state)
		#obs_0.append(traj["next_observations"][k][0])
		#obs_1.append(traj["next_observations"][k][1])
		#obs_2.append(traj["next_observations"][k][2])
		#obs_4.append(traj["next_observations"][k][4])
		#obs_5.append(traj["next_observations"][k][5])
		#obs_6.append(traj["next_observations"][k][6])

		obs_0_r.append(next_state[0])
		obs_1_r.append(next_state[1])
		obs_2_r.append(next_state[2])
		#obs_4_r.append(next_state[4])
		#obs_5_r.append(next_state[5])
		#obs_6_r.append(next_state[6])				

		#print("correct object location: ", observations[k][-7:-4])
		#print("actual object location: ", env.get_observation()[-7:-4])
		image = env.render()
		images.append(image)
		#print("correct full observation: ", observations[k + 2])
		#print("actual full observation: ", env.get_observation())
		#print("\n")
	#ax = plt.axes(projection='3d')
	#ax.scatter3D(obs_0, obs_1, obs_2)
	#ax.scatter3D(obs_0_r, obs_1_r, obs_2_r)

	for j in range(len(images)):
	    writer.writeFrame(images[j])
	writer.close()

	print("############\n\n")