import time
import roboverse
import numpy as np
#import skvideo.io
import os

#%matplotlib inline
data = np.load('vr_demos_success + 2020-01-10T14-38-19.npy', allow_pickle=True)
for i in range(5):
	traj = data[i]
	observations = traj["observations"]
	actions = traj["actions"]

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
	#filename = 'replay_demo_{}_new.mp4'.format(i)
	

	for k in range(len(actions)):
		a = actions[k]
		next_state, reward, done, info = env.step(a)
		time.sleep(0.02)
		print("correct object location: ", observations[k][-7:-4])
		print("actual object location: ", env.get_observation()[-7:-4])
		#print("correct full observation: ", observations[k + 2])
		#print("actual full observation: ", env.get_observation())


	print("############\n\n")