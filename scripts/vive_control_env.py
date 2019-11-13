import numpy as np
import pdb 
import pybullet as p
import roboverse
import roboverse.bullet as bullet
import math
from roboverse.bullet.misc import load_urdf, load_obj, load_random_objects
import os
# connect to the VR server (by p.SHARED_MEMORY)
bullet.connect()
bullet.setup()

# query event which is a tuple of changes in controller. Index corresponds to POSITION, ORIENTATION, BUTTTON etc
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6

# set the environment
env = roboverse.make('SawyerGraspOne-v0', render=True)

controllers = [e[0] for e in p.getVREvents()]

trigger = 0

num_grasps = 0
save_video = True
curr_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(curr_dir)
pklPath = home_dir + '/data'
trajectories = []
image_data = []
num_of_sample = 2

for i in range(num_of_sample):
	env.reset()
	target_pos = env.get_object_midpoint('duck')
	target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
	images = []
	trajectory = []

	while True:

		ee_pos = env.get_end_effector_pos()
		grasping_data = []
		grasping_data.append(env.get_observation())

		events = p.getVREvents()

		if events:
			e = events[0]
		else:
			continue

		# Detect change in button, and change trigger state
		if e[BUTTONS][33] & p.VR_BUTTON_WAS_TRIGGERED:
			trigger = 1
		if e[BUTTONS][33] & p.VR_BUTTON_WAS_RELEASED:
			trigger = 0

		if e[0] != controllers[0]:
			break

		# pass controller position and orientation into the environment
		cont_pos = e[POSITION]
		cont_orient = e[ORIENTATION]

		action = [cont_pos[0] - ee_pos[0], cont_pos[1] - ee_pos[1], cont_pos[2] - ee_pos[2]]
		grip = trigger

		action = np.append(action, [grip])
		action = np.append(action, list(cont_orient))
		#action = np.append(action, [grip])
		img = env.render()
		images.append(np.uint8(img))

		next_state, reward, done, info = env.step(action)
		grasping_data.append(next_state)
		grasping_data.append(action)
		grasping_data.append(reward)
		grasping_data.append(done)
		trajectory.append(grasping_data)

		object_pos = env.get_object_midpoint('duck')

		if object_pos[2] > -0.1:
			num_grasps += 1
			break

	print(trajectory)
	trajectories.append(trajectory)


#print('Num attempts: {}'.format(j))
print('Num grasps: {}'.format(num_grasps))

