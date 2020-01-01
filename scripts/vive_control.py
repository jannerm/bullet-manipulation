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
curr_dir = os.path.dirname(os.path.abspath(__file__))
object_path = os.path.join(curr_dir, '../roboverse/envs/assets/ShapeNetCore')
load_random_objects(object_path, 4)

# directly control sawyer robot for now; will create wrapper in env later
sawyer = env._sawyer

duck = bullet.load_urdf('duck_vhacd.urdf', [.75, .2, 4], [0, 0, 1, 0], scale=0.8)

end_effector = bullet.get_index_by_attribute(sawyer, 'link_name', 'right_l6')
pos = np.array([0.5, 0, 0])
theta = [0.7071,0.7071,0,0]
bullet.position_control(sawyer, end_effector, pos, theta)

controllers = [e[0] for e in p.getVREvents()]

trigger = 0

while True:

	# Get new state from controller
	events = p.getVREvents()

	for e in (events):

		# Detect change in button, and change trigger state
		if e[BUTTONS][33] & p.VR_BUTTON_WAS_TRIGGERED:
			trigger = 1
		if e[BUTTONS][33] & p.VR_BUTTON_WAS_RELEASED:
			trigger = 0

		if e[0] != controllers[0]:
			break

		# pass controller position and orientation into the environment
		cont_pos = e[POSITION]


		cont_orient = e[ORIENTATION]#p.getQuaternionFromEuler([0, -math.pi, 0]) #e[ORIENTATION]#p.getQuaternionFromEuler([0, -math.pi, 0])#e[ORIENTATION]
		#p.getQuaternionFromEuler([0, -math.pi, 0]) #e[ORIENTATION]#
		#p.getQuaternionFromEuler([0, -math.pi, 0]) #e[ORIENTATION]#
		#e[ORIENTATION]

		# use sawyer_ik function to update the changes in simulation
		bullet.sawyer_ik(sawyer, end_effector, cont_pos, cont_orient, trigger)
		bullet.step()
		pos = bullet.get_link_state(sawyer, end_effector, 'pos')

