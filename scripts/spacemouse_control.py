import roboverse as rv
from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC
from PIL import Image
import numpy as np


spacemouse = rv.devices.SpaceMouse()
env = SawyerLiftEnvGC(
            action_scale=.06,
            action_repeat=10,
            timestep=1./120,
            solver_iterations=500,
            max_force=1000,
            gui=True,
            pos_init=[.75, -.3, 0],
            pos_high=[.75, .4, .3],
            pos_low=[.75, -.4, -.36],
            reset_obj_in_hand_rate=0.0,
            bowl_bounds=[-0.2, 0.2],
            use_rotated_gripper=False,
            use_wide_gripper=True,
            soft_clip=True,
            #obj_urdf='spam',
            max_joint_velocity=None,
            hand_reward=True,
            gripper_reward=True,
            bowl_reward=True,
            goal_sampling_mode='ground',
            random_init_bowl_pos=False,
            bowl_type='fixed',
            num_obj=4,
            visualize=False)
env.reset()
i = 0
while True:
	if i % 50 == 0:
		a = input('save image?')
		if a == 'y':
			img = env.render_obs()
			img = Image.fromarray(np.uint8(img))
			img.save('/Users/sasha/Desktop/images/{0}.jpg'.format(i))
	i += 1


	action = spacemouse.get_action()
	next_obs, rew, term, info = env.step(action)
	#print(rew)
	if term: break
