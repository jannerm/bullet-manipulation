import argparse
import numpy as np
import roboverse as rv
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='SawyerSoup-v0')
parser.add_argument('--savepath', type=str, default='data/spacemouse-soup/')
parser.add_argument('--gui', type=rv.utils.str2bool, default=True)
parser.add_argument('--render', type=rv.utils.str2bool, default=None)
parser.add_argument('--horizon', type=int, default=1000)
parser.add_argument('--num_episodes', type=int, default=1)
args = parser.parse_args()

rv.utils.make_dir(args.savepath)

env = rv.make(args.env, gui=args.gui, gripper_bounds=[0,1])
spacemouse = rv.devices.SpaceMouse()
pool = rv.utils.DemoPool()
print('Observation space: {} | Action space: {}'.format(env.observation_space, env.action_space))

for ep in range(args.num_episodes):
	obs = env.reset()
	ep_rew = 0
	images = []
	for i in range(args.horizon):
		act = spacemouse.get_action()
		next_obs, rew, term, info = env.step(act)
		pool.add_sample(obs, act, next_obs, rew, term)
		obs = next_obs

		print(act, rew, term)
		ep_rew += rew
		if args.render:
			img = env.render()
			images.append(img)
			rv.utils.save_image('{}/{}.png'.format(args.savepath, i), img)
		
		if term: break
		
	print(ep, i+1, ep_rew)
	if args.render:
		rv.utils.save_video('{}/{}.avi'.format(args.savepath, ep), images)

pool.save(args.savepath, '{}_pool_{}.pkl'.format(rv.utils.timestamp(), pool.size))