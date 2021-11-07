import roboverse as rv
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
import time
from gym.wrappers import TimeLimit
import os
import shutil


#change to done 2 seperate
#spacemouse = rv.devices.SpaceMouse(DoF=6)
env = rv.make('RemoveLid-v0', gui=False)
#env = rv.make('RemoveLid-v0', gui=False)
# env = rv.make('MugDishRack-v0', gui=False)
# env = rv.make('FlipPot-v0', gui=True)
env = TimeLimit(env, max_episode_steps=50)

lengths = []

data_folder = 'dataset'
if os.path.exists(data_folder):
	shutil.rmtree(data_folder)

start = time.time()
num_traj = 5
for j in range(num_traj):
	traj_folder = os.path.join(data_folder, str(j))
	os.makedirs(traj_folder, exist_ok=True)
	env.reset()
	env.demo_reset()
	done = False
	returns = 0
	i = 0
	actions = []
	rewards = []
	while not done:
		img = Image.fromarray(np.uint8(env.render_obs()))
		img.save(os.path.join(traj_folder, f'{i}.png'))
		# human_action = spacemouse.get_action()
		action, noisy_action = env.get_demo_action()
		# noisy_action = env.action_space.sample()

		next_observation, reward, done, info = env.step(noisy_action)

		actions.append(noisy_action)
		rewards.append(reward)

		returns += reward
		i += 1
	
	img = Image.fromarray(np.uint8(env.render_obs()))
	img.save(os.path.join(traj_folder, f'{i}.png'))

	actions = np.stack(actions, 0)
	rewards = np.stack(rewards, 0)
	
	with open(os.path.join(traj_folder, 'data.npz'), 'wb') as f:
		np.savez(f, actions=actions, rewards=rewards)

	lengths.append(len(actions))

	print(f"Episode {j} returns {returns}")


with open(os.path.join(data_folder, 'lengths.np'), 'wb') as f:
	np.save(f, lengths)

print('Simulation Time:', (time.time() - start) / num_traj)

path = './rollout.mp4'

# video = ImageSequenceClip(images, fps=24)
# video.write_videofile(path)
