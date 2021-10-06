import roboverse as rv
import numpy as np
from tqdm import tqdm

env = rv.make('RemoveLid-v0')
num_traj = 300
timesteps = 50
imlength = 96 * 128 * 3
a_dim = 7
data_save_path = '/Users/sasha/Desktop/lid_demos.npy'
all_returns = 0


dataset = {
    'observations': np.zeros((num_traj, timesteps, imlength), dtype=np.uint8),
    'actions': np.zeros((num_traj, timesteps, a_dim)),
}

def collect_demo(timesteps):
	env.demo_reset()
	returns = 0
	traj = {
	    'observations': np.zeros((timesteps, imlength), dtype=np.uint8),
	    'actions': np.zeros((timesteps, a_dim)),
	}
	for i in range(timesteps):
		# DOING A BC TRICK!
		traj['observations'][i] = np.uint8(env.render_obs().transpose()).flatten()
		action, noisy_action = env.get_demo_action()
		traj['actions'][i] = action
		next_observation, reward, done, info = env.step(noisy_action)
		returns += reward
	return traj, returns



for j in tqdm(range(num_traj)):
	while True:
		traj, returns = collect_demo(timesteps)
		success = env.get_info()['task_achieved']
		if success:
			dataset['observations'][j] = traj['observations']
			dataset['actions'][j]  = traj['actions']
			all_returns += returns
			break

print('Average Returns:', all_returns / num_traj)
np.save(data_save_path, dataset)