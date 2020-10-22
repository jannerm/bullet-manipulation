import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg 
import os
from PIL import Image
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=5000)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--subset", type=str, default='train')
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")

args = parser.parse_args()
demo_data_save_path = "/home/ashvin/data/sasha/demos/gr_" + args.name + "_demos"
recon_data_save_path = "/home/ashvin/data/sasha/demos/gr_" + args.name + "_images.npy"

state_env = roboverse.make('SawyerRigAffordances-v0')

# FOR TESTING, TURN COLORS OFF
imsize = state_env.obs_img_dim

renderer_kwargs=dict(
        create_image_format='HWC',
        output_image_format='CWH',
        width=imsize,
        height=imsize,
        flatten_image=True,)

renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
env = InsertImageEnv(state_env, renderer=renderer)
imlength = env.obs_img_dim * env.obs_img_dim * 3

success = 0
returns = 0
act_dim = env.action_space.shape[0]

demo_dataset = []
recon_dataset = {
    'observations': np.zeros((args.num_trajectories, args.num_timesteps, imlength), dtype=np.uint8),
    'object': [],
    'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
}

avg_tasks_done = 0
for j in tqdm(range(args.num_trajectories)):
    env.demo_reset()
    recon_dataset['env'][j, :] = np.uint8(env.render_obs().transpose()).flatten()
    recon_dataset['object'].append(env.curr_object)
    trajectory = {
        'observations': [],
        'next_observations': [],
        'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float),
        'rewards': np.zeros((args.num_timesteps), dtype=np.float),
        'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
        'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'object_name': env.curr_object,
    }

    for i in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        recon_dataset['observations'][j, i, :] = img.transpose().flatten()

        observation = env.get_observation()

        action = env.get_demo_action()
        next_observation, reward, done, info = env.step(action)

        trajectory['observations'].append(observation)
        trajectory['actions'][i, :] = action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][i] = reward

    demo_dataset.append(trajectory)
    avg_tasks_done += env.tasks_done

print('Success Rate: {}'.format(avg_tasks_done / args.num_trajectories))
np.save(recon_data_save_path, recon_dataset)
step_size = 1000

num_steps = math.ceil(args.num_trajectories / step_size)
for i in range(num_steps):
    curr_name = demo_data_save_path + '_{0}.pkl'.format(i)
    start_ind, end_ind = i*step_size, (i+1)*step_size
    curr_data = demo_dataset[start_ind:end_ind]
    file = open(curr_name, 'wb')
    pkl.dump(curr_data, file)
    file.close()
