import argparse
import os

import numpy as np
import pickle
from tqdm import tqdm

import roboverse
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str)
parser.add_argument('--num_trajectories', type=int, default=8000)
parser.add_argument('--num_timesteps', type=int, default=75)
parser.add_argument('--reset_interval', type=int, default=10)
parser.add_argument('--num_trajectories_per_file', type=int, default=500)
parser.add_argument('--downsample', action='store_true')

parser.add_argument('--drawer_sliding', action='store_true')
parser.add_argument('--fix_drawer_orientation', action='store_true')
parser.add_argument('--fix_drawer_orientation_semicircle', action='store_true')
parser.add_argument('--new_view', action='store_true')
parser.add_argument('--close_view', action='store_true')
parser.add_argument('--red_drawer_base', action='store_true')

args = parser.parse_args()

kwargs = {
    'drawer_sliding': args.drawer_sliding,
    'fix_drawer_orientation': args.fix_drawer_orientation,
    'fix_drawer_orientation_semicircle': (
        args.fix_drawer_orientation_semicircle),
    'new_view': args.new_view,
    'close_view': args.close_view,
    'red_drawer_base': args.red_drawer_base,
}

if args.downsample:
    kwargs['downsample'] = True
    kwargs['env_obs_img_dim'] = 196

state_env = roboverse.make(
    'SawyerRigAffordances-v1',
    random_color_p=0.0,
    expl=True,
    reset_interval=args.reset_interval,
    **kwargs)

# FOR TESTING, TURN COLORS OFF
imsize = state_env.obs_img_dim

renderer_kwargs = dict(
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
num_datasets = 0
demo_dataset = []
recon_dataset = {
    'observations': np.zeros(
        (args.num_trajectories, args.num_timesteps, imlength),
        dtype=np.uint8),
    'object': [],
    'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
}

for traj_id in tqdm(range(args.num_trajectories)):
    env.demo_reset()

    recon_dataset['env'][traj_id, :] = np.uint8(
        env.render_obs().transpose()).flatten()
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

    for t in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        recon_dataset['observations'][traj_id, t, :] = (
            img.transpose().flatten())

        observation = env.get_observation()

        action = env.get_demo_action(
            first_timestep=(t == 0),
            final_timestep=(t == args.num_timesteps-1))
        next_observation, reward, done, info = env.step(action)

        trajectory['observations'].append(observation)
        trajectory['actions'][t, :] = action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][t] = reward

    demo_dataset.append(trajectory)

    if ((traj_id + 1) % args.num_trajectories_per_file) == 0:
        demo_data_save_path = os.path.join(
            args.output_dir, 'demos_%04d.pkl' % (num_datasets))

        file = open(demo_data_save_path, 'wb')
        pickle.dump(demo_dataset, file)
        file.close()

        num_datasets += 1
        demo_dataset = []

image_data_save_path = os.path.join(args.output_dir, 'images.npy')
np.save(image_data_save_path, recon_dataset)
