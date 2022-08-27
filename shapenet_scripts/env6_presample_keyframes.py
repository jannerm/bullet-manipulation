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
from multiprocess import Pool
import gc

from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands  # NOQA

def collect(id):
    state_env = roboverse.make('SawyerRigAffordances-v6', expl=True, **kwargs)

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
    num_datasets = 0
    demo_dataset = []

    recon_dataset = {
        'observations': np.zeros((args.num_trajectories_per_demo, args.num_timesteps, imlength), dtype=np.uint8),
        'env': np.zeros((args.num_trajectories_per_demo, imlength), dtype=np.uint8),
        'skill_id': np.zeros((args.num_trajectories_per_demo, ), dtype=np.uint8)
    }

    for j in tqdm(range(args.num_trajectories_per_demo)):
        env.demo_reset()
        recon_dataset['env'][j, :] = np.uint8(env.render_obs().transpose()).flatten()
        trajectory = {
            'observations': [],
            'next_observations': [],
            'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float),
            'rewards': np.zeros((args.num_timesteps), dtype=np.float),
            'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
            'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
            'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
            'skill_id': 0,
        }
        for i in range(args.num_timesteps):
            img = np.uint8(env.render_obs())
            recon_dataset['observations'][j, i, :] = img.transpose().flatten()

            observation = env.get_observation()

            action = env.get_demo_action(first_timestep=(i == 0))
            next_observation, reward, done, info = env.step(action)

            trajectory['observations'].append(observation)
            trajectory['actions'][i, :] = action
            trajectory['next_observations'].append(next_observation)
            trajectory['rewards'][i] = reward
            trajectory['skill_id'] = info['skill_id']
            recon_dataset['skill_id'][j] = info['skill_id']

        demo_dataset.append(trajectory)

        if ((j + 1) % args.num_trajectories_per_demo) == 0:
            file = open(os.path.join(prefix, f'demos_{id}.pkl'), 'wb')
            pkl.dump(demo_dataset, file)
            file.close()

            del demo_dataset
            gc.collect()
            demo_dataset = []

            np.save(os.path.join(prefix, f'images_{id}.npy'), recon_dataset)
            del recon_dataset
            gc.collect()
            recon_dataset = []

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/media/ashvin/data1/patrickhaoy/data/resetfree_drawerpnppush')
    parser.add_argument('--test_env_seed', type=int)
    parser.add_argument("--num_trajectories", type=int, default=400)
    parser.add_argument("--num_trajectories_per_demo", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--num_timesteps", type=int, default=75)
    parser.add_argument("--reset_interval", type=int, default=20)

    args = parser.parse_args()
    prefix = os.path.join(args.data_path, f'keyframes_trajectories/evalseed{args.test_env_seed}')

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    kwargs = {
        'demo_num_ts': args.num_timesteps,
        'reset_interval': args.reset_interval,
        'expert_policy_std': .05,
        'downsample': True,
        'env_obs_img_dim': 196,
        'random_init_gripper_pos': True,
        'random_init_gripper_yaw': False,
    }
    if args.test_env_seed != -1:
        kwargs.update({
            'test_env_command': drawer_pnp_push_commands[args.test_env_seed],
            'use_test_env_command_sequence': False,
            'test_env': True,
        })

    assert args.num_trajectories % args.num_trajectories_per_demo == 0

    pool = Pool(args.num_threads)
    ids = [id for id in range(args.num_trajectories // args.num_trajectories_per_demo)]
    results = pool.map(collect, ids)
    pool.close()
    pool.join()