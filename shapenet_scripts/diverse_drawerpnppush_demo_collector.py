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

NUM_TIMESTEPS=100

def collect(id):
    state_env = roboverse.make(
        'SawyerDiverseDrawerPnpPush-v0', 
        expl=True, 
        **kwargs
    )

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
        'observations': np.zeros((args.num_trajectories_per_demo//args.reset_interval, NUM_TIMESTEPS*args.reset_interval, imlength), dtype=np.uint8),
        'env': np.zeros((args.num_trajectories_per_demo//args.reset_interval, imlength), dtype=np.uint8),
        'skill_id': np.zeros((args.num_trajectories_per_demo//args.reset_interval, ), dtype=np.uint8)
    }

    for j in tqdm(range(args.num_trajectories_per_demo)):
        offset = j % args.reset_interval * NUM_TIMESTEPS 
        traj_j = j // args.reset_interval

        # is_done = False
        # while not is_done:
        env.demo_reset()
        if j % args.reset_interval == 0:
            recon_dataset['env'][traj_j, :] = np.uint8(env.render_obs().transpose()).flatten()
            trajectory = {
                'observations': [],
                'next_observations': [],
                'actions': np.zeros((NUM_TIMESTEPS*args.reset_interval, act_dim), dtype=np.float),
                'rewards': np.zeros((NUM_TIMESTEPS*args.reset_interval), dtype=np.float),
                'terminals': np.zeros((NUM_TIMESTEPS*args.reset_interval), dtype=np.uint8),
                'agent_infos': np.zeros((NUM_TIMESTEPS*args.reset_interval), dtype=np.uint8),
                'env_infos': np.zeros((NUM_TIMESTEPS*args.reset_interval), dtype=np.uint8),
                'skill_id': 0,
            }
        for i in range(NUM_TIMESTEPS):
            i_offset = i + offset
            img = np.uint8(env.render_obs())
            recon_dataset['observations'][traj_j, i_offset, :] = img.transpose().flatten()

            observation = env.get_observation()

            action, done = env.get_demo_action(first_timestep=(i == 0), return_done=True)
            next_observation, reward, _, info = env.step(action)
            if done:
                is_done = True

            trajectory['observations'].append(observation)
            trajectory['actions'][i_offset, :] = action
            trajectory['next_observations'].append(next_observation)
            trajectory['rewards'][i_offset] = reward

            ## TODO(patrick): not correct if reset_interval > 1
            trajectory['skill_id'] = info['skill_id']
            recon_dataset['skill_id'][traj_j] = info['skill_id']

        if j % args.reset_interval == args.reset_interval - 1:
            demo_dataset.append(trajectory)

    ## Save contents    
    file = open(os.path.join(prefix, f'{id}_demos.pkl'), 'wb')
    pkl.dump(demo_dataset, file)
    file.close()
    del demo_dataset
    gc.collect()
    demo_dataset = []

    np.save(os.path.join(prefix, f'{id}_images.npy'), recon_dataset)
    del recon_dataset
    gc.collect()
    recon_dataset = []

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='/media/ashvin/data1/patrickhaoy/data/test')
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--num_trajectories_per_demo", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fix_camera_yaw_pitch', action='store_true')
    parser.add_argument('--reset_interval', type=int, default=1)

    args = parser.parse_args()
    assert args.num_trajectories % args.num_trajectories_per_demo == 0
    assert args.num_trajectories_per_demo % args.reset_interval == 0
    prefix = args.save_path

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    kwargs = {
        'demo_num_ts': NUM_TIMESTEPS,
        'reset_interval': args.reset_interval,
        'expert_policy_std': .05,
        'downsample': True,
        'env_obs_img_dim': 196,
        'random_init_gripper_pos': True,
        'random_init_gripper_yaw': True,
        'use_target_config': args.debug,
        'fix_camera_yaw_pitch': args.fix_camera_yaw_pitch,
    }

    pool = Pool(args.num_threads)
    ids = [id for id in range(args.num_trajectories // args.num_trajectories_per_demo)]
    results = pool.map(collect, ids)
    pool.close()
    pool.join()