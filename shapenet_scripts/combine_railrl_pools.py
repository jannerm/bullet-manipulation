import pickle
import argparse
import roboverse
import os
import os.path as osp
import numpy as np
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
from roboverse.envs.env_list import PROXY_ENVS_MAP

EXTRA_POOL_SPACE = int(1e5)
REWARD_NEGATIVE = -10
REWARD_POSITIVE = 1
NFS_PATH = '/nfs/kun1/users/avi/batch_rl_datasets/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, required=True)
    parser.add_argument("-d", "--data-directory", type=str, required=True)
    parser.add_argument("-o", "--observation-mode", type=str, default='state',
                        choices=('state', 'pixels', 'pixels_debug'))
    parser.add_argument('--downwards', action='store_true',default=False)
    parser.add_argument("--success-only", dest="success_only", action="store_true", default=False)
    parser.add_argument('--output', type=str, default='railrl_consolidated.pkl')
    args = parser.parse_args()

    # if args.env == 

    if osp.exists(NFS_PATH):
        data_directory = osp.join(NFS_PATH, args.data_directory)
    else:
        data_directory = osp.join(
            os.path.dirname(__file__), "..", 'data', args.data_directory)
    print(data_directory)
    # keys = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
    timestamp = roboverse.utils.timestamp()

    pools = []

    for root, dirs, files in os.walk(data_directory):
        for f in files:
            merge_f = ("pool" in f) and (("success_only" in f) == args.success_only)
            if merge_f:
                with open(os.path.join(root, f), 'rb') as fp:
                    print("f", f)
                    pool = pickle.load(fp)
                pools.append(pool)

    original_pool_size = 0
    for pool in pools:
        original_pool_size += pool._top
    pool_size = original_pool_size + EXTRA_POOL_SPACE

    if args.env in PROXY_ENVS_MAP:
        roboverse_env_name = PROXY_ENVS_MAP[args.env]
    else:
        roboverse_env_name = args.env
        
    if args.downwards:
        env = roboverse.make(roboverse_env_name,
                         observation_mode=args.observation_mode,
                         transpose_image=True, downwards = True)
    else:     
        env = roboverse.make(roboverse_env_name,
                            observation_mode=args.observation_mode,
                            transpose_image=True)
    if args.observation_mode == 'state':
        consolidated_pool = EnvReplayBuffer(pool_size, env)
        for pool in pools:
            for i in range(pool._top):

                # if pool._rewards[i] < 0:
                #     reward_corrected = REWARD_NEGATIVE
                # elif pool._rewards[i] > 0:
                #     reward_corrected = REWARD_POSITIVE
                # else:
                #     raise ValueError

                reward_corrected = pool._rewards[i]
                consolidated_pool.add_sample(
                    observation=pool._observations[i],
                    action=pool._actions[i],
                    reward=reward_corrected,
                    next_observation=pool._next_obs[i],
                    terminal=pool._terminals[i],
                    env_info={},
                )

    elif args.observation_mode in ['pixels', 'pixels_debug']:
        try:
            image_obs_key, state_obs_key = env.cnn_input_key, env.fc_input_key
        except:
            image_obs_key, state_obs_key = ('image','state',)
        obs_keys = (image_obs_key, state_obs_key)
        consolidated_pool = ObsDictReplayBuffer(pool_size, env,
                                                observation_keys=obs_keys)
        for pool in pools:
            # import IPython; IPython.embed()
            # import ipdb; ipdb.set_trace() 
            path = dict(
                rewards=[],
                actions=[],
                terminals=[],
                observations=[],
                next_observations=[],
            )
            for i in range(pool._top):
                path['rewards'].append(pool._rewards[i])
                path['actions'].append(pool._actions[i])
                path['terminals'].append(pool._terminals[i])
                path['observations'].append(
                    {image_obs_key: pool._obs[image_obs_key][i],
                         state_obs_key: pool._obs[state_obs_key][i]}
                )
                path['next_observations'].append(
                    {image_obs_key: pool._next_obs[image_obs_key][i],
                         state_obs_key: pool._next_obs[state_obs_key][i]}
                )

                # path_done = ((env.terminates and pool._terminals[i]) or
                #              (not env.terminates and
                #               (i + 1) % env.scripted_traj_len == 0))
                path_done = True

                if path_done:
                    consolidated_pool.add_path(path)
                    path = dict(
                        rewards=[],
                        actions=[],
                        terminals=[],
                        observations=[],
                        next_observations=[],
                    )

            # if len(path['rewards']) > 0:
                # consolidated_pool.add_path(path)

    else:
        raise NotImplementedError

    if not args.success_only:
        path = osp.join(data_directory, 'railrl_consolidated.pkl')
    else:
        path = osp.join(data_directory, 'railrl_consolidated_success.pkl')

    # path = osp.join(data_directory, args.output)
    pickle.dump(consolidated_pool, open(path, 'wb'), protocol=4)
