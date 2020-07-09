import pickle
import argparse
import os.path as osp

import numpy as np
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-base", type=str, required=True)
    parser.add_argument("--buffer-target", type=str, required=True)
    args = parser.parse_args()

    save_file_output = osp.join(args.buffer_target, 'pool_pick_and_place.pkl')

    with open(args.buffer_base, 'rb') as f:
        replay_buffer_base = pickle.load(f)
    with open(args.buffer_target, 'rb') as f:
        replay_buffer_target = pickle.load(f)

    replay_buffer_base._rewards = 0.0*replay_buffer_base._rewards

    total_size = replay_buffer_base._top + replay_buffer_target._top + 1

    observation_keys = replay_buffer_base.observation_keys
    env = replay_buffer_base.env
    image_obs_key, state_obs_key = env.cnn_input_key, env.fc_input_key
    obs_keys = (image_obs_key, state_obs_key)

    output_pool = ObsDictReplayBuffer(
        total_size, env, observation_keys=observation_keys)

    for pool in [replay_buffer_base, replay_buffer_target]:
        env = pool.env

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

            path_done = ((env.terminates and pool._terminals[i]) or
                         (not env.terminates and
                          (i + 1) % env.scripted_traj_len == 0))
            if path_done:
                output_pool.add_path(path)
                path = dict(
                    rewards=[],
                    actions=[],
                    terminals=[],
                    observations=[],
                    next_observations=[],
                )

    data_directory = osp.dirname(args.buffer_target)
    print('Output size: {}'.format(output_pool._top))
    path = osp.join(data_directory, 'prior_data_merged.pkl')
    print(path)
    pickle.dump(output_pool, open(path, 'wb'), protocol=4)