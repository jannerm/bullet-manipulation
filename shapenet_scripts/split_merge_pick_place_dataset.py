import pickle
import argparse
import os.path as osp

import numpy as np
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, required=True)
    args = parser.parse_args()

    input_buffer_dir = osp.dirname(args.buffer)

    with open(args.buffer, 'rb') as fp:
        input_pool = pickle.load(fp)

    env = input_pool.env
    image_obs_key, state_obs_key = env.cnn_input_key, env.fc_input_key
    pool_size = input_pool._top
    print('Input buffer size: {}'.format(pool_size))
    observation_keys = input_pool.observation_keys

    output_pool = ObsDictReplayBuffer(
         pool_size, env, observation_keys=observation_keys)

    traj_len = env.scripted_traj_len
    assert pool_size % traj_len == 0
    num_traj = int(input_pool._top / traj_len)
    # assert num_traj % 2 == 0
    mid_point = int(num_traj/2)

    for j in range(mid_point):
        start = j*traj_len
        end = start + traj_len
        path = dict(
            rewards=[],
            actions=[],
            terminals=[],
            observations=[],
            next_observations=[],
        )
        for i in range(start, end):
            path['rewards'].append(input_pool._rewards[i])
            path['actions'].append(input_pool._actions[i])
            path['terminals'].append(input_pool._terminals[i])
            path['observations'].append(
                {image_obs_key: input_pool._obs[image_obs_key][i],
                 state_obs_key: input_pool._obs[state_obs_key][i]}
            )
            path['next_observations'].append(
                {image_obs_key: input_pool._next_obs[image_obs_key][i],
                 state_obs_key: input_pool._next_obs[state_obs_key][i]}
            )

            if input_pool._actions[i][4] < -0.5:
                output_pool.add_path(path)
                break

    output_pool._rewards = 0.0*output_pool._rewards

    for j in range(mid_point, num_traj):
        start = j*traj_len
        end = start + traj_len
        path = dict(
            rewards=[],
            actions=[],
            terminals=[],
            observations=[],
            next_observations=[],
        )
        add_data = False
        for i in range(start+1, end):
            if input_pool._actions[i-1][4] < -0.5:
                add_data = True
            if add_data:
                path['rewards'].append(input_pool._rewards[i])
                path['actions'].append(input_pool._actions[i])
                path['terminals'].append(input_pool._terminals[i])
                path['observations'].append(
                    {image_obs_key: input_pool._obs[image_obs_key][i],
                     state_obs_key: input_pool._obs[state_obs_key][i]}
                )
                path['next_observations'].append(
                    {image_obs_key: input_pool._next_obs[image_obs_key][i],
                     state_obs_key: input_pool._next_obs[state_obs_key][i]}
                )

        if len(path['rewards']) > 0:
            output_pool.add_path(path)

    save_file_place = osp.join(input_buffer_dir, 'pool_pick_and_place.pkl')
    pickle.dump(output_pool, open(save_file_place, 'wb'), protocol=4)
    print('Output pool size: {}'.format(output_pool._top))
