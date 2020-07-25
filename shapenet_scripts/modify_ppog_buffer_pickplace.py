import pickle
import numpy as np
import roboverse
from tqdm import tqdm

from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

INPUT_BUFFER = (
    '/nfs/kun1/users/avi/batch_rl_datasets/jul24_max1reset_zeroaction_Widow200'
    'GraspV6DrawerPlaceThenOpenV0PickPlaceOnly-v0_pixels_debug_40K_sparse_rewa'
    'rd_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl')

OUTPUT_BUFFER = (
    '/nfs/kun1/users/avi/batch_rl_datasets/jul24_max1reset_zeroaction_Widow200'
    'GraspV6DrawerPlaceThenOpenV0PickPlaceOnly-v0_pixels_debug_40K_sparse_rewa'
    'rd_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated_removelastactions.pkl')


def main():
    pool = pickle.load(open(INPUT_BUFFER, 'rb'))

    input_buffer_size = pool._top
    env = roboverse.make('Widow200GraspV6DrawerPlaceThenOpenV0PickPlaceOnly-v0',
                         observation_mode='pixels_debug', transpose_image=True)

    image_obs_key, state_obs_key = env.cnn_input_key, env.fc_input_key
    obs_keys = (image_obs_key, state_obs_key)
    output_buffer = ObsDictReplayBuffer(
        input_buffer_size + 100, env, observation_keys=obs_keys)

    traj_len = 30
    assert input_buffer_size%traj_len == 0
    num_traj = int(input_buffer_size/traj_len)
    print('num_traj', num_traj)

    for k in tqdm(range(num_traj)):
        traj_start_ind = k*traj_len
        traj_end_ind = (k+1)*traj_len
        path = dict(
            rewards=[],
            actions=[],
            terminals=[],
            observations=[],
            next_observations=[],
        )

        for i in range(traj_start_ind, traj_end_ind):
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

            if path_done or pool._actions[i][5] > 0.5:
                output_buffer.add_path(path)

                path = dict(
                    rewards=[],
                    actions=[],
                    terminals=[],
                    observations=[],
                    next_observations=[],
                )
                break

    with open(OUTPUT_BUFFER, 'wb+') as fp:
        pickle.dump(output_buffer, fp)
        print('saved to {}'.format(OUTPUT_BUFFER), protocol=4)


if __name__ == "__main__":
    main()