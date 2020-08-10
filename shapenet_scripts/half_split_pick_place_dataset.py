import pickle
import argparse
import os.path as osp
import numpy as np

from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

def reshape_array_by_traj(arr, num_trajs, traj_len, image=False):
    reshape_dims = ((num_trajs, traj_len, arr.shape[-1])
        if not image else (num_trajs, traj_len, 3, 48, 48))
    print("reshape_dims", reshape_dims)
    return np.reshape(arr, reshape_dims)

def split_half(arr):
    """
    arr: (num_trajs, traj_len, ...)
    """
    half_traj_len = arr.shape[1] // 2
    return arr[:,:half_traj_len], arr[:,half_traj_len:]

def add_all_data_into_buffer(pool, rewards_by_traj, actions_by_traj,
        terminals_by_traj, images_by_traj, robot_states_by_traj,
        n_images_by_traj, n_robot_states_by_traj):
    num_trajs = rewards_by_traj.shape[0]
    trunc_traj_len = rewards_by_traj.shape[1] # the truncated one; original / 2
    print("trunc_traj_len", trunc_traj_len)
    for i in range(num_trajs):
        path = dict(
            rewards=list(rewards_by_traj[i]),
            actions=list(actions_by_traj[i]),
            terminals=list(terminals_by_traj[i]),
            observations=[{image_obs_key:images_by_traj[i][j],
                          state_obs_key:robot_states_by_traj[i][j]} for j in range(trunc_traj_len)],
            next_observations=[{image_obs_key:n_images_by_traj[i][j],
                               state_obs_key:n_robot_states_by_traj[i][j]} for j in range(trunc_traj_len)],
        )
        pool.add_path(path)

    del rewards_by_traj
    del actions_by_traj
    del terminals_by_traj
    del images_by_traj
    del robot_states_by_traj
    del n_images_by_traj
    del n_robot_states_by_traj

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
    observation_keys = input_pool.observation_keys

    pool_pick = ObsDictReplayBuffer(
        pool_size, env, observation_keys=observation_keys)
    pool_place = ObsDictReplayBuffer(
        pool_size, env, observation_keys=observation_keys)

    traj_len = env.scripted_traj_len
    assert pool_size % traj_len == 0
    num_trajs = pool_size // traj_len
    print("pool_size", pool_size)
    print("input_pool._rewards.shape", input_pool._rewards.shape)
    print("input_pool._actions.shape", input_pool._actions.shape)
    print("input_pool._terminals.shape", input_pool._terminals.shape)
    print("input_pool._obs", input_pool._obs)
    print("input_pool._next_obs", input_pool._next_obs)

    rewards_by_traj = reshape_array_by_traj(input_pool._rewards[:pool_size], num_trajs, traj_len)
    actions_by_traj = reshape_array_by_traj(input_pool._actions[:pool_size], num_trajs, traj_len)
    terminals_by_traj = reshape_array_by_traj(input_pool._terminals[:pool_size], num_trajs, traj_len)
    images_by_traj = reshape_array_by_traj(input_pool._obs[image_obs_key][:pool_size], num_trajs, traj_len, image=True)
    robot_states_by_traj = reshape_array_by_traj(input_pool._obs[state_obs_key][:pool_size], num_trajs, traj_len)
    n_images_by_traj = reshape_array_by_traj(input_pool._next_obs[image_obs_key][:pool_size], num_trajs, traj_len, image=True)
    n_robot_states_by_traj = reshape_array_by_traj(input_pool._next_obs[state_obs_key][:pool_size], num_trajs, traj_len)

    del input_pool

    # print("np.squeeze(rewards_by_traj)", np.squeeze(rewards_by_traj))
    rewards_by_traj_pt0, rewards_by_traj_pt1 = split_half(rewards_by_traj)
    # print("np.squeeze(rewards_by_traj_pt0)", np.squeeze(rewards_by_traj_pt0))
    # print("np.squeeze(rewards_by_traj_pt1)", np.squeeze(rewards_by_traj_pt1))
    actions_by_traj_pt0, actions_by_traj_pt1 = split_half(actions_by_traj)
    terminals_by_traj_pt0, terminals_by_traj_pt1 = split_half(terminals_by_traj)
    images_by_traj_pt0, images_by_traj_pt1 = split_half(images_by_traj)
    robot_states_by_traj_pt0, robot_states_by_traj_pt1 = split_half(robot_states_by_traj)
    n_images_by_traj_pt0, n_images_by_traj_pt1 = split_half(n_images_by_traj)
    n_robot_states_by_traj_pt0, n_robot_states_by_traj_pt1 = split_half(n_robot_states_by_traj)

    del rewards_by_traj
    del actions_by_traj
    del terminals_by_traj
    del images_by_traj
    del robot_states_by_traj
    del n_images_by_traj
    del n_robot_states_by_traj

    add_all_data_into_buffer(pool_pick, rewards_by_traj_pt0, actions_by_traj_pt0,
        terminals_by_traj_pt0, images_by_traj_pt0, robot_states_by_traj_pt0,
        n_images_by_traj_pt0, n_robot_states_by_traj_pt0)
    add_all_data_into_buffer(pool_place, rewards_by_traj_pt1, actions_by_traj_pt1,
        terminals_by_traj_pt1, images_by_traj_pt1, robot_states_by_traj_pt1,
        n_images_by_traj_pt1, n_robot_states_by_traj_pt1)

    save_file_place = osp.join(input_buffer_dir, 'pool_place.pkl')
    pickle.dump(pool_place, open(save_file_place, 'wb'), protocol=4)
    print('Pick size: {}'.format(pool_pick._top))

    save_file_pick = osp.join(input_buffer_dir, 'pool_pick.pkl')
    pickle.dump(pool_pick, open(save_file_pick, 'wb'), protocol=4)
    print('Place size: {}'.format(pool_place._top))

