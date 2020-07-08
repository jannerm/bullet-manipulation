import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import os.path as osp
import itertools as it

# FILE_PICK = '/media/avi/data/Work/data/2020-07-02T11-09-18_pool_50001_pick.pkl'
# FILE_PLACE = '/media/avi/data/Work/data/2020-07-02T16-59-49_pool_20001_place.pkl'
# FILE_PICK = '/media/avi/data/Work/github/jannerm/bullet-manipulation/data/july6_test_4_Widow200GraspV7BoxV0-v0_pixels_debug_50_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
# FILE_PLACE = '/media/avi/data/Work/github/jannerm/bullet-manipulation/data/july6_test_2_Widow200GraspV6BoxPlaceOnlyV0-v0_pixels_debug_50_sparse_reward_scripted_actions_fixed_position_noise_std_0.05/railrl_consolidated.pkl'
# OUTPUT_DIR = '/media/avi/data/Work/data/stitch_debug_v7_part2/'
FILE_PICK = '/nfs/kun1/users/avi/batch_rl_datasets/july6_Widow200GraspV7BoxV0-v0_pixels_debug_40K_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
FILE_PLACE = '/nfs/kun1/users/avi/batch_rl_datasets/july6_Widow200GraspV6BoxPlaceOnlyV0-v0_pixels_debug_40K_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
FILE_PICK_PLACE = '/nfs/kun1/users/albert/batch_rl_datasets/jun26_Widow200GraspV6BoxPlaceV0-v0_pixels_debug_10K_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
OUTPUT_DIR = 'stitch_data'
PICK_STATE_ARRAY = osp.join(OUTPUT_DIR, "pick_robot_state.npy")
PLACE_STATE_ARRAY = osp.join(OUTPUT_DIR, "place_robot_state.npy")
PICK_REWARDS_ARRAY = osp.join(OUTPUT_DIR, "pick_rewards.npy")
PLACE_REWARDS_ARRAY = osp.join(OUTPUT_DIR, "place_rewards.npy")
PICK_PLACE_STATE_ARRAY = osp.join(OUTPUT_DIR, "pick_place_robot_state.npy")
PICK_PLACE_REWARDS_ARRAY = osp.join(OUTPUT_DIR, "pick_place_robot_state.npy")

if not osp.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
def process_image(image):
    image = np.reshape(image, (3, 48, 48))
    image = np.transpose(image, (1, 2, 0))
    return image

def save_to_npy(rb, prefix):
    fnames_arrays = [
        ("robot_state", rb._obs['robot_state']),
        ("rewards", rb._rewards)
    ]
    for name, array in fnames_arrays:
        np.save("{}/{}_{}".format(OUTPUT_DIR, prefix, name), array)

if not osp.exists(PICK_STATE_ARRAY) or not osp.exists(PICK_REWARDS_ARRAY):
    with open(FILE_PICK, 'rb') as f:
        pick = pickle.load(f)
    save_to_npy(pick, "pick")

if not osp.exists(PLACE_STATE_ARRAY) or not osp.exists(PLACE_REWARDS_ARRAY):
    with open(FILE_PLACE, 'rb') as f:
        place = pickle.load(f)
    save_to_npy(place, "place")

if not osp.exists(PICK_PLACE_STATE_ARRAY) or not osp.exists(PICK_PLACE_REWARDS_ARRAY):
    with open(FILE_PICK_PLACE, 'rb') as f:
        pick_place = pickle.load(f)
    save_to_npy(pick_place, "pick_place")

def reshape_obs_by_traj(obs_array, traj_len):
    print("obs_array.shape", obs_array.shape)
    return np.reshape(obs_array, (obs_array.shape[0] // traj_len, traj_len, obs_array.shape[1]))

def get_final_traj_rewards(rewards_array, traj_len):
    print("rewards_array.shape", rewards_array.shape)
    rewards_array = np.reshape(rewards_array, (rewards_array.shape[0] // traj_len, traj_len, rewards_array.shape[1]))
    return rewards_array[:,-1]

def get_successful_obs_traj(obs_array_by_traj, final_rewards):
    """
    obs_array_by_traj: (num_trajs, traj_len, obs_size)
    final_rewards: (num_trajs, 1)
    """
    assert obs_array_by_traj.shape[0] == final_rewards.shape[0]
    print("final_rewards.shape", final_rewards.shape)
    success_idx = np.argwhere(final_rewards == 1)[:,0]
    fail_idx = np.argwhere(final_rewards == 0)[:,1]
    success_trajs = obs_array_by_traj[success_idx]
    fail_trajs = obs_array_by_traj[fail_idx]
    print("success_trajs.shape", success_trajs.shape)
    print("fail_trajs.shape", fail_trajs.shape)
    return success_trajs

def get_obs_first_reward(obs_array_by_traj):
    """
    obs_array_by_traj: (num_trajs, traj_len, obs_size)
    rewards_array_by_traj: (num_trajs, traj_len, 1)
    """
    z_idx = 2
    grasptime_obs_array_by_traj = []
    for i in range(obs_array_by_traj.shape[0]):
        print("obs_array_by_traj.shape", obs_array_by_traj.shape)
        # print("obs_array_by_traj[i]", obs_array_by_traj[i])
        t_with_closed_gripper = np.argwhere(obs_array_by_traj[i][:,z_idx] > -.2).squeeze()
        if len(t_with_closed_gripper.shape) == 0 or len(list(t_with_closed_gripper)) == 0:
            # gripper never closes
            grasptime_obs_array_by_traj.append(np.array([np.nan] * obs_array_by_traj.shape[-1]))
            continue

        grasp_exec_time = t_with_closed_gripper[0]
        print("obs_array_by_traj[i][grasp_exec_time]", obs_array_by_traj[i][grasp_exec_time])
        grasptime_obs_array_by_traj.append(obs_array_by_traj[i][grasp_exec_time])
    return np.array(grasptime_obs_array_by_traj)
    
def plot_trajs(x_list, y_list, num_to_plot, color, plot_name, alpha=0.05, zorder=-1):
    plt.xlabel(plot_name[0])
    plt.ylabel(plot_name[1])
    plt.title(plot_name)
    for i in range(num_to_plot):
        plt.plot(x_list[i], y_list[i], color=color, alpha=alpha, zorder=zorder)

def plot_dots(x_list, y_list, num_to_plot, color, dot="last", alpha=0.2):
    assert dot in ["first", "last", None]
    if dot is not None:
        dot_idx = -1 if dot == "last" else 0
        x_data, y_data = x_list[:num_to_plot][:,dot_idx], y_list[:num_to_plot][:,dot_idx]
    else:
        x_data, y_data = x_list[:num_to_plot], y_list[:num_to_plot]
    plt.scatter(x_data, y_data, color=color, alpha=alpha, zorder=1)

def plot_trajs_dots_2dims(dim0, dim1, dims_name, pick_success_obs, place_success_obs, pick_place_success_obs, pick_place_first_lifted_pos, num_to_plot):
    plt.figure(next(plot_num_counter))
    plot_trajs(pick_success_obs[:,:,dim0], pick_success_obs[:,:,dim1], num_to_plot, "green", dims_name)
    plot_trajs(place_success_obs[:,:,dim0], place_success_obs[:,:,dim1], num_to_plot, "orange", dims_name)
    plot_dots(pick_success_obs[:,:,dim0], pick_success_obs[:,:,dim1], num_to_plot, "darkgreen", dot="last")
    plot_dots(place_success_obs[:,:,dim0], place_success_obs[:,:,dim1], num_to_plot, "darkorange", dot="first")
    plot_trajs(pick_place_success_obs[:,:,dim0], pick_place_success_obs[:,:,dim1], num_to_plot // 10, "blue", dims_name, zorder=2)
    plot_dots(pick_place_first_lifted_pos[:,dim0], pick_place_first_lifted_pos[:,dim1], num_to_plot // 10, "darkblue", dot=None)
    plt.savefig("stitch_data/{}.png".format(dims_name))

def plot_all_trajs_and_dots(pick_success_obs, place_success_obs, pick_place_success_obs, pick_place_first_lifted_pos, num_to_plot):
    dims_list_to_plot = [(0, 1, "xy"), (1, 2, "yz"), (0, 2, "xz")]
    for dim0, dim1, dims_name in dims_list_to_plot:
        plot_trajs_dots_2dims(dim0, dim1, dims_name, pick_success_obs, place_success_obs, pick_place_success_obs, pick_place_first_lifted_pos, num_to_plot)

num_to_plot = 500
place_starting_img = []
pick_final_img = []
num_succ = 0
pick_traj_len = 25
place_traj_len = 10
pick_place_traj_len = 30
empty_traj_size = 100000
plot_num_counter = it.count(0)

# Process PICK data
# pick_obs_array = pick._obs['robot_state'][:-empty_traj_size]
pick_obs_array = np.load(PICK_STATE_ARRAY)[:-empty_traj_size]
pick_obs_array = reshape_obs_by_traj(pick_obs_array, pick_traj_len)
# pick_rewards_array = pick._rewards[:-empty_traj_size]
pick_rewards_array = np.load(PICK_REWARDS_ARRAY)[:-empty_traj_size]
pick_final_rewards_array = get_final_traj_rewards(pick_rewards_array, pick_traj_len)
pick_successful_obs_array = get_successful_obs_traj(pick_obs_array, pick_final_rewards_array)

# Process PLACE data
# place_obs_array = place._obs['robot_state'][:-empty_traj_size]
place_obs_array = np.load(PLACE_STATE_ARRAY)[:-empty_traj_size]
place_obs_array = reshape_obs_by_traj(place_obs_array, place_traj_len)
# place_rewards_array = place._rewards[:-empty_traj_size]
place_rewards_array = np.load(PLACE_REWARDS_ARRAY)[:-empty_traj_size]
place_final_rewards_array = get_final_traj_rewards(place_rewards_array, place_traj_len)
place_successful_obs_array = get_successful_obs_traj(place_obs_array, place_final_rewards_array)

# Process PICK PLACE data
# place_obs_array = place._obs['robot_state'][:-empty_traj_size]
pick_place_obs_array = np.load(PICK_PLACE_STATE_ARRAY)[:-empty_traj_size]
pick_place_obs_array = reshape_obs_by_traj(pick_place_obs_array, pick_place_traj_len)
# place_rewards_array = place._rewards[:-empty_traj_size]
pick_place_rewards_array = np.load(PICK_PLACE_REWARDS_ARRAY)[:-empty_traj_size]
pick_place_final_rewards_array = get_final_traj_rewards(pick_place_rewards_array, pick_place_traj_len)
pick_place_successful_obs_array = get_successful_obs_traj(pick_place_obs_array, pick_place_final_rewards_array)

print("pick_place_obs_array.shape", pick_place_obs_array.shape)
pick_place_first_lifted_pos = get_obs_first_reward(pick_place_successful_obs_array)
print("pick_place_first_lifted_pos.shape", pick_place_first_lifted_pos.shape)

plot_all_trajs_and_dots(pick_successful_obs_array, place_successful_obs_array, pick_place_successful_obs_array, pick_place_first_lifted_pos, num_to_plot=num_to_plot)

# for i in range(0, pick._rewards.shape[0], 5):
#     if pick._rewards[i] > 0.0:
#         # pick_final_states.append(pick._obs['robot_state'][i][:3])
#         img = pick._obs['image'][i]
#         img = process_image(img)
#         plt.figure(num_succ)
#         # plt.axis('off')
#         save_file = osp.join(OUTPUT_DIR, 'pick_{}.png'.format(num_succ))
#         plt.imsave(save_file, img)
#         place_starting_img.append(img)
#         num_succ += 1
#     if num_succ > points_to_plot:
#         break
# for i in range(points_to_plot):
#     img = place._obs['image'][i*10]
#     img = process_image(img)
#     # plt.axis('off')
#     # plt.imshow(img)
#     plt.figure(num_succ)
#     save_file = osp.join(OUTPUT_DIR, 'place_{}.png'.format(i))
#     plt.imsave(save_file, img)
#     place_starting_img.append(img)
