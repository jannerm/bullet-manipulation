import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import os.path as osp
import itertools as it

########### PARAMS TO MODIFY:
# Three variables: Base (lick pick), Target (like place), and Full (like pick and place)
FILE_BASE = 'shapenet_scripts/../data/test_jul20_v11_Widow200GraspV6DrawerOpenOnlyV0-v0_pixels_debug_100_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
FILE_TARGET = 'shapenet_scripts/../data/test_jul20_v11_Widow200GraspV6DrawerGraspOnlyV0-v0_pixels_debug_100_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
FILE_FULL = 'shapenet_scripts/../data/test_jul20_v3_Widow200GraspV6DrawerPlaceThenOpenV0-v0_pixels_debug_100_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
OUTPUT_DIR = 'stitch_data'

num_to_plot = 100
base_traj_len = 30
target_traj_len = 25
full_traj_len = 50
empty_traj_size = 100000
load_from_saved_npy = False
########## END PARAMS TO MODIFY

BASE_STATE_ARRAY = osp.join(OUTPUT_DIR, "base_robot_state.npy")
TARGET_STATE_ARRAY = osp.join(OUTPUT_DIR, "target_robot_state.npy")
BASE_REWARDS_ARRAY = osp.join(OUTPUT_DIR, "base_rewards.npy")
TARGET_REWARDS_ARRAY = osp.join(OUTPUT_DIR, "target_rewards.npy")
FULL_STATE_ARRAY = osp.join(OUTPUT_DIR, "full_robot_state.npy")
FULL_REWARDS_ARRAY = osp.join(OUTPUT_DIR, "full_robot_state.npy")

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

if load_from_saved_npy:
    if not osp.exists(BASE_STATE_ARRAY) or not osp.exists(BASE_REWARDS_ARRAY):
        with open(FILE_BASE, 'rb') as f:
            base = pickle.load(f)
        save_to_npy(base, "base")

    if not osp.exists(TARGET_STATE_ARRAY) or not osp.exists(TARGET_REWARDS_ARRAY):
        with open(FILE_TARGET, 'rb') as f:
            target = pickle.load(f)
        save_to_npy(target, "target")

    if not osp.exists(FULL_STATE_ARRAY) or not osp.exists(FULL_REWARDS_ARRAY):
        with open(FILE_FULL, 'rb') as f:
            full = pickle.load(f)
        save_to_npy(full, "full")
else:
    with open(FILE_BASE, 'rb') as f:
        base = pickle.load(f)

    with open(FILE_TARGET, 'rb') as f:
        target = pickle.load(f)

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

def plot_trajs_dots_2dims(dim0, dim1, dims_name, base_success_obs, target_success_obs, full_success_obs, full_first_lifted_pos, num_to_plot):
    plt.figure(next(plot_num_counter))
    plot_trajs(base_success_obs[:,:,dim0], base_success_obs[:,:,dim1], num_to_plot, "green", dims_name)
    plot_trajs(target_success_obs[:,:,dim0], target_success_obs[:,:,dim1], num_to_plot, "orange", dims_name)
    plot_dots(base_success_obs[:,:,dim0], base_success_obs[:,:,dim1], num_to_plot, "darkgreen", dot="last")
    plot_dots(target_success_obs[:,:,dim0], target_success_obs[:,:,dim1], num_to_plot, "darkorange", dot="first")
    plot_trajs(full_success_obs[:,:,dim0], full_success_obs[:,:,dim1], num_to_plot // 10, "blue", dims_name, zorder=2)
    plot_dots(full_first_lifted_pos[:,dim0], full_first_lifted_pos[:,dim1], num_to_plot // 10, "darkblue", dot=None)
    plt.savefig("{}/{}.png".format(OUTPUT_DIR, dims_name))

def plot_all_trajs_and_dots(base_success_obs, target_success_obs, full_success_obs, full_first_lifted_pos, num_to_plot):
    dims_list_to_plot = [(0, 1, "xy"), (1, 2, "yz"), (0, 2, "xz")]
    for dim0, dim1, dims_name in dims_list_to_plot:
        plot_trajs_dots_2dims(dim0, dim1, dims_name, base_success_obs, target_success_obs, full_success_obs, full_first_lifted_pos, num_to_plot)

def get_avg_theta_traj(obs_traj_array):
    assert obs_traj_array.shape[2] == 5
    print("obs_traj_array.shape", obs_traj_array.shape)
    theta_idx = 3
    return np.mean(obs_traj_array[:,:,theta_idx], axis=0)

def plot_avg_theta(base_avg_theta_traj, target_avg_theta_traj, full_avg_theta_traj):
    plt.figure(next(plot_num_counter))
    plt.title("Average Theta")
    plt.plot(list(range(len(base_avg_theta_traj))), base_avg_theta_traj, label="Avg Base Theta")
    plt.plot(list(np.array(list(range(len(target_avg_theta_traj)))) + len(base_avg_theta_traj)), target_avg_theta_traj, label="Avg Target Theta")
    plt.plot(list(range(len(full_avg_theta_traj))), full_avg_theta_traj, label="Avg Full Theta")
    plt.legend(loc=3)
    plt.xlabel("Timestep")
    plt.ylabel("Theta")
    plt.savefig("{}/{}.png".format(OUTPUT_DIR, "avg_theta"))

target_starting_img = []
base_final_img = []
num_succ = 0
plot_num_counter = it.count(0)

# # Process BASE data
# # base_obs_array = base._obs['robot_state'][:-empty_traj_size]
# base_obs_array = np.load(BASE_STATE_ARRAY)[:-empty_traj_size]
# base_obs_array = reshape_obs_by_traj(base_obs_array, base_traj_len)
# # base_rewards_array = base._rewards[:-empty_traj_size]
# base_rewards_array = np.load(BASE_REWARDS_ARRAY)[:-empty_traj_size]
# base_final_rewards_array = get_final_traj_rewards(base_rewards_array, base_traj_len)
# base_successful_obs_array = get_successful_obs_traj(base_obs_array, base_final_rewards_array)

# # Process TARGET data
# # target_obs_array = target._obs['robot_state'][:-empty_traj_size]
# target_obs_array = np.load(TARGET_STATE_ARRAY)[:-empty_traj_size]
# target_obs_array = reshape_obs_by_traj(target_obs_array, target_traj_len)
# # target_rewards_array = target._rewards[:-empty_traj_size]
# target_rewards_array = np.load(TARGET_REWARDS_ARRAY)[:-empty_traj_size]
# target_final_rewards_array = get_final_traj_rewards(target_rewards_array, target_traj_len)
# target_successful_obs_array = get_successful_obs_traj(target_obs_array, target_final_rewards_array)

# # Process FULL data
# # target_obs_array = target._obs['robot_state'][:-empty_traj_size]
# full_obs_array = np.load(FULL_STATE_ARRAY)[:-empty_traj_size]
# full_obs_array = reshape_obs_by_traj(full_obs_array, full_traj_len)
# # target_rewards_array = target._rewards[:-empty_traj_size]
# full_rewards_array = np.load(FULL_REWARDS_ARRAY)[:-empty_traj_size]
# full_final_rewards_array = get_final_traj_rewards(full_rewards_array, full_traj_len)
# full_successful_obs_array = get_successful_obs_traj(full_obs_array, full_final_rewards_array)

# print("full_obs_array.shape", full_obs_array.shape)
# full_first_lifted_pos = get_obs_first_reward(full_successful_obs_array)
# print("full_first_lifted_pos.shape", full_first_lifted_pos.shape)

# plot_all_trajs_and_dots(base_successful_obs_array, target_successful_obs_array, full_successful_obs_array, full_first_lifted_pos, num_to_plot=num_to_plot)

# base_avg_theta = get_avg_theta_traj(base_successful_obs_array)
# print("base_avg_theta", base_avg_theta)
# target_avg_theta = get_avg_theta_traj(target_successful_obs_array)
# print("target_avg_theta", target_avg_theta)
# full_avg_theta = get_avg_theta_traj(full_successful_obs_array)
# print("full_avg_theta", full_avg_theta)
# plot_avg_theta(base_avg_theta, target_avg_theta, full_avg_theta)

for i in range(base_traj_len - 1, np.load(BASE_REWARDS_ARRAY).shape[0], base_traj_len):
    if np.load(BASE_REWARDS_ARRAY)[i] >= 0.0:
        # base_final_states.append(base._obs['robot_state'][i][:3])
        img = base._obs['image'][i]
        img = process_image(img)
        plt.figure(num_succ)
        # plt.axis('off')
        save_file = osp.join(OUTPUT_DIR, 'base_{}.png'.format(num_succ))
        plt.imsave(save_file, img)
        target_starting_img.append(img)
        num_succ += 1
    if num_succ > num_to_plot:
        break
for i in range(num_to_plot):
    img = target._obs['image'][i * target_traj_len]
    img = process_image(img)
    # plt.axis('off')
    # plt.imshow(img)
    plt.figure(num_succ)
    save_file = osp.join(OUTPUT_DIR, 'target_{}.png'.format(i))
    plt.imsave(save_file, img)
    target_starting_img.append(img)
