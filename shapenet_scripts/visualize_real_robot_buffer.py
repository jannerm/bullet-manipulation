from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
import gym_replab
import numpy as np
import pickle
from PIL import Image
import skimage.io as skii
import skimage.transform as skit
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pool_path = "/nfs/kun1/users/jonathan/data_v6/full_7k/combined_training_pool.pkl"

def load_data_by_key(pool_path, key):
    with open(pool_path, 'rb') as f:
        data = pickle.load(f)
        print("data.keys()", list(data.keys()))
    return data[key]

def load_images_into_array(obs_array, traj_len):
    print("obs_array.shape", obs_array.shape)
    image_array = np.array([obs_i_array['image'] for obs_i_array in np.squeeze(obs_array)])
    print("image_array.shape", image_array.shape)
    num_grasps = obs_array.shape[0] // traj_len
    image_side = int((image_array.shape[1] // 3) ** 0.5)
    image_array_by_grasp_and_time = image_array.reshape((num_grasps, traj_len, 3, image_side, image_side))
    return image_array_by_grasp_and_time

def load_goals_rewards_into_array(obs_array, action_array_by_grasp_time, traj_len):
    def get_grasp_location(state_array_traj, action_array_traj):
        # print("state_array_traj", state_array_traj)
        # print("action_array_traj", action_array_traj)
        assert len(state_array_traj.shape) == len(action_array_traj.shape) == 2
        grasp_idx = 4
        t_with_closed_gripper = np.argwhere(action_array_traj[:,grasp_idx] < 0).squeeze()
        # print("t_with_closed_gripper", t_with_closed_gripper)

        if len(t_with_closed_gripper.shape) == 0 or len(list(t_with_closed_gripper)) == 0:
            # gripper never closes
            return np.array([np.nan] * 3)

        grasp_exec_time = t_with_closed_gripper[0]
        return state_array_traj[grasp_exec_time][:3]

    state_array = np.array([obs_i_array['state'] for obs_i_array in np.squeeze(obs_array)])
    num_grasps = state_array.shape[0] // traj_len
    state_array_by_grasp_and_time = state_array.reshape((num_grasps, traj_len, state_array.shape[1]))
    # print("grasp location", get_grasp_location(state_array_by_grasp_and_time[1], action_array_by_grasp_time[1]))
    grasp_locations_array = np.array([get_grasp_location(state_array_by_grasp_and_time[i], action_array_by_grasp_time[i]) for i in range(num_grasps)])
    return grasp_locations_array

def load_rewards_into_array(rewards_array, traj_len):
    print("rewards_array.shape", rewards_array.shape)
    num_grasps = rewards_array.shape[0] // traj_len
    rewards_array_by_grasp_and_time = rewards_array.reshape((num_grasps, traj_len, rewards_array.shape[1]))
    print(np.sum(rewards_array_by_grasp_and_time[:,-1]))
    print("processed rewards array shape", rewards_array_by_grasp_and_time[:,-1].shape)
    return rewards_array_by_grasp_and_time[:,-1]

def load_actions_into_array(action_array, traj_len):
    num_grasps = action_array.shape[0] // traj_len
    action_array_by_grasp_time = action_array.reshape((num_grasps, traj_len, action_array.shape[1]))
    return action_array_by_grasp_time

def get_avg_traj_actions(action_array_by_grasp_time, traj_len):
    avg_action_traj = np.mean(action_array_by_grasp_time, axis=0)
    return avg_action_traj

def plot_subplot(
    title,
    xlabel,
    ylabel,
    xlist,
    ylist,
    index):
    plt.subplot(num_plots_y, num_plots_x, index + 1)
    plt.title(title, size=14)
    plt.xlabel(xlabel, size=12)
    plt.plot(xlist, ylist)

def plot_action_distribution(avg_action_traj, traj_len):
    # set figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 3
    plt.rcParams["figure.figsize"] = fig_size

    plt.figure(0)
    for dim in range(5):
        action_i_over_time = avg_action_traj[:,dim]
        timesteps = list(range(traj_len))
        plot_subplot("action[{}]".format(dim),
            "timestep", "", timesteps, action_i_over_time, dim)
    plt.tight_layout()
    plt.savefig("plot.png")

def get_grasp_success_fail_locations(grasp_locations_array, rewards_array):
    print("grasp_locations_array.shape", grasp_locations_array.shape)
    assert grasp_locations_array.shape[0] == rewards_array.shape[0]
    success_idx = np.argwhere(rewards_array == 1)[:,0]
    fail_idx = np.argwhere(rewards_array == 0)[:,0]
    successful_grasp_locations = grasp_locations_array[success_idx]
    failed_grasp_locations = grasp_locations_array[fail_idx]
    print("successful_grasp_locations.shape", successful_grasp_locations.shape)
    print("failed_grasp_locations.shape", failed_grasp_locations.shape)
    return successful_grasp_locations, failed_grasp_locations

def plot_grasp_locations_and_success(successful_grasp_locations, failed_grasp_locations):
    print("successful_grasp_locations.shape", successful_grasp_locations.shape)
    print("failed_grasp_locations.shape", failed_grasp_locations.shape)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.title("Grasp Attempts by Location and Success")
    scatter_labels = ["failure", "success"]
    f = ax.scatter(failed_grasp_locations[:,1], failed_grasp_locations[:,0], s=5, facecolors='none', edgecolors='red', label=scatter_labels[0])
    s = ax.scatter(successful_grasp_locations[:,1], successful_grasp_locations[:,0], s=5, facecolors='green', edgecolors='green', label=scatter_labels[1])
    fig.legend([f, s], labels=scatter_labels, loc='upper right', borderaxespad=0.1)
    plt.tight_layout()
    ax.invert_xaxis() # invert the x axis.
    plt.savefig("plot_success_fail.png")

    # plot success only:
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    plt.title("Successful Grasp Attempts")
    scatter_labels = ["failure", "success"]
    # f = ax.scatter(failed_grasp_locations[:,1], failed_grasp_locations[:,0], s=5, facecolors='none', edgecolors='red', label=scatter_labels[0])
    s = ax.scatter(successful_grasp_locations[:,1], successful_grasp_locations[:,0], s=5, facecolors='green', edgecolors='green', label=scatter_labels[1])
    # fig.legend([f, s], labels=scatter_labels, loc='upper right', borderaxespad=0.1)
    plt.tight_layout()
    ax.invert_xaxis() # invert the x axis.
    plt.savefig("plot_success_only.png")

def save_extremes_grasp_success_video(grasp_locs_data, rewards_last_ts_array, image_array_by_grasp_and_time):
    # Get index of leftmost and rightmost grasp. max(grasp_locs[:,1])
    grasp_locs_left_right_coord = np.where(rewards_last_ts_array.squeeze() == 1, grasp_locs_data[:,1], np.nan)
    print("grasp_locs_data[:,1].shape", grasp_locs_data[:,1].shape)
    print("rewards_last_ts_array.squeeze().shape", rewards_last_ts_array.squeeze().shape)
    print("grasp_locs_left_right_coord.shape", grasp_locs_left_right_coord.shape)
    # nan-ify all failed grasp locations so that we ignore them during the argmax.
    leftmost_grasp_idx = np.nanargmax(grasp_locs_left_right_coord)
    print("leftmost_grasp_loc", grasp_locs_data[leftmost_grasp_idx])
    rightmost_grasp_idx = np.nanargmin(grasp_locs_left_right_coord)
    print("rightmost_grasp_loc", grasp_locs_data[rightmost_grasp_idx])
    save_video(image_array_by_grasp_and_time[leftmost_grasp_idx], "leftmost_grasp_success")
    save_video(image_array_by_grasp_and_time[rightmost_grasp_idx], "rightmost_grasp_success")

    # get index of topmost and bottommost grasps.
    grasp_locs_top_bottom_coord = np.where(rewards_last_ts_array.squeeze() == 1, grasp_locs_data[:,0], np.nan)
    # nan-ify all failed grasp locations so that we ignore them during the argmax.
    topmost_grasp_idx = np.nanargmax(grasp_locs_top_bottom_coord)
    print("topmost_grasp_loc", grasp_locs_data[topmost_grasp_idx])
    bottommost_grasp_idx = np.nanargmin(grasp_locs_top_bottom_coord)
    print("bottommost_grasp_loc", grasp_locs_data[bottommost_grasp_idx])
    save_video(image_array_by_grasp_and_time[topmost_grasp_idx], "topmost_grasp_success")
    save_video(image_array_by_grasp_and_time[bottommost_grasp_idx], "bottommost_grasp_success")


def save_video(img_array, name):
    images = [Image.fromarray(np.uint8(np.transpose(img_array[i], (1, 2, 0)) * 255)) for i in range(traj_len)]
    images[0].save('data/{}.gif'.format(name),
        save_all=True, append_images=images[1:], duration=traj_len*2, loop=0)

def save_video_from_img_array(img_array_by_grasp_and_time, traj_len, num_rollouts_to_save):
    if not os.path.exists('data'):
        os.makedirs('data')
    for n in range(num_rollouts_to_save):
        # Save each 3 x 64 x 64 image array as 64 x 64 x 3 into a pillow image.
        save_video(img_array_by_grasp_and_time[n], "real_robot_rollout_{}".format(n))

if __name__ == "__main__":
    traj_len = 15
    num_rollouts_to_save = 1
    actions_data = load_data_by_key(pool_path, "actions")
    actions_array_by_grasp_time = load_actions_into_array(actions_data, traj_len)
    # avg_action_traj = get_avg_traj_actions(actions_array_by_grasp_time, traj_len)

    # num_plots_x, num_plots_y = (5, 1)
    # plot_action_distribution(avg_action_traj, traj_len)

    obs_data = load_data_by_key(pool_path, "observations")
    image_array_by_grasp_and_time = load_images_into_array(obs_data, traj_len)
    # save_video_from_img_array(image_array_by_grasp_and_time, traj_len, num_rollouts_to_save)

    grasp_locs_data = load_goals_rewards_into_array(obs_data, actions_array_by_grasp_time, traj_len)
    rewards_data = load_data_by_key(pool_path, "rewards")
    rewards_last_ts_array = load_rewards_into_array(rewards_data, traj_len)
    successful_grasp_locations, failed_grasp_locations = get_grasp_success_fail_locations(grasp_locs_data, rewards_last_ts_array)
    plot_grasp_locations_and_success(successful_grasp_locations, failed_grasp_locations)
    save_extremes_grasp_success_video(grasp_locs_data, rewards_last_ts_array, image_array_by_grasp_and_time)
