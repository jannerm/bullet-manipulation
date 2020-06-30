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
    image_array_by_grasp_and_time = image_array.reshape((num_grasps, traj_len, image_side, image_side, 3))
    return image_array_by_grasp_and_time
    
def get_avg_traj_actions(action_array, traj_len):
    num_grasps = action_array.shape[0] // traj_len
    action_array_by_time = action_array.reshape((num_grasps, traj_len, action_array.shape[1]))
    # print("first traj", action_array_by_time[0])
    # print("last traj", action_array_by_time[-1])
    avg_action_traj = np.mean(action_array_by_time, axis=0)
    # print("avg_action_traj", avg_action_traj)
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

def save_video_from_img_array(img_array_by_grasp_and_time, traj_len, num_rollouts_to_save):
    if not os.path.exists('data'):
        os.makedirs('data')
    for n in range(num_rollouts_to_save):
        images = [Image.fromarray(np.uint8(img_array_by_grasp_and_time[n][i] * 255)) for i in range(traj_len)]
        images[0].save('data/real_robot_rollout_{}.gif'.format(n),
            save_all=True, append_images=images[1:], duration=traj_len*2, loop=0)

if __name__ == "__main__":
    img_side = 48
    traj_len = 15
    num_rollouts_to_save = 1
    actions_data = load_data_by_key(pool_path, "actions")
    avg_action_traj = get_avg_traj_actions(actions_data, traj_len)

    num_plots_x, num_plots_y = (5, 1)
    plot_action_distribution(avg_action_traj, traj_len)

    obs_data = load_data_by_key(pool_path, "observations")
    image_array_by_grasp_and_time = load_images_into_array(obs_data, traj_len)
    save_video_from_img_array(image_array_by_grasp_and_time, traj_len, num_rollouts_to_save)
