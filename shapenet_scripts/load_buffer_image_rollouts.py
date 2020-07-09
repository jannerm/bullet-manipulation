from railrl.data_management.replay_buffer import ReplayBuffer
import numpy as np
import pickle
from PIL import Image
import skimage.io as skii
import skimage.transform as skit
import os

# buffer_path = "/home/albert/dev/bullet-manipulation-avi-master/data/data_Widow200GraspV6BoxPlaceV0-v0_pixels_debug_1_sparse_reward_scripted_actions_fixed_position_noise_std_0.1/2020-06-22T10-42-24/2020-06-22T10-42-24_pool_31.pkl"

buffer_path = "/nfs/kun1/users/albert/batch_rl_datasets/july8_Widow200GraspV6BoxPlaceV0-v0_pixels_debug_40K_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/pool_pick.pkl"

def get_img_np_from_buffer(buffer_path, img_side):
    with open(buffer_path, 'rb') as f:
        rb = pickle.load(f)
        print("rb", rb)
        img_np = rb._obs['image'].np_array
        num_ts = img_np.shape[0]
        print("img_np.shape", img_np.shape)
        img_obs = img_np.reshape((num_ts, 3, img_side, img_side))
        return img_obs

def save_video_from_img_obs(img_obs, img_side, traj_len, num_rollouts_to_save):
    if not os.path.exists('data'):
        os.makedirs('data')
    for n in range(num_rollouts_to_save):
        images = []
        for i in range(traj_len):
            img_fi = np.transpose(img_obs[n * traj_len + i], (1, 2, 0))
            img_fi_resized = skit.resize(img_fi, (img_side * 3, img_side * 3, 3))
            images.append(Image.fromarray(np.uint8(img_fi_resized*255)))
        images[0].save('data/rollout_{}.gif'.format(n),
            save_all=True, append_images=images[1:], duration=traj_len*2, loop=0)

if __name__ == "__main__":
    img_side = 48
    traj_len = 15
    num_rollouts_to_save = 10
    img_obs = get_img_np_from_buffer(buffer_path, img_side)
    save_video_from_img_obs(img_obs, img_side, traj_len, num_rollouts_to_save)
