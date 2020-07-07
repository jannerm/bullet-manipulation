import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import os.path as osp
# FILE_PICK = '/media/avi/data/Work/data/2020-07-02T11-09-18_pool_50001_pick.pkl'
# FILE_PLACE = '/media/avi/data/Work/data/2020-07-02T16-59-49_pool_20001_place.pkl'
# FILE_PICK = '/media/avi/data/Work/github/jannerm/bullet-manipulation/data/july6_test_4_Widow200GraspV7BoxV0-v0_pixels_debug_50_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
# FILE_PLACE = '/media/avi/data/Work/github/jannerm/bullet-manipulation/data/july6_test_2_Widow200GraspV6BoxPlaceOnlyV0-v0_pixels_debug_50_sparse_reward_scripted_actions_fixed_position_noise_std_0.05/railrl_consolidated.pkl'
# OUTPUT_DIR = '/media/avi/data/Work/data/stitch_debug_v7_part2/'
FILE_PICK = '/nfs/kun1/users/avi/batch_rl_datasets/july6_Widow200GraspV7BoxV0-v0_pixels_debug_40K_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
FILE_PLACE = '/nfs/kun1/users/avi/batch_rl_datasets/july6_Widow200GraspV6BoxPlaceOnlyV0-v0_pixels_debug_40K_sparse_reward_scripted_actions_fixed_position_noise_std_0.2/railrl_consolidated.pkl'
OUTPUT_DIR = 'data'
if not osp.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
def process_image(image):
    image = np.reshape(image, (3, 48, 48))
    image = np.transpose(image, (1, 2, 0))
    return image
with open(FILE_PICK, 'rb') as f:
    pick = pickle.load(f)
with open(FILE_PLACE, 'rb') as f:
    place = pickle.load(f)
# points_to_plot = 100
#
# place_starting_states = []
# pick_final_states = []
# for i in range(points_to_plot):
#     place_starting_states.append(place._obs['robot_state'][i*10][:3])
#     num_succ = 0
#
# for i in range(pick._rewards.shape[0]):
#     if pick._rewards[i] > 0.0:
#         pick_final_states.append(pick._obs['robot_state'][i][:3])
#     if num_succ > points_to_plot:
#         break
#
# place_starting_states = np.asarray(place_starting_states)
# pick_final_states = np.asarray(pick_final_states)
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Horizontally stacked subplots')
#
# ax1.scatter(place_starting_states[:, 0], place_starting_states[:, 1], c='g')
# ax1.scatter(pick_final_states[:, 0], pick_final_states[:, 1], c='b', alpha=0.3)
# ax2.scatter(place_starting_states[:, 0], place_starting_states[:, 2], c='g')
# ax2.scatter(pick_final_states[:, 0], pick_final_states[:, 2], c='b', alpha=0.3)
# ax3.scatter(place_starting_states[:, 1], place_starting_states[:, 2], c='g')
# ax3.scatter(pick_final_states[:, 1], pick_final_states[:, 2], c='b', alpha=0.3)
# plt.show()
########
#IMAGES#
########
points_to_plot = 100
place_starting_img = []
pick_final_img = []
num_succ = 0
print("pick._rewards.shape", pick._rewards.shape)
print("place._rewards.shape", place._rewards.shape)
for i in range(0, pick._rewards.shape[0], 5):
    if pick._rewards[i] > 0.0:
        # pick_final_states.append(pick._obs['robot_state'][i][:3])
        img = pick._obs['image'][i]
        img = process_image(img)
        plt.axis('off')
        plt.imshow(img)
        save_file = osp.join(OUTPUT_DIR, 'pick_{}.png'.format(num_succ))
        plt.savefig(save_file)
        place_starting_img.append(img)
        num_succ += 1
    if num_succ > points_to_plot:
        break
for i in range(points_to_plot):
    img = place._obs['image'][i*10]
    img = process_image(img)
    plt.axis('off')
    plt.imshow(img)
    save_file = osp.join(OUTPUT_DIR, 'place_{}.png'.format(i))
    plt.savefig(save_file)
    place_starting_img.append(img)
# for i in range(points_to_plot):
#     img = pick._obs['image'][25*i + 24]
#     img = process_image(img)
#     plt.axis('off')
#     plt.imshow(img)
#     plt.savefig('/media/avi/data/Work/data/stitch_debug/pick_{}.png'.format(i))
#     place_starting_img.append(img)
# import IPython; IPython.embed()
