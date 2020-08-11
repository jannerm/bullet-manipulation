import roboverse
import numpy as np
#import skvideo.io
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PIL import Image
import sys
import numpy as np
import pdb 
import pybullet as p
import roboverse
import roboverse.bullet as bullet
import math
from roboverse.bullet.misc import load_urdf, load_obj
import os

from tqdm import tqdm
from PIL import Image
import argparse
import time

fig = plt.figure()

data = np.load(sys.argv[1], allow_pickle=True)

success = []
failed_id = []

reward_type = "shaped"
env = roboverse.make('SawyerGraspOneV2-v0', reward_type=reward_type,
                     observation_mode="pixels_debug",
                     num_objects=3,
                     obs_img_dim=48,
                     randomize=False,
                     trimodal=True)

for i in tqdm(range(2)):
    ax = plt.axes(projection='3d')
    traj = data[i]
    observations = np.array([traj["observations"][j]["state"] for j in range(len(traj["observations"]))])
    ax.plot3D(observations[:, 0], observations[:, 1], observations[:, 2], label="original trajs", color="red")
        
    actions = traj["actions"]
    first_observation = traj["observations"][0]

    all_obj_pos = []
    for j in range(env._num_objects):
        obj_pos = first_observation["state"][-(j + 1) * 7:-(j + 1) * 7 + 3]
        all_obj_pos.append([obj_pos[0], obj_pos[1], obj_pos[2]])

    env._trimodal_positions = all_obj_pos
    env.reset()
    images = []
    
    replay_obs_list = []
    obs = env.get_observation()
    replay_obs_list.append(obs["state"])
    for index in range(len(actions)):
        data[i]["observations"][index] = obs
        a = actions[index]
        obs, rew, done, info = env.step(a)
        obs = env.get_observation()
        replay_obs_list.append(obs["state"])
        images.append(Image.fromarray(env.render_obs()))
    print(info["grasp_success"])
    success.append(info["grasp_success"])

    if not info["grasp_success"]:
        failed_id.append(i)

    replay_obs_list = np.array(replay_obs_list)
    ax.plot3D(replay_obs_list[:, 0], replay_obs_list[:, 1], replay_obs_list[:, 2], label="replay", color="blue")
    
    all_obj_pos = np.array(all_obj_pos)
    ax.scatter3D(all_obj_pos[:,0], all_obj_pos[:,1], all_obj_pos[:,2], label="actual position")
    replay_pos = np.array(env._trimodal_positions)
    ax.scatter3D(replay_pos[:,0], replay_pos[:,1], replay_pos[:,2], label="replayed position")

    if not os.path.exists('replay_videos_trimodal_random'):
        os.mkdir('replay_videos_trimodal_random')
    #video_save_path = os.path.join(".", "videos")

    images[0].save('{}/{}.gif'.format("replay_videos_trimodal_random", i),
                format='GIF', append_images=images[1:],
                save_all=True, duration=100, loop=0)
    plt.legend()
    fig.savefig('{}/{}.png'.format("replay_videos_trimodal_random", i))
    fig.clf()
    
success = np.array(success)
print("mean: ", success.mean())
print("std: ", success.std())
print("all success: ", (success == 1).all())
print("failed id: ")
print(failed_id)
#plt.show()

path = "{}_replayed.npy".format(sys.argv[1][:-4])
np.save(path, data)