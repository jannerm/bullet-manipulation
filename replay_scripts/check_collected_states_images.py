import roboverse
import numpy as np
# import skvideo.io
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

for i in range(len(data)):
    ax = plt.axes(projection='3d')
    traj = data[i]
    observations = np.array([traj["observations"][j]["robot_state"] for j in range(len(traj["observations"]))])
    next_observations = np.array([traj["next_observations"][j]["robot_state"] for j in range(len(traj["next_observations"]))])
    ax.plot3D(observations[:, 0], observations[:, 1], observations[:, 2], label="obs_{}".format(i))
    ax.plot3D(next_observations[:, 0], next_observations[:, 1], next_observations[:, 2], label="next_obs_{}".format(i))

    observations = traj["observations"]
    next_observations = traj["next_observations"]
    images = []
    for index in range(len(observations)):
        image = observations[index]["image"]
        images.append(Image.fromarray(np.uint8(image)))

    images_next = []
    for index in range(len(observations)):
        image = next_observations[index]["image"]
        images_next.append(Image.fromarray(np.uint8(image)))

    if not os.path.exists('test_collected_images_pad_action'):
        os.mkdir('test_collected_images_pad_action')

    images[0].save('{}/{}.gif'.format("test_collected_images_pad_action", i),
                format='GIF', append_images=images[1:],
                save_all=True, duration=100, loop=0)

    images_next[0].save('{}/{}_next.gif'.format("test_collected_images_pad_action", i),
                format='GIF', append_images=images_next[1:],
                save_all=True, duration=100, loop=0)

    plt.legend()
    fig.savefig('{}/{}.png'.format("test_collected_images_pad_action", i))
    fig.clf()