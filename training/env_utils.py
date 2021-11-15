import os
import pkgutil
from typing import Any, Tuple

import d4rl
import gym
import numpy as np
import pybullet
import roboverse as rv

from pixel_iql import wrappers
from pixel_iql.common import Batch


def sample(dataset, batch_size):
    indx = np.random.choice(dataset['indices'], size=batch_size)
    return Batch(observations=dataset['observations'][indx],
                 actions=dataset['actions'][indx],
                 rewards=dataset['rewards'][indx],
                 masks=dataset['masks'][indx],
                 next_observations=dataset['observations'][indx + 1])


def make_env_and_dataset(env_name: str,
                         seed: int,
                         max_episode_steps: int = 50) -> Tuple[gym.Env, Any]:
    env = rv.make('RemoveLid-v0', gui=False)

    egl = pkgutil.get_loader('eglRenderer')
    if (egl):
        pluginId = pybullet.loadPlugin(egl.get_filename(),
                                       "_eglRendererPlugin")
    else:
        pluginId = pybullet.loadPlugin("eglRendererPlugin")

    env = wrappers.Roboverse(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    data_file = os.path.join('dataset', 'traj_100', '0.npz')

    with open(data_file, 'rb') as f:
        dataset = dict(np.load(f))

    dataset['indices'] = np.arange(len(
        dataset['observations']))[np.logical_not(dataset['dummys'])]

    return env, dataset
