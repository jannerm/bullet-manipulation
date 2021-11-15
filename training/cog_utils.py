import os
from typing import Any, Tuple

import d4rl
import gym
import numpy as np
import roboverse

from pixel_iql import wrappers
from pixel_iql.common import Batch


def sample(dataset, batch_size):
    indx = np.random.randint(len(dataset['observations']), size=batch_size)
    return Batch(observations=dataset['observations'][indx],
                 actions=dataset['actions'][indx],
                 rewards=dataset['rewards'][indx],
                 masks=dataset['masks'][indx],
                 next_observations=dataset['next_observations'][indx])


def make_env_and_dataset(env_name: str, prior_buffer: str, task_buffer: str,
                         max_episode_steps: int,
                         seed: int) -> Tuple[gym.Env, Any]:
    env = roboverse.make(env_name, transpose_image=True, gui=False)
    env = wrappers.Roboverse(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    data_folder = os.path.join(d4rl.offline_env.DATASET_PATH, 'cog')

    prior_dataset = np.load(os.path.join(data_folder, f'{prior_buffer}.npy'),
                            allow_pickle=True)

    task_dataset = np.load(os.path.join(data_folder, f'{task_buffer}.npy'),
                           allow_pickle=True)

    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'masks': []
    }

    for ds in [prior_dataset, task_dataset]:
        for traj in ds:
            for k in traj.keys():
                for data in traj[k]:
                    if 'observation' in k:
                        dataset[k].append(data['image'])
                    elif 'terminals' in k:
                        dataset['masks'].append(1.0 - float(data))
                    elif k in ['rewards', 'actions']:
                        dataset[k].append(data)

    for k in dataset.keys():
        dataset[k] = np.stack(dataset[k])

    return env, dataset
