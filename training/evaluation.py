from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    episode_keys = ['return', 'length']

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in episode_keys:
            stats[k].append(info['episode'][k])

        for k in info.keys():
            if 'success' in k:
                if k not in stats:
                    stats[k] = []
                stats[k].append(float(info[k]))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
