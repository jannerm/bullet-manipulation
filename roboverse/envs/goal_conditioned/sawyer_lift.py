import numpy as np
import gym
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_lift import SawyerLiftEnv
from collections import OrderedDict

from gym.spaces import Box, Dict

class SawyerLiftEnvGC(SawyerLiftEnv):

    def __init__(self, *args, goal_pos=[.75, -.4, .2], **kwargs):
        self._goal_pos = goal_pos
        self.record_args(locals())
        super().__init__(*args, goal_pos=goal_pos, **kwargs)

        self._objects = self._objects
        self._init_states = self._get_body_states()

    # See sawyer2dEnv::get_observation
    def get_flat_observation(self):
        ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        bodies = sorted([v for k, v in self._objects.items() if not bullet.has_fixed_root(v)])
        obj_pos = [bullet.get_body_info(body, 'pos') for body in bodies]

        ee_pos = np.array(ee_pos[1:])
        obj_pos = np.array([pos[1:] for pos in obj_pos])
        observation = np.concatenate((ee_pos, obj_pos.flatten()))
        return observation

    def get_observation(self):
        obs = self.get_flat_observation()
        goal = self._goal_pos

        cube_pos = bullet.get_midpoint(self._objects['cube'])
        return {
            'observation': obs,
            'desired_goal': goal,
            'achieved_goal': cube_pos,
            'state_observation': obs,
            'state_desired_goal': goal,
            'state_achieved_goal': cube_pos,
        }

    def sample_goals(self, batch_size):
        return np.tile(self._goal_pos, (batch_size, 1))

    def _set_spaces(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        obs = self.reset()
        observation_dim = len(obs)
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        obs_space = gym.spaces.Box(-obs_high, obs_high)

        goal_high = np.ones(3) * obs_bound
        goal_space = gym.spaces.Box(-goal_high, goal_high)

        self.observation_space = Dict([
            ('observation', obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space),
        ])

    def get_info(self):
        cube_pos = bullet.get_midpoint(self._objects['cube'])
        ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        ee_dist = bullet.l2_dist(cube_pos, ee_pos)
        goal_dist = bullet.l2_dist(cube_pos, self._goal_pos)

        return {
            'hand_dist': ee_dist,
            'obj_dist': goal_dist,
        }

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_dist',
            'obj_dist',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def sawyer_step(self, act, *args, **kwargs):
        act[0] = 0
        out = super().step(act, *args, **kwargs)

        current_states = self._get_body_states()
        for name, body in self._objects.items():
            current = current_states[name]
            init = self._init_states[name]

            x = init['pos'][0:1]
            yz = current['pos'][1:3]
            theta = current['theta']
            pos = x + yz
            bullet.set_body_state(body, pos, theta)

        obs = self.get_observation()
        rew = self.get_reward(obs['state_observation'])
        term = False
        info = {}
        return obs, rew, term, info

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.sawyer_step(*args, **kwargs)
        info = {
            **info,
            **self.get_info()
        }
        return obs, reward, done, info

    def _get_body_states(self):
        states = {}
        for name, body in self._objects.items():
            state = bullet.get_body_info(body, ['pos', 'theta'])
            states[name] = state
        return states

    def render(self, *args, **kwargs):
        return self.render(*args, **kwargs)

