import numpy as np
import gym
from types import MethodType
import pdb

import roboverse.bullet as bullet
from roboverse.envs.sawyer_lift import SawyerLiftEnv
from roboverse.envs.sawyer_2d import Sawyer2dEnv
from collections import OrderedDict

from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)


from gym.spaces import Box, Dict

class SawyerLiftEnvGC(Sawyer2dEnv):

    def __init__(self, *args, goal_pos=[.75, -.4, .2],
                 reset_obj_in_hand_rate=0.5, **kwargs):
        self._goal_pos = goal_pos
        self.hand_and_obj_goal = np.tile(goal_pos[1:], (2))
        self.reset_obj_in_hand_rate = reset_obj_in_hand_rate
        super().__init__(*args, env='SawyerLift2d-v0', goal_pos=goal_pos, **kwargs)
        self.record_args(locals())

    def get_info(self):
        cube_pos = bullet.get_midpoint(self._objects['cube'])
        ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        ee_dist = bullet.l2_dist(cube_pos[1:], ee_pos[1:])
        goal_dist = bullet.l2_dist(cube_pos[1:], self._goal_pos[1:])
        hand_goal_dist = bullet.l2_dist(ee_pos[1:], self._goal_pos[1:])

        return {
            'hand_dist': ee_dist,
            'obj_dist': goal_dist,
            'hand_goal_dist': hand_goal_dist,
        }

    def reset(self):
        self._env._pos_init[:] = np.random.uniform(
            low=self._env._pos_low,
            high=self._env._pos_high)
        super().reset()
        self._env.open_gripper()

        ee_pos = bullet.get_link_state(self._env._sawyer, self._env._end_effector, 'pos')
        cube_pos = np.random.uniform(
            low=self._pos_low,
            high=self._pos_high)
        cube_pos[-1] = -.3
        if np.random.random() > 1 - self.reset_obj_in_hand_rate:
            ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
            cube_pos = ee_pos
        bullet.set_body_state(self._objects['cube'],
                              cube_pos, deg=[90,0,-90])

        obs = self.get_dict_observation()
        return obs

    def step(self, action, *args, **kwargs):
        action[3] = 0.3
        if action[3] > 0:
            action[3] = 10
        else:
            action[3] = -10
        obs, reward, done, info = super().step(action, *args, **kwargs)
        # cube_pos = np.random.uniform(
            # low=self._pos_low,
            # high=self._pos_high)
        # ee_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        # cube_pos = ee_pos
        # bullet.set_body_state(self._objects['cube'],
                              # cube_pos, deg=[90,0,-90])


        info = {
            **info,
            **self.get_info()
        }

        obs = self.get_dict_observation()
        reward = self.compute_reward(action, obs)
        # print(obs['state_observation'][:2], reward, self.is_eval)

        return obs, reward, done, info

    def ee_pos_from_obs(self, obs):
        return obs[:2]

    def cube_pos_from_obs(self, obs):
        return obs[2:4]

    def achieved_goal_from_obs(self, obs):
        return np.concatenate(
            (self.ee_pos_from_obs(obs),
             self.cube_pos_from_obs(obs))
        )

    def get_dict_observation(self):
        obs = self.get_observation()
        gripper = self.get_gripper_dist()
        obs = np.r_[obs, gripper]
        achieved_goal = self.achieved_goal_from_obs(obs)

        cube_pos = bullet.get_midpoint(self._objects['cube'])[1:]
        return {
            'observation': obs,
            'desired_goal': self.hand_and_obj_goal,
            'achieved_goal': achieved_goal,
            'state_observation': obs,
            'state_desired_goal': self.hand_and_obj_goal,
            'state_achieved_goal': achieved_goal,
        }

    def compute_reward(self, action, observation):
        actions = action[None]
        observations = {}
        for k, v in observation.items():
            observations[k] = v[None]
        return self.compute_rewards(actions, observations)[0]

    def compute_rewards(self, actions, observations):
        ee_pos = observations['state_achieved_goal'][:, :2]
        cube_pos = observations['state_achieved_goal'][:, 2:]
        ee_dist = bullet.l2_dist2d(cube_pos, ee_pos)
        goal_dist = bullet.l2_dist2d(
            cube_pos,
            np.tile(self._goal_pos[1:], (len(cube_pos), 1))
        )
        reward = -(ee_dist + self._goal_mult * goal_dist)
        reward = np.clip(reward, self._min_reward, 1e10)
        reward[goal_dist < 0.25] += self._bonus
        return reward

    def _set_spaces(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        obs = self.reset()
        observation_dim = len(obs['observation'])
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        obs_space = gym.spaces.Box(-obs_high, obs_high)

        goal_high = np.ones(len(obs['state_achieved_goal'])) * obs_bound
        goal_space = gym.spaces.Box(-goal_high, goal_high)
        # goal_space = obs_space

        self.observation_space = Dict([
            ('observation', obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space),
        ])

    def get_image(self, width, height):
        img = self.render(mode='rgb_array', width=width, height=height)
        return img

    def get_goal(self):
        return self._goal_pos

    def get_env_state(self):
        return

    def set_env_state(self, state):
        return

    def set_to_goal(self, goal):
        return

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

    def sample_goals(self, batch_size):
        goals = np.tile(self.hand_and_obj_goal, (batch_size, 1))
        return {
            'state_desired_goal' : goals,
            'desired_goal' : goals,
        }

    def get_gripper_dist(self):
        l_grip_id = bullet.get_index_by_attribute(self._env._sawyer, 'link_name',
                                               'right_gripper_l_finger')
        r_grip_id = bullet.get_index_by_attribute(self._env._sawyer, 'link_name',
                                               'right_gripper_r_finger')
        l_grip = np.array(bullet.get_link_state(
            self._env._sawyer, l_grip_id, 'pos'))
        r_grip = np.array(bullet.get_link_state(
            self._env._sawyer, r_grip_id, 'pos'))
        return (r_grip - l_grip)[1] * 10


