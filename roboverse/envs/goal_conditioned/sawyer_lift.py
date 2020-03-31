import numpy as np
import gym
from gym.spaces import Dict

import roboverse.bullet as bullet
from roboverse.envs.sawyer_2d import Sawyer2dEnv

class SawyerLiftEnvGC(Sawyer2dEnv):

    X_POS = .75

    def __init__(self, *args, reset_obj_in_hand_rate=0.5,
                 goal_mode='obj_in_air', **kwargs):
        self.reset_obj_in_hand_rate = reset_obj_in_hand_rate
        self.goal_mode = goal_mode
        super().__init__(*args, env='SawyerLift2d-v0', **kwargs)
        self.record_args(locals())

    def reset(self):
        self._goal_pos = self.sample_goals(1)['state_desired_goal'][0]
        self._env._pos_init[:] = np.random.uniform(
            low=self._env._pos_low,
            high=self._env._pos_high)
        super().reset()
        self._env.open_gripper()

        ee_reset_pos = bullet.get_link_state(self._env._sawyer,
                                             self._env._end_effector, 'pos')
        cube_reset_pos = np.random.uniform(
            low=self._pos_low,
            high=self._pos_high)
        # By default, reset cube pos on ground.
        cube_reset_pos[-1] = -.3
        if np.random.random() > 1 - self.reset_obj_in_hand_rate:
            cube_reset_pos = ee_reset_pos
        bullet.set_body_state(self._objects['cube'], cube_reset_pos,
                              deg=[90,0,-90])

        obs = self.get_dict_observation()
        return obs

    def step(self, action, *args, **kwargs):
        new_action = action.copy()
        # Discretize gripper
        if new_action[3] > 0:
            new_action[3] = 10
        else:
            new_action[3] = -10
        obs, reward, done, info = super().step(new_action, *args, **kwargs)

        info = {
            **info,
            **self.get_info()
        }

        obs = self.get_dict_observation()
        reward = self.compute_reward(action, obs)
        return obs, reward, done, info

    def get_achieved_goal(self):
        return np.r_[
            self.get_2d_hand_pos(),
            self.get_2d_obj_pos(),
        ]

    def get_info_from_achieved_goal(self, achieved_goal):
        goal_info = self.get_info_from_achieved_goals(achieved_goal[None])

        goal_info_single = {}
        for k, v in goal_info.items():
            goal_info_single[k] = v[0]
        return goal_info_single

    def get_info_from_achieved_goals(self, achieved_goals):
        hand_pos = achieved_goals[:, :2]
        cube_pos = achieved_goals[:, 2:4]
        return {
            'hand_pos': hand_pos,
            'obj_pos': cube_pos,
        }

    def get_dict_observation(self):
        obs = self.get_observation()
        gripper = self.get_gripper_dist()
        obs = np.r_[obs, gripper]
        achieved_goal = self.get_achieved_goal()
        return {
            'observation': obs,
            'desired_goal': self._goal_pos,
            'achieved_goal': achieved_goal,
            'state_observation': obs,
            'state_desired_goal': self._goal_pos,
            'state_achieved_goal': achieved_goal,
        }

    def compute_reward(self, action, observation):
        actions = action[None]
        observations = {}
        for k, v in observation.items():
            observations[k] = v[None]
        return self.compute_rewards(actions, observations)[0]

    def compute_rewards(self, actions, observations):
        # Current computes reward based off only cube goal. Hand goal is ignored
        achieved_goal_info = self.get_info_from_achieved_goals(
            observations['state_achieved_goal'])
        ee_pos = achieved_goal_info['hand_pos']
        obj_pos = achieved_goal_info['obj_pos']
        ee_dist = bullet.l2_dist2d(obj_pos, ee_pos)
        goal_dist = bullet.l2_dist2d(
            obj_pos,
            np.tile(self._goal_pos[2:], (len(obj_pos), 1))
        )
        reward = -(ee_dist + self._goal_mult * goal_dist)
        reward = np.clip(reward, self._min_reward, 1e10)
        reward[goal_dist < 0.25] += self._bonus
        return reward

    def get_info(self):
        goal_info = self.get_info_from_achieved_goal(self._goal_pos)
        cube_pos = self.get_2d_obj_pos()
        cube_goal = goal_info['obj_pos']
        ee_pos = self.get_2d_hand_pos()

        ee_dist = bullet.l2_dist(cube_pos, ee_pos)
        obj_goal_dist = bullet.l2_dist(cube_pos, cube_goal)
        hand_goal_dist = bullet.l2_dist(ee_pos, cube_goal)

        return {
            'hand_dist': ee_dist,
            'obj_dist': obj_goal_dist,
            'hand_goal_dist': hand_goal_dist,
        }

    def get_image(self, width, height):
        img = self.render(mode='rgb_array', width=width, height=height)
        return img

    def get_goal(self):
        return self._goal_pos

    def get_env_state(self):
        return self.get_achieved_goal()

    def set_env_state(self, state):
        goal_info = self.get_info_from_achieved_goal(state)
        bullet.position_control(
            self._sawyer,
            self._end_effector,
            np.r_[SawyerLiftEnvGC.X_POS, goal_info['hand_pos']],
            self.theta,
        )
        self.open_gripper()
        bullet.set_body_state(
            self._objects['cube'],
            np.r_[SawyerLiftEnvGC.X_POS, goal_info['obj_pos']],
            deg=[90,0,-90],
        )

    def set_to_goal(self, goal):
        self.set_env_state(goal)

    def sample_goals(self, batch_size):
        if self.goal_mode == 'obj_in_air':
            obj_goals = np.random.uniform(
                low=self._env._pos_low,
                high=self._env._pos_high,
                size=(batch_size, len(self._env._pos_low)))
            # Make sure the objects is in the air
            obj_goals[:, -1] = obj_goals[:, -1].clip(0, 1e10)
        elif self.goal_mode == 'obj_in_bowl':
            obj_goals = np.tile(
                bullet.get_midpoint(self._objects['bowl']),
                (batch_size, 1))
        else:
            raise RuntimeError("Invalid goal mode: {}".format(self.goal_mode))

        # hand_goals = np.random.uniform(
            # low=self._env._pos_low,
            # high=self._env._pos_high,
            # size=(batch_size, len(self._env._pos_low)))
        hand_goals = obj_goals.copy()

        goals_2d = np.c_[
            hand_goals[:, 1:],
            obj_goals[:, 1:],
        ]
        return {
            'state_desired_goal': goals_2d,
            'desired_goal': goals_2d,
        }

    def get_gripper_dist(self):
        l_grip_id = bullet.get_index_by_attribute(
            self._env._sawyer, 'link_name', 'right_gripper_l_finger')
        r_grip_id = bullet.get_index_by_attribute(
            self._env._sawyer, 'link_name', 'right_gripper_r_finger')
        l_grip = np.array(
            bullet.get_link_state(self._env._sawyer, l_grip_id, 'pos'))
        r_grip = np.array(
            bullet.get_link_state(self._env._sawyer, r_grip_id, 'pos'))
        return (r_grip - l_grip)[1] * 10

    def get_2d_obj_pos(self):
        return bullet.get_midpoint(self._objects['cube'])[1:]

    def get_2d_hand_pos(self):
        return bullet.get_link_state(self._env._sawyer,
                                     self._env._end_effector, 'pos')[1:]

    def _set_spaces(self):
        obs = self.reset()
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        obs_bound = 100
        obs_high = np.ones(len(obs['observation'])) * obs_bound
        obs_space = gym.spaces.Box(-obs_high, obs_high)

        goal_high = np.ones(len(obs['state_achieved_goal'])) * obs_bound
        goal_space = gym.spaces.Box(-goal_high, goal_high)

        self.observation_space = Dict([
            ('observation', obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space),
        ])
