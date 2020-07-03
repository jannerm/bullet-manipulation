import numpy as np
import gym
from gym.spaces import Dict

import roboverse.bullet as bullet
import pybullet as p
from roboverse.envs.sawyer_2d import Sawyer2dEnv

class SawyerLiftEnvGC(Sawyer2dEnv):

    X_POS = .75
    # GROUND_Y = -0.355
    GROUND_Y = -0.325

    def __init__(
            self,
            *args,
            reset_obj_in_hand_rate=0.5,
            goal_sampling_mode='obj_in_air',
            reward_type='hand_dist+obj_dist',
            random_init_bowl_pos=False,
            sample_valid_rollout_goals=True,
            bowl_bounds=[-0.20, 0.20],
            **kwargs
    ):
        self.reset_obj_in_hand_rate = reset_obj_in_hand_rate
        self.goal_sampling_mode = goal_sampling_mode
        self.reward_type = reward_type
        self.random_init_bowl_pos = random_init_bowl_pos
        self.sample_valid_rollout_goals = sample_valid_rollout_goals
        self.bowl_bounds = bowl_bounds
        super().__init__(*args, env='SawyerLiftMulti-v0', **kwargs)
        self.record_args(locals())

    def sample_goal_for_rollout(self):
        goals_dict = self.sample_goals(
            batch_size=1,
        )
        goal = goals_dict['state_desired_goal'][0]

        if self.sample_valid_rollout_goals:
            self.set_env_state(goal)

            # Allow the objects to settle down after they are dropped in sim
            for _ in range(5):
                self._env.step(np.array([0, 0, 0, 1]))

            goal = self.get_env_state()

        return goal

    def reset(self):
        ## set the box position
        self._bowl_pos = [.75, 0.0, -.3]
        if self.random_init_bowl_pos:
            self._bowl_pos[1] = np.random.uniform(
                low=self.bowl_bounds[0],
                high=self.bowl_bounds[1],
            )
        # self._env.set_bowl_pos(self._bowl_pos)

        self._goal_pos = self.sample_goal_for_rollout()

        self._env._pos_init[:] = np.random.uniform(
            low=self._env._pos_low,
            high=self._env._pos_high)
        super().reset()
        if not self._lite_reset:
            self._env.open_gripper()

        ee_reset_pos = bullet.get_link_state(self._env._sawyer,
                                             self._env._end_effector, 'pos')
        obj_id_to_put_in_hand = -1
        if (np.random.random() > 1 - self.reset_obj_in_hand_rate) and (self.num_obj > 0):
            obj_id_to_put_in_hand = np.random.choice(self.num_obj)

        for obj_id in range(self.num_obj):
            cube_reset_pos = np.random.uniform(
                low=self._pos_low,
                high=self._pos_high)
            # By default, reset cube pos on ground.
            cube_reset_pos[-1] = SawyerLiftEnvGC.GROUND_Y
            if obj_id == obj_id_to_put_in_hand:
                cube_reset_pos = ee_reset_pos
            bullet.set_body_state(self._objects[self.get_obj_name(obj_id)],
                                  cube_reset_pos, deg=[90,0,-90])
        self.set_bowl_position(self._bowl_pos)

        # Allow the objects to settle down after they are dropped in sim
        for _ in range(5):
            self._env.step(np.array([0, 0, 0, 1]))

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
        obj_achieved_goals = np.r_[[
            self.get_2d_obj_pos(obj_id) for obj_id in range(self.num_obj)]
        ].flatten()
        achieved_goal = np.r_[
            self.get_2d_hand_pos(),
            obj_achieved_goals,
        ]
        if self._sliding_bowl:
            achieved_goal = np.r_[
                achieved_goal,
                self.get_bowl_position()[1],
            ]
        return achieved_goal

    def get_info_from_achieved_goal(self, achieved_goal):
        goal_info = self.get_info_from_achieved_goals(achieved_goal[None])

        goal_info_single = {}
        for k, v in goal_info.items():
            goal_info_single[k] = v[0]
        return goal_info_single

    def get_info_from_achieved_goals(self, achieved_goals):
        hand_pos = achieved_goals[:, :2]
        info = {
            'hand_pos': hand_pos,
        }
        if self._sliding_bowl:
            assert achieved_goals.shape[1] == (1 + self.num_obj) * 2 + 1
        else:
            assert achieved_goals.shape[1] == (1 + self.num_obj) * 2
        for cube_id in range(self.num_obj):
            idx_start = (1 + cube_id) * 2
            idx_end = idx_start + 2
            info[self.get_obj_name(cube_id)] = achieved_goals[:, idx_start:idx_end]
        if self._sliding_bowl:
            info['bowl_pos'] = achieved_goals[:,-1]
        return info

    def get_dict_observation(self):
        obs = self.get_observation()
        bowl = self.get_bowl_position()[1]
        gripper = self.get_gripper_dist()
        obs = np.r_[obs, bowl, gripper]
        achieved_goal = self.get_achieved_goal()

        # clip the observations, in case objects fall off table
        obs = np.clip(obs, -1, 1)
        achieved_goal = np.clip(achieved_goal, -1, 1)

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

        desired_goal_info = self.get_info_from_achieved_goals(
            observations['state_desired_goal'])

        # ee_dist = np.ones(len(ee_pos)) * 1e10
        #
        # total_goal_dist = 0
        # for obj_id in range(self.num_obj):
        #     obj_pos = achieved_goal_info[self.get_obj_name(obj_id)]
        #     obj_desired_goal = desired_goal_info[self.get_obj_name(obj_id)]
        #
        #     goal_dist = bullet.l2_dist2d(
        #         obj_pos,
        #         obj_desired_goal
        #     )
        #     total_goal_dist += goal_dist
        #     ee_dist = np.clip(bullet.l2_dist2d(obj_pos, ee_pos), -1e10, ee_dist)
        # reward = -(ee_dist * self._goal_mult + total_goal_dist)
        # reward = np.clip(reward, self.num_obj * self._min_reward, 1e10)
        # reward[total_goal_dist < 0.25 * self.num_obj] += self._bonus
        # return reward

        rewards = self.reward_type.split('+')
        dist = np.zeros(len(ee_pos))

        ee_desired_goal = desired_goal_info['hand_pos']
        ee_dist = bullet.l2_dist2d(
            ee_pos,
            ee_desired_goal
        )

        if 'hand_dist' in rewards:
            dist += ee_dist

        for obj_id in range(self.num_obj):
            obj_pos = achieved_goal_info[self.get_obj_name(obj_id)]
            obj_desired_goal = desired_goal_info[self.get_obj_name(obj_id)]

            obj_dist = bullet.l2_dist2d(
                obj_pos,
                obj_desired_goal
            )
            if 'obj_dist' in rewards:
                dist += obj_dist
        if self._sliding_bowl and 'obj_dist' in rewards:
            dist += np.abs(achieved_goal_info['bowl_pos'] - desired_goal_info['bowl_pos'])

        reward = -dist
        return reward

    def get_info(self):
        desired_goal_info = self.get_info_from_achieved_goal(self._goal_pos)
        achieved_goal_info = self.get_info_from_achieved_goal(
            self.get_achieved_goal())
        ee_pos = achieved_goal_info['hand_pos']

        # info = {}
        # hand_dist = 1e10
        # for obj_id in range(self.num_obj):
        #     cube_pos = achieved_goal_info[self.get_obj_name(obj_id)]
        #     cube_goal = desired_goal_info[self.get_obj_name(obj_id)]
        #
        #     obj_goal_dist = bullet.l2_dist(cube_pos, cube_goal)
        #     hand_dist = min(hand_dist, bullet.l2_dist(cube_pos, ee_pos))
        #     info['{}_dist'.format(self.get_obj_name(obj_id))] = obj_goal_dist
        # info['hand_dist'] = hand_dist
        # return info

        info = {}
        ee_goal = desired_goal_info['hand_pos']
        info['hand_dist'] = bullet.l2_dist(ee_pos, ee_goal)
        for obj_id in range(self.num_obj):
            cube_pos = achieved_goal_info[self.get_obj_name(obj_id)]
            cube_goal = desired_goal_info[self.get_obj_name(obj_id)]

            obj_goal_dist = bullet.l2_dist(cube_pos, cube_goal)
            info['{}_dist'.format(self.get_obj_name(obj_id))] = obj_goal_dist
            info['{}_success'.format(self.get_obj_name(obj_id))] = \
                float(np.abs(cube_pos[0] - self._bowl_pos[1]) <= 0.09)
        if self._sliding_bowl:
            info['bowl_dist'] = np.abs(achieved_goal_info['bowl_pos'] - desired_goal_info['bowl_pos'])
        return info

    def get_image(self, width, height):
        img = self.render(mode='rgb_array', width=width, height=height)
        return img

    def get_goal(self):
        return {
            'state_desired_goal': self._goal_pos,
            'desired_goal': self._goal_pos
        }

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
        for obj_id in range(self.num_obj):
            bullet.set_body_state(
                self._objects[self.get_obj_name(obj_id)],
                np.r_[SawyerLiftEnvGC.X_POS,
                      goal_info[self.get_obj_name(obj_id)]],
                deg=[90,0,-90],
            )
        if self._sliding_bowl:
            self.set_bowl_position([0.75, goal_info['bowl_pos'], -0.3])

    def set_to_goal(self, goal):
        self.set_env_state(goal['state_desired_goal'])

    def sample_goals(self, batch_size, goal_sampling_mode=None):
        if goal_sampling_mode is None:
            goal_sampling_mode = self.goal_sampling_mode

        low = self._env._pos_low[1:]
        high = self._env._pos_high[1:]

        if self._sliding_bowl:
            bowl_goals = np.random.uniform(
                low=self.bowl_bounds[0],
                high=self.bowl_bounds[1],
                size=(batch_size, 1))
        else:
            bowl_goals = np.random.uniform(
                low=self._bowl_pos[1],
                high=self._bowl_pos[1],
                size=(batch_size, 1))

        if goal_sampling_mode == 'uniform':
            obj_goals = np.random.uniform(
                low=low,
                high=high,
                size=(batch_size * self.num_obj, len(low))
            ).reshape((batch_size, -1))
        elif goal_sampling_mode == 'ground':
            high_ground = high.copy()
            high_ground[1] = low[1] + 0.03
            obj_goals = np.random.uniform(
                low=low,
                high=high_ground,
                size=(batch_size * self.num_obj, len(low))
            ).reshape((batch_size, -1))
        elif goal_sampling_mode == 'obj_in_air':
            high_ground = high.copy()
            high_ground[1] = low[1]
            # Have them all start off on the ground
            obj_goals = np.random.uniform(
                low=low,
                high=high_ground,
                size=(1 * self.num_obj, len(low))
            ).reshape((1, -1))

            # Choose which ones should be in the air
            is_air = (
                np.random.random(1) > .5
            )
            num_in_air = is_air.sum()
            obj_id_in_air = np.random.choice(self.num_obj, num_in_air)
            y_idx = (len(low) - 1)
            obj_goal_in_air_idx = (obj_id_in_air) * len(low) + y_idx
            obj_goals[is_air, obj_goal_in_air_idx] = np.random.uniform(
                low=low[1],
                high=high[1],
                size=num_in_air)
            # Make sure the objects is in the air
            # obj_goals[:, -1] = obj_goals[:, -1].clip(0, 1e10)
        elif goal_sampling_mode == 'obj_in_bowl':
            # obj_goals = np.tile(
            #     # bullet.get_midpoint(self._objects['bowl'])[1:],
            #     self._bowl_pos[1:],
            #     (batch_size, self.num_obj))
            obj_goals = np.c_[
                bowl_goals,
                self._bowl_pos[2] * np.ones(batch_size),
            ]
            obj_goals = np.tile(
                obj_goals,
                (1, self.num_obj)
            )
        else:
            raise RuntimeError("Invalid goal mode: {}".format(self.goal_sampling_mode))

        hand_goals = np.random.uniform(
            low=low,
            high=high,
            size=(batch_size, len(low)))

        goals_2d = np.c_[
            hand_goals,
            obj_goals,
        ]
        if self._sliding_bowl:
            goals_2d = np.c_[
                goals_2d,
                bowl_goals,
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
        if self._use_rotated_gripper:
            result = (r_grip - l_grip)[0] * 10
        else:
            result = (r_grip - l_grip)[1] * 10
        return result

    def get_2d_obj_pos(self, obj_id):
        return bullet.get_midpoint(self._objects[self.get_obj_name(obj_id)])[1:]

    def get_2d_hand_pos(self):
        return bullet.get_link_state(self._env._sawyer,
                                     self._env._end_effector, 'pos')[1:]

    def _set_spaces(self):
        obs = self.reset()
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        obs_bound = 1
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

    def get_contextual_diagnostics(self, paths, contexts):
        diagnostics = {}
        return diagnostics

    def goal_conditioned_diagnostics(self, paths, contexts):
        diagnostics = {}
        return diagnostics
