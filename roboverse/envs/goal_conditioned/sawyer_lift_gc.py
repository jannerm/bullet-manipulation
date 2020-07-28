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
            random_init_bowl_pos=False,
            sample_valid_rollout_goals=True,
            bowl_bounds=[-0.40, 0.40],
            hand_reward=True,
            gripper_reward=True,
            bowl_reward=True,
            reward_type=None,
            objs_to_reset_outside_bowl=[],
            obj_success_threshold=0.10,
            **kwargs
    ):
        self.reset_obj_in_hand_rate = reset_obj_in_hand_rate
        self.goal_sampling_mode = goal_sampling_mode
        self.random_init_bowl_pos = random_init_bowl_pos
        self.sample_valid_rollout_goals = sample_valid_rollout_goals
        self.bowl_bounds = bowl_bounds
        self.hand_reward = hand_reward
        self.gripper_reward = gripper_reward
        self.bowl_reward = bowl_reward
        self.reward_type = reward_type
        self.objs_to_reset_outside_bowl = objs_to_reset_outside_bowl
        self.obj_success_threshold = obj_success_threshold
        super().__init__(*args, env='SawyerLiftMulti-v0', **kwargs)
        self.record_args(locals())

    def reset(self):
        ## set the box position
        self._bowl_pos = [.75, 0.0, -.3]
        if self.random_init_bowl_pos:
            self._bowl_pos[1] = np.random.uniform(
                low=self.bowl_bounds[0],
                high=self.bowl_bounds[1],
            )

        self._goal_pos = self.sample_goals(1)['state_desired_goal'][0]

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
            if obj_id in self.objs_to_reset_outside_bowl:
                bowl_xpos = self._bowl_pos[1]
                bowl_xrange = [bowl_xpos - 0.15, bowl_xpos + 0.15]
                while True:
                    cube_reset_pos = np.random.uniform(
                        low=self._pos_low,
                        high=self._pos_high)
                    if cube_reset_pos[1] <= bowl_xrange[0] or cube_reset_pos[1] >= bowl_xrange[1]:
                        break
            else:
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

    def get_full_observation(self):
        obs = self.get_observation()
        bowl = self.get_bowl_position()[1]
        gripper = self.get_gripper_dist()
        obs = np.r_[obs, bowl, gripper]
        return obs

    def get_achieved_goal(self):
        return self.get_full_observation()

    def get_info_from_achieved_goal(self, achieved_goal):
        goal_info = self.get_info_from_achieved_goals(achieved_goal[None])

        goal_info_single = {}
        for k, v in goal_info.items():
            goal_info_single[k] = v[0]
        return goal_info_single

    def get_info_from_achieved_goals(self, achieved_goals):
        assert achieved_goals.shape[1] == (1 + self.num_obj) * 2 + 1 + 1
        info = {
            'hand_pos': achieved_goals[:, :2],
            'bowl_pos': achieved_goals[:, -2],
            'gripper_size': achieved_goals[:, -1],
        }
        for cube_id in range(self.num_obj):
            idx_start = (1 + cube_id) * 2
            idx_end = idx_start + 2
            info[self.get_obj_name(cube_id)] = achieved_goals[:, idx_start:idx_end]
        return info

    def get_dict_observation(self):
        obs = self.get_full_observation()
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
        assert self.reward_type in ['bowl_cube0_dist', None]
        if self.reward_type == 'bowl_cube0_dist':
            states = observations['state_achieved_goal']
            state_dim = states.shape[1]
            x_dist_sq = (states[:, 2] - states[:,state_dim - 2]) ** 2
            y_dist_sq = (states[:, 3] - (-0.35)) ** 2
            dist = np.sqrt(x_dist_sq + y_dist_sq)

            reward = -dist
            return reward


        achieved_goal = observations['state_achieved_goal']
        desired_goal = observations['state_desired_goal']

        batch_size, goal_dim = achieved_goal.shape

        dims_to_use = np.arange(2, (1+self.num_obj)*2)
        if self.hand_reward:
            dims_to_use = np.hstack((dims_to_use, np.arange(0, 2)))
        if self.bowl_reward:
            dims_to_use = np.hstack((dims_to_use, [goal_dim - 2]))
        if self.gripper_reward:
            dims_to_use = np.hstack((dims_to_use, [goal_dim - 1]))

        mask = np.zeros(goal_dim)
        mask[dims_to_use] = 1.0
        dist = np.linalg.norm((desired_goal - achieved_goal) * mask, axis=-1)

        reward = -dist
        return reward

    def get_info(self, achieved_goal=None, desired_goal=None):
        if achieved_goal is None:
            achieved_goal = self.get_achieved_goal()
        if desired_goal is None:
            desired_goal = self._goal_pos
        desired_goal_info = self.get_info_from_achieved_goal(desired_goal)
        achieved_goal_info = self.get_info_from_achieved_goal(achieved_goal)

        info = {}

        num_obj_success = 0
        num_bowl_obj_success = 0
        for obj_id in range(self.num_obj):
            obj_name = self.get_obj_name(obj_id)
            cube_pos = achieved_goal_info[obj_name]
            cube_goal = desired_goal_info[obj_name]

            obj_goal_dist = bullet.l2_dist(cube_pos, cube_goal)
            info['{}_dist'.format(obj_name)] = obj_goal_dist
            obj_success = float(obj_goal_dist <= self.obj_success_threshold) # 0.09
            info['{}_success'.format(obj_name)] = obj_success
            num_obj_success += obj_success

            obj_bowl_dist = np.abs(achieved_goal_info['bowl_pos'] - cube_pos[0])
            info['bowl_{}_dist'.format(obj_name)] = obj_bowl_dist
            bowl_obj_success = float(obj_bowl_dist <= 0.10)
            info['bowl_{}_success'.format(obj_name)] = bowl_obj_success
            num_bowl_obj_success += bowl_obj_success
        info['num_obj_success'] = num_obj_success
        info['num_bowl_obj_success'] = num_bowl_obj_success

        info['hand_dist'] = bullet.l2_dist(achieved_goal_info['hand_pos'], desired_goal_info['hand_pos'])
        info['bowl_dist'] = np.abs(achieved_goal_info['bowl_pos'] - desired_goal_info['bowl_pos'])
        info['gripper_dist'] = np.abs(achieved_goal_info['gripper_size'] - desired_goal_info['gripper_size'])

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
        self.set_bowl_position([0.75, goal_info['bowl_pos'], -0.3])
        ### disregard the gripper openning position ###

    def set_to_goal(self, goal):
        self.set_env_state(goal['state_desired_goal'])

    def sample_goals(self, batch_size, goal_sampling_mode=None):
        if goal_sampling_mode is None:
            goal_sampling_mode = self.goal_sampling_mode

        assert goal_sampling_mode in [
            'uniform',
            'ground',
            'obj_in_bowl',
            '50p_ground__50p_obj_in_bowl',
            'first_obj_in_bowl_oracle',
            'ground_away_from_curr_state',
        ]

        if goal_sampling_mode == '50p_ground__50p_obj_in_bowl':
            if batch_size == 1:
                if np.random.uniform() <= 0.5:
                    return self.sample_goals(1, goal_sampling_mode='ground')
                else:
                    return self.sample_goals(1, goal_sampling_mode='obj_in_bowl')
            else:
                num_ground_goals = batch_size // 2
                num_obj_in_bowl_goals = batch_size - num_ground_goals

                ground_goals = self.sample_goals(
                    num_ground_goals,
                    goal_sampling_mode='ground'
                )
                obj_in_bowl_goals = self.sample_goals(
                    num_obj_in_bowl_goals,
                    goal_sampling_mode='obj_in_bowl'
                )

                goals = {}
                for key in ground_goals.keys():
                    goals[key] = np.vstack((
                        ground_goals[key],
                        obj_in_bowl_goals[key]
                    ))
                return goals

        low = self._env._pos_low[1:]
        high = self._env._pos_high[1:]

        # hand goals
        hand_goals = np.random.uniform(
            low=low,
            high=high,
            size=(batch_size, len(low)))

        if goal_sampling_mode == 'first_obj_in_bowl_oracle':
            curr_state = self.get_achieved_goal()
            bowl_goals = np.ones((batch_size, 1)) * curr_state[-2]
        elif self.random_init_bowl_pos or self._bowl_type != 'fixed':
            bowl_goals = np.random.uniform(
                low=self.bowl_bounds[0],
                high=self.bowl_bounds[1],
                size=(batch_size, 1))
        else:
            bowl_goals = np.zeros((batch_size, 1))

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
        elif goal_sampling_mode == 'ground_away_from_curr_state':
            high_ground = high.copy()
            high_ground[1] = low[1] + 0.03
            curr_state = self.get_achieved_goal()
            obj_goal = []
            for obj_id in range(self.num_obj):
                obj_xpos = curr_state[2+obj_id*2]
                obj_xrange = [obj_xpos - self.obj_success_threshold, obj_xpos + self.obj_success_threshold]
                while True:
                    sampled_pos = np.random.uniform(
                        low=low,
                        high=high_ground)
                    if sampled_pos[0] <= obj_xrange[0] or sampled_pos[0] >= obj_xrange[1]:
                        break
                obj_goal += list(sampled_pos)

            obj_goals = np.tile(
                np.array(obj_goal).reshape((1, -1)),
                (batch_size, 1)
            )
        elif goal_sampling_mode == 'obj_in_bowl':
            obj_goals = np.tile(
                np.c_[
                    bowl_goals,
                    self._bowl_pos[2] * np.ones(batch_size),
                ],
                (1, self.num_obj)
            )
        elif goal_sampling_mode == 'first_obj_in_bowl_oracle':
            curr_state = self.get_achieved_goal()
            obj_goals = np.tile(
                curr_state[2:(self.num_obj+1)*2],
                (batch_size, 1)
            )
            obj_goals[:,0] = curr_state[-2]
        else:
            raise RuntimeError("Invalid goal mode: {}".format(self.goal_sampling_mode))

        gripper_goals = np.random.uniform(
            low=-0.5,
            high=0.5,
            size=(batch_size, 1))

        goals_2d = np.c_[
            hand_goals,
            obj_goals,
            bowl_goals,
            gripper_goals,
        ]

        if batch_size == 1 and self.sample_valid_rollout_goals:
            curr_state = self.get_achieved_goal()
            self.set_env_state(goals_2d[0])

            # Allow the objects to settle down after they are dropped in sim
            for _ in range(5):
                self._env.step(np.array([0, 0, 0, 1]))

            goals_2d = self.get_env_state()[None]
            self.set_env_state(curr_state)

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
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        pos_low = self._env._pos_low[1:]
        pos_high = self._env._pos_high[1:]

        if (self._bowl_type == 'fixed') and (not self.random_init_bowl_pos):
            bowl_low, bowl_high = 0, 0
        else:
            bowl_low, bowl_high = self.bowl_bounds

        obs_low = np.hstack((
            np.tile(pos_low, self.num_obj+1), # hand and blocks
            bowl_low, # bowl
            -0.53 # gripper
        ))
        obs_high = np.hstack((
            np.tile(pos_high, self.num_obj+1), # hand and blocks
            bowl_high, # bowl
            0.22 # gripper
        ))

        obs_space = gym.spaces.Box(obs_low, obs_high)
        goal_space = gym.spaces.Box(obs_low, obs_high)

        self.observation_space = Dict([
            ('observation', obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space),
        ])

    def goal_conditioned_diagnostics(self, paths, contexts):
        from collections import OrderedDict, defaultdict
        from multiworld.envs.env_util import create_stats_ordered_dict

        statistics = OrderedDict()
        stat_to_lists = defaultdict(list)

        for path, desired_goal in zip(paths, contexts):
            achieved_goals = path['observations']

            path_infos = []
            for achieved_goal in achieved_goals:
                info = self.get_info(achieved_goal, desired_goal)
                path_infos.append(info)

            for k in path_infos[0].keys():
                stat_to_lists[k].append([info[k] for info in path_infos])

        for stat_name, stat_list in stat_to_lists.items():
            statistics.update(create_stats_ordered_dict(
                'env_infos/{}'.format(stat_name),
                stat_list,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'env_infos/final/{}'.format(stat_name),
                [s[-1:] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'env_infos/initial/{}'.format(stat_name),
                [s[:1] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        return statistics
