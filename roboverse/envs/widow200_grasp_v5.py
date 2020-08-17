from roboverse.envs.widow200_grasp_v2 import Widow200GraspV2Env
import roboverse.bullet as bullet
import numpy as np
import gym

REWARD_FAIL = 0.0
REWARD_SUCCESS = 1.0


class Widow200GraspV5Env(Widow200GraspV2Env):
    def __init__(self, *args, **kwargs):
        # Used for obs and railrl-private CNN forward.
        self.cnn_input_key = "image"
        self.fc_input_key = "robot_state"
        self.object_obs_key = "object_state"
        self.gripper_goal_location = np.asarray([0.81, -0.05, -0.20])
        super().__init__(*args, **kwargs)

    def _set_action_space(self):
        act_dim = 6
        # first three actions are delta x,y,z
        # action 4 is wrist rotation
        # action 5 is gripper open/close (> 0.5 for open, < -0.5 for close)
        # action 6 is terminate episode  (> 0.5 for termination)
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_spaces(self):
        self._set_action_space()
        # obs = self.reset()
        robot_obs_dim = 3 + 1 + 1
        obj_obs_dim = 7 * self._num_objects
        obs_bound = 100
        robot_obs_high = np.ones(robot_obs_dim) * obs_bound
        obj_obs_high = np.ones(obj_obs_dim) * obs_bound
        full_state_high = np.ones(obj_obs_dim + robot_obs_dim) * obs_bound
        robot_obs_space = gym.spaces.Box(-robot_obs_high, robot_obs_high)
        obj_obs_space = gym.spaces.Box(-obj_obs_high, obj_obs_high)
        if self._observation_mode == 'state':
            self.observation_space = gym.spaces.Box(
                -full_state_high, full_state_high)
        elif self._observation_mode == 'pixels' or self._observation_mode == 'pixels_debug':
            img_space = gym.spaces.Box(0, 1, (self.image_length,), dtype=np.float32)
            if self._observation_mode == 'pixels':
                spaces = {self.cnn_input_key: img_space, self.fc_input_key: robot_obs_space}
            elif self._observation_mode == 'pixels_debug':
                spaces = {
                    self.cnn_input_key: img_space,
                    self.fc_input_key: robot_obs_space,
                    self.object_obs_key: obj_obs_space
                }
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def is_object_grasped(self, object_name):
        """Returns true if any object is above reward height thresh."""
        is_grasped = False
        object_info = bullet.get_body_info(self._objects[object_name],
                                           quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])
        object_height = object_pos[2]
        if object_height > self._reward_height_thresh:
            end_effector_pos = np.asarray(self.get_end_effector_pos())
            object_gripper_distance = np.linalg.norm(
                object_pos - end_effector_pos)
            if object_gripper_distance < 0.1:
                is_grasped = True
        return is_grasped

    def get_reward(self, info):
        reward = 0.0
        if self.target_object is None:
            object_list = self._objects.keys()
            for object_name in object_list:
                if self.is_object_grasped(object_name):
                    reward = 1.0
        else:
            if self.is_object_grasped(self.target_object):
                reward = 1.0

        reward = self.adjust_rew_if_use_positive(reward)
        return reward

    def get_wrist_joint_angle(self):
        # Returns scalar corresponding to gripper wrist angle.
        joints, current = bullet.get_joint_positions(self._robot_id)
        return current[joints[4]]

    def get_observation(self):
        # gripper_tips_distance = self.get_gripper_tips_distance()
        gripper_open = np.array([float(self._gripper_open)])
        wrist_joint_angle = np.array(
            [self.get_wrist_joint_angle()]) # shape (1,) array
        end_effector_pos = self.get_end_effector_pos()
        # end_effector_theta = bullet.get_link_state(
        #     self._robot_id, self._end_effector, 'theta', quat_to_deg=False)

        if self._observation_mode == 'state':
            state_observation = np.concatenate(
                (end_effector_pos, wrist_joint_angle, gripper_open))
            object_observation = self.get_obj_obs_array()
            observation = np.concatenate(
                (state_observation, object_observation)
            )

        elif self._observation_mode == 'pixels':
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
            # image_observation = np.zeros((48, 48, 3), dtype=np.uint8)
            observation = {
                self.fc_input_key: np.concatenate(
                    (end_effector_pos, wrist_joint_angle, gripper_open)),
                self.cnn_input_key: image_observation
            }
        elif self._observation_mode == 'pixels_debug':
            # This mode passes in all the true state information + images
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
            state_observation = np.concatenate(
                (end_effector_pos, wrist_joint_angle, gripper_open))

            object_observation = self.get_obj_obs_array()

            observation = {
                self.fc_input_key: state_observation,
                self.object_obs_key: object_observation,
                self.cnn_input_key: image_observation,
            }
        else:
            raise NotImplementedError

        return observation

    def _gripper_simulate(self, pos, target_theta, delta_theta, gripper_action):
        # is_gripper_open = self._is_gripper_open()
        is_gripper_open = self._gripper_open
        if gripper_action > 0.5 and is_gripper_open:
            # keep it open
            gripper = -0.8
            self._simulate(pos, target_theta, gripper, delta_theta=delta_theta)
        elif gripper_action > 0.5 and not is_gripper_open:
            # gripper is currently closed and we want to open it
            gripper = -0.8
            for _ in range(5):
                self._simulate(pos, target_theta, gripper, delta_theta=0)
            self._gripper_open = True
        elif gripper_action < -0.5 and not is_gripper_open:
            # keep it closed
            gripper = 0.8
            self._simulate(pos, target_theta, gripper, delta_theta=delta_theta)
        elif gripper_action < -0.5 and is_gripper_open:
            # gripper is open and we want to close it
            gripper = +0.8
            for _ in range(5):
                self._simulate(pos, target_theta, gripper, delta_theta=0)
            # we will also lift the object up a little
            for _ in range(5):
                pos = list(self.gripper_goal_location)
                self._simulate(pos, target_theta, gripper, delta_theta=0)

            self._gripper_open = False
        elif gripper_action <= 0.5 and gripper_action >= -0.5:
            # maintain current status
            if is_gripper_open:
                gripper = -0.8
            else:
                gripper = 0.8
            self._simulate(pos, target_theta, gripper, delta_theta=delta_theta)
            pass
        else:
            raise NotImplementedError

    def step(self, action):
        action = np.asarray(action)
        pos = list(bullet.get_link_state(self._robot_id, self._end_effector, 'pos'))
        delta_pos = action[:3]
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        theta = list(bullet.get_link_state(self._robot_id, self._end_effector,
                                           'theta'))
        target_theta = theta
        delta_theta = action[3]
        target_theta = np.clip(target_theta, [0, 85, 137], [180, 85, 137])
        target_theta = bullet.deg_to_quat(target_theta)

        gripper_action = action[4]
        self._gripper_simulate(pos, target_theta, delta_theta, gripper_action)

        if action[5] > 0.5:
            done = True
            reward = float(self.is_object_grasped())
            reward = self.adjust_rew_if_use_positive(reward)
        else:
            done = False
            reward = self.adjust_rew_if_use_positive(REWARD_FAIL)
        info = {'grasp_success': float(self.is_object_grasped())}

        observation = self.get_observation()
        self._prev_pos = bullet.get_link_state(self._robot_id, self._end_effector,
                                               'pos')
        return observation, reward, done, info

    def reset(self):
        obs = super().reset()
        self._gripper_open = True
        return obs


class Widow200GraspV5RandObjEnv(Widow200GraspV5Env):
    def __init__(self,
                 *args,
                 scaling_local_list=[0.3]*10,
                 **kwargs):
        self.possible_objects = [
            'smushed_dumbbell',
            'jar',
            'beer_bottle',
            'mug',
            'square_prism_bin',
            'conic_bin',
            'ball',
            'shed',
            'sack_vase',
            'conic_cup'
        ]
        # chosen_object = np.random.choice(self.possible_objects)
        super().__init__(*args,
            object_names=self.possible_objects,
            scaling_local_list=scaling_local_list,
            **kwargs)

    def reset(self):
        """Currently only implemented for selecting 1 object at random"""
        self.object_names = list([np.random.choice(self.possible_objects)])
        print("self.object_names", self.object_names)
        return super().reset()


if __name__ == "__main__":
    import roboverse
    import time
    env = roboverse.make("Widow200GraspThreeV5-v0",
                         gui=True,
                         observation_mode='pixels_debug',)

    object_ind = 0
    EPSILON = 0.05
    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        for _ in range(env.scripted_traj_len):
            if isinstance(obs, dict):
                state_obs = obs[env.fc_input_key]
                obj_obs = obs[env.object_obs_key]

            ee_pos = state_obs[:3]
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.
            # theta_action = np.random.uniform()
            # print(object_gripper_dist)
            if object_gripper_dist > dist_thresh and env._gripper_open:
                # print('approaching')
                action = (object_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.02:
                    action[2] = 0.0
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif env._gripper_open:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            else:
                # print('terminating')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.7])))

            action[:3] += np.random.normal(scale=0.1, size=(3,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print(action)
            obs, rew, done, info = env.step(action)
            time.sleep(0.05)
            if done:
                print('reward: {}'.format(rew))
                break
