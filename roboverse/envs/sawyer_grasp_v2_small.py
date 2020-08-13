import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.sawyer_base import SawyerBaseEnv
import gym
from roboverse.bullet.misc import load_obj
import os.path as osp
import pickle
import random

REWARD_NEGATIVE = -1.0
REWARD_POSITIVE = 30.0
SHAPENET_ASSET_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)), 'assets/ShapeNetCore')


def load_shapenet_object(object_path, scaling, object_position, scale_local=0.5):
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    obj = load_obj(
        SHAPENET_ASSET_PATH + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(
            dir_name, object_name),
        SHAPENET_ASSET_PATH + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
            dir_name, object_name),
        object_position,
        [0, 0, 1, 0],
        scale=scale_local*scaling[
            '{0}/{1}'.format(dir_name, object_name)])
    return obj

class SawyerGraspV2Env(SawyerBaseEnv):

    def __init__(self,
                 observation_mode='state',
                 num_objects=5,
                 obs_img_dim=48,
                 transpose_image=False,
                 invisible_robot=False,
                 reward_type="sparse",
                 randomize=True,
                 height_threshold=-0.32,
                 reward_height_thresh=-0.3,
                 object_position_low = (.60, .05, -.20),
                 object_position_high = (.80, .25, -.20),
                 object_ids = [0, 1, 25, 30, 50, 215, 255, 265, 300, 310],
                 fix_obj_position=None,
                 trimodal=True,
                 trimodal_positions=None,
                 normalize_and_flatten=False,
                 single_obj_reward=-1,
                 trimodal_positions_choice=0,
                 all_random=True,
                 trimodal_positions_two=None,
                 *args,
                 **kwargs
                 ):
        """
        :param observation_mode: state, pixels
        :param obs_img_dim:
        :param transpose_image:
        :param invisible_robot:
        """
        assert observation_mode != 'pixels', "Do not use pixels mode"
        self.reward_type = reward_type
        self._observation_mode = observation_mode
        self._num_objects = num_objects
        self.obs_img_dim = obs_img_dim
        self.image_shape = (obs_img_dim, obs_img_dim)
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self._height_threshold = height_threshold
        self._reward_height_thresh = reward_height_thresh
        self.randomize = randomize
        self._object_position_low = object_position_low
        self._object_position_high = object_position_high

        self._fix_obj_position = fix_obj_position
        self.trimodal = trimodal
        self._trimodal_positions = trimodal_positions
        self.normalize_and_flatten = normalize_and_flatten
        self._single_obj_reward = single_obj_reward
        self.trimodal_positions_choice = trimodal_positions_choice
        self.all_random = all_random
        self.trimodal_positions_two = trimodal_positions_two
        self.cnn_input_key = 'image'

        # TODO(avi) optimize the view matrix
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.75, +.15, -0.2], distance=0.3,
            yaw=90, pitch=-45, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)

        self.image_length = obs_img_dim*obs_img_dim*3
        self.object_ids = object_ids
        self._scaling_local = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5]
        shapenet_data = pickle.load(
            open(osp.join(SHAPENET_ASSET_PATH, 'metadata.pkl'), 'rb'))
        self.object_list = shapenet_data['object_list']
        self.scaling = shapenet_data['scaling']

        super().__init__(*args, **kwargs)
        self.theta = bullet.deg_to_quat([180, 0, 90])


    def _set_action_space(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_spaces(self):
        self._set_action_space()
        # obs = self.reset()
        if self._observation_mode == 'state':
            observation_dim = 7 + 1 + 7*self._num_objects
            obs_bound = 100
            obs_high = np.ones(observation_dim) * obs_bound
            self.observation_space = gym.spaces.Box(-obs_high, obs_high)
        elif self._observation_mode == 'pixels' or self._observation_mode == 'pixels_debug':
            img_space = gym.spaces.Box(0, 1, (self.image_length,), dtype=np.float32)
            if self._observation_mode == 'pixels':
                observation_dim = 7
            elif self._observation_mode == 'pixels_debug':
                observation_dim = 7 + 1 + 7*self._num_objects
            obs_bound = 100
            obs_high = np.ones(observation_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError


    def _load_meshes(self):
        if self._invisible_robot:
            self._sawyer = bullet.objects.sawyer_invisible()
        else:
            self._sawyer = bullet.objects.sawyer()

        self._table = bullet.objects.table()
        self._tray = bullet.objects.tray()

        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])

        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')
        if self.trimodal:
            self._num_objects = 3
            min_distance_threshold = 0.08
            if self.randomize:
                if self.all_random:
                    object_positions = []

                    while len(object_positions) < self._num_objects:
                        l, h = self._object_position_low, self._object_position_high
                        new_pos = np.array([random.uniform(l[0], h[0]), random.uniform(l[1], h[1]), random.uniform(l[2], h[2])])
                        okay = True
                        for pos in object_positions:
                            if np.linalg.norm(pos - new_pos) < min_distance_threshold:
                                okay = False
                        if okay:
                            object_positions.append(new_pos)
                else:
                    choice = random.randint(0, 1)
                    object_positions = self.trimodal_positions_two[choice]
                self._trimodal_positions = object_positions

            else:
                object_positions = self._trimodal_positions

            self._objects = {
                0: bullet.objects.cube(pos=object_positions[0], rgba=[1, 0, 0, 1], scale=0.033),
                1: bullet.objects.cube(pos=object_positions[1], rgba=[0, 1, 0, 1], scale=0.033),
                2: bullet.objects.cube(pos=object_positions[2], rgba=[0, 1, 1, 1], scale=0.033)
            }

        else: #single
            self._num_objects = 1
            if self.randomize:
                object_position = np.random.uniform(
                    low=self._object_position_low, high=self._object_position_high)
                self._fix_obj_position = object_position
            else:
                object_position = self._fix_obj_position

            self._objects = {
                0: bullet.objects.lego(pos=object_position)
            }

        for _ in range(50):
            bullet.step()

        object_positions_after_landing = []
        for object_name in range(self._num_objects):
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = object_info['pos']
            object_positions_after_landing.append(object_pos)

    def get_reward(self, info):
        object_list = self._objects.keys()
        reward = REWARD_NEGATIVE
        if self._single_obj_reward > -1:
            object_info = bullet.get_body_info(self._objects[self._single_obj_reward],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            object_height = object_pos[2]
            if object_height > self._reward_height_thresh:
                end_effector_pos = np.asarray(self.get_end_effector_pos())
                object_gripper_distance = np.linalg.norm(
                    object_pos - end_effector_pos)
                if object_gripper_distance < 0.1:
                    reward = REWARD_POSITIVE

        else:
            for object_name in object_list:
                object_info = bullet.get_body_info(self._objects[object_name],
                                                   quat_to_deg=False)
                object_pos = np.asarray(object_info['pos'])
                object_height = object_pos[2]
                if object_height > self._reward_height_thresh:
                    end_effector_pos = np.asarray(self.get_end_effector_pos())
                    object_gripper_distance = np.linalg.norm(
                        object_pos - end_effector_pos)
                    if object_gripper_distance < 0.1:
                        reward = REWARD_POSITIVE
            """            
            if self.reward_type == "dense":
                min_object_gripper_distance = 10000
                for object_name in self._obj_list:
                    object_info = bullet.get_body_info(self._objects[object_name],
                                                       quat_to_deg=False)
                    target_pos = np.asarray(object_info['pos'])
                    curr_object_gripper_distance = np.linalg.norm(target_pos - end_effector_pos)
                    min_object_gripper_distance = min(min_object_gripper_distance, curr_object_gripper_distance)
                reward = reward - min_object_gripper_distance
            """

        return reward

    def step(self, action):
        action = np.asarray(action)
        action = action * 10
        pos = list(bullet.get_link_state(self._sawyer, self._end_effector, 'pos'))
        delta_pos = action[:3]
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        theta = list(bullet.get_link_state(self._sawyer, self._end_effector, 'theta'))
        delta_theta = action[3]
        target_theta = theta + np.asarray([0., 0., delta_theta*20])
        target_theta = np.clip(target_theta, [180, 0., 0.], [180, 0., 180.])
        target_theta = bullet.deg_to_quat(target_theta)
        gripper = -0.8

        self._simulate(pos, target_theta, gripper)
        # if self._visualize: self.visualize_targets(pos)

        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')

        # Get object gripper distance before lifting
        if self._single_obj_reward > -1:
            object_info = bullet.get_body_info(self._objects[self._single_obj_reward],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            end_effector_pos = np.asarray(self.get_end_effector_pos())
            object_gripper_distance_goal = np.linalg.norm(object_pos - end_effector_pos)

        object_list = self._objects.keys()
        all_object_gripper_distance = []
        for object_name in object_list:
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            end_effector_pos = np.asarray(self.get_end_effector_pos())
            object_gripper_distance = np.linalg.norm(object_pos - end_effector_pos)
            all_object_gripper_distance.append(object_gripper_distance)

        if pos[2] < self._height_threshold:
            gripper = 0.8
            for i in range(10):
                self._simulate(pos, target_theta, gripper)
            for i in range(50):
                pos = bullet.get_link_state(self._sawyer, self._end_effector,
                                            'pos')
                pos = list(pos)
                pos = np.clip(pos, self._pos_low, self._pos_high)
                pos[2] += 0.05
                self._simulate(pos, target_theta, gripper)
            done = True
            reward = self.get_reward({})
            if reward > 0:
                info = {'grasp_success': 1.0,
                        'grasp_attempt': done,
                        'all_object_gripper_distance': all_object_gripper_distance,
                        'all_object_gripper_distance_min': min(all_object_gripper_distance)}
            else:
                info = {'grasp_success': 0.0,
                        'grasp_attempt': done,
                        'all_object_gripper_distance': all_object_gripper_distance,
                        'all_object_gripper_distance_min': min(all_object_gripper_distance)}
        else:
            done = False
            reward = REWARD_NEGATIVE
            if self.reward_type == "dense":
                if self._single_obj_reward > -1:
                    reward = reward - object_gripper_distance_goal
                else:
                    reward = reward - min(all_object_gripper_distance)
            info = {'grasp_success': 0.0,
                    'grasp_attempt': done,
                    'all_object_gripper_distance': all_object_gripper_distance}

        observation = self.get_observation()
        self._prev_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        return observation, reward, done, info

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        #print(img)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()
        end_effector_theta = bullet.get_link_state(
            self._sawyer, self._end_effector, 'theta', quat_to_deg=False)
        
        if self._observation_mode == 'state':
            observation = np.concatenate(
                (end_effector_pos, end_effector_theta, gripper_tips_distance))
            # object_list = self._objects.keys()
            for object_name in range(self._num_objects):
                object_info = bullet.get_body_info(self._objects[object_name],
                                                   quat_to_deg=False)
                object_pos = object_info['pos']
                object_theta = object_info['theta']
                observation = np.concatenate(
                    (observation, object_pos, object_theta))
        elif self._observation_mode == 'pixels':
            image_observation = self.render_obs()
            if self.normalize_and_flatten:
                image_observation = np.float32(image_observation.flatten()) / 255.0
            observation = {
                'state': np.concatenate(
                    (end_effector_pos, gripper_tips_distance)),
                'image': image_observation
            }
        elif self._observation_mode == 'pixels_debug':
            # This mode passes in all the true state information + images
            image_observation = self.render_obs()
            if self.normalize_and_flatten:
                image_observation = np.float32(image_observation.flatten()) / 255.0
            state_observation = np.concatenate(
                (end_effector_pos, end_effector_theta, gripper_tips_distance))

            for object_name in range(self._num_objects):
                object_info = bullet.get_body_info(self._objects[object_name],
                                                   quat_to_deg=False)
                object_pos = object_info['pos']
                object_theta = object_info['theta']
                state_observation = np.concatenate(
                    (state_observation, object_pos, object_theta))
            observation = {
                'state': state_observation,
                'image': image_observation,
            }
        else:
            raise NotImplementedError

        return observation


if __name__ == "__main__":
    import roboverse
    import time

    num_objects = 1
    env = roboverse.make("SawyerGraspOneV2-v0",
                         gui=True,
                         observation_mode='pixels_debug',)
                         # num_objects=num_objects)
    obs = env.reset()
    # object_ind = np.random.randint(0, env._num_objects)
    object_ind = num_objects - 1
    i = 0
    action = env.action_space.sample()
    for _ in range(5000):
        time.sleep(0.1)
        object_pos = obs["state"][object_ind*7 + 8: object_ind*7 + 8 + 3]
        ee_pos = obs["state"][:3]
        action = object_pos - ee_pos
        action = action*4.0
        action += np.random.normal(scale=0.1, size=(3,))

        # action = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        # if np.random.uniform() < 0.9:
        #     action[2] = -1
        # theta_action = np.random.uniform()
        theta_action = 0.0

        action = np.concatenate((action, np.asarray([theta_action])))
        obs, rew, done, info = env.step(action)
        env.render_obs()
        i+=1
        if done or i > 50:
            # object_ind = np.random.randint(0, env._num_objects)
            object_ind = num_objects - 1
            obs = env.reset()
            i = 0
            print('Reward: {}'.format(rew))
        # print(obs)

