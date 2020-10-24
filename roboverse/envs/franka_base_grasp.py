import numpy as np
import gym
import pdb

import roboverse.bullet as bullet
from roboverse.envs.serializable import Serializable
from roboverse.envs.franka_grasp_v2 import load_shapenet_object
import pybullet as p
from roboverse.bullet.ik import get_joint_positions
from roboverse.bullet.misc import load_obj
import os.path as osp
import pickle
SHAPENET_ASSET_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)), 'assets/bullet-objects/ShapeNetCore')

REWARD_NEGATIVE = 0
REWARD_POSITIVE = 1


class FrankaBaseGraspEnv(gym.Env, Serializable):
    def __init__(self,
                 img_dim=256,
                 gui=False,
                 action_scale=0.2,
                 action_repeat=10,
                 timestep=1./120,
                 solver_iterations=150,
                 gripper_bounds=[-1,1],
                 pos_init=[0.5, 0, 0],
                 pos_high=[1, .4, 0.2],
                 pos_low=[.4, -.6, -.36],
                # pos_low=[.4,-.6,0],
                 max_force=1000.,
                 visualize=True,
                 obs_img_dim=512,
                 transpose_image=False,
                 observation_mode='pixels_debug',
                 ):
        self._observation_mode = observation_mode
        self._gui = gui
        self._action_scale = action_scale
        self._action_repeat = action_repeat
        self._timestep = timestep
        self._solver_iterations = solver_iterations
        self._gripper_bounds = gripper_bounds
        self._pos_init = pos_init
        self._pos_low = pos_low
        self._pos_high = pos_high
        self._max_force = max_force
        self._visualize = visualize
        self._id = 'FrankaBaseGraspEnv'
        
        self._reward_height_thresh = 0.3
        
        self.camera_target_pos=[0.8, -0.2, -0.36]
        self.camera_distance=0.8
        self.camera_yaw=45
        self.camera_pitch=-30
        self.camera_roll=0

        self._object_position_low = (.67, -0.2, -.30)
        self._object_position_high = (.72, -0.2, -.30)
        self._num_objects = 1

        self.obs_img_dim = obs_img_dim
        self.image_shape = (obs_img_dim, obs_img_dim)
        self._transpose_image = transpose_image
        
        shapenet_data = pickle.load(open(osp.join(SHAPENET_ASSET_PATH, 'metadata.pkl'), 'rb'))
        self.object_list = shapenet_data['object_list']
        self.scaling = shapenet_data['scaling']
        self._scaling_local = [0.4,0.4]

        self._gripper_range = range(9, 11)

        view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance, yaw=self.camera_yaw,
                                pitch=self.camera_pitch, roll=self.camera_roll, up_axis_index=2)

        self._img_dim = img_dim
        self._view_matrix_obs = bullet.get_view_matrix(**view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(self._img_dim, self._img_dim)
        self.image_length = obs_img_dim*obs_img_dim*3

        self.theta = bullet.deg_to_quat([180, 0, 0])

        bullet.connect_headless(self._gui)
        # self.set_reset_hook()
        self._set_spaces()
        self.reset()
        

    def get_params(self):
        labels = ['_action_scale', '_action_repeat', 
                  '_timestep', '_solver_iterations',
                  '_gripper_bounds', '_pos_low', '_pos_high', '_id']
        params = {label: getattr(self, label) for label in labels}
        return params

    @property
    def parallel(self):
        return False
    
    def check_params(self, other):
        params = self.get_params()
        assert set(params.keys()) == set(other.keys())
        for key, val in params.items():
            if val != other[key]:
                message = 'Found mismatch in {} | env : {} | demos : {}'.format(
                    key, val, other[key]
                )
                raise RuntimeError(message)
   
    def _set_action_space(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    # def get_constructor(self):
    #     return lambda: self.__class__(*self.args_, **self.kwargs_)

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

    def reset(self):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_meshes()
        self._format_state_query()

        self._prev_pos = np.array(self._pos_init)
        bullet.position_control(self._franka, self._end_effector, self._prev_pos, self.theta)
        q_pos = np.array([0, np.pi/32, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi/4])
        
        joint_indices, _ = get_joint_positions(self._franka)
        for joint_ind, pos in zip(joint_indices, q_pos):
            p.resetJointState(self._franka, joint_ind, pos)
        # self._reset_hook(self)
        for _ in range(6):
            self.step([np.random.random()*0.2-0.1,np.random.random()*0.2-0.1,0.08,-1])
            # self.step([0,0,0.1, -1])
        # self.open_gripper()
        return self.get_observation()

    # def set_reset_hook(self, fn=lambda env: None):
    #     self._reset_hook = fn

    def open_gripper(self, act_repeat=10):
        delta_pos = [0,0,0]
        gripper = 0
        for _ in range(act_repeat):
            self.step(delta_pos, gripper)

    def get_body(self, name):
        if name == 'franka':
            return self._franka
        else:
            return self._objects[name]

    def get_object_midpoint(self, object_key):
        return bullet.get_midpoint(self._objects[object_key])

    def get_end_effector_pos(self):
        return bullet.get_link_state(self._franka, self._end_effector, 'pos')

    def _load_meshes(self):
        self._franka = bullet.objects.franka()
        self._table = bullet.objects.table()
        self._tray = bullet.objects.tray_v2()
        self._cube = bullet.objects.cube2()

        self.object_ids = [1,1]

        self._objects = {}
        self._sensors = {}
        # print(self._pos_low)
        # print(self._pos_high)
        self._workspace = bullet.Sensor(self._franka,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=self._visualize, rgba=[0,1,0,.1])


        self._end_effector = bullet.get_index_by_attribute(
            self._franka, 'link_name', 'panda_hand')

        import scipy.spatial
        min_distance_threshold = 0.12
        object_positions = np.random.uniform(
            low=self._object_position_low, high=self._object_position_high)
        object_positions = np.reshape(object_positions, (1,3))
        while object_positions.shape[0] < self._num_objects:
            object_position_candidate = np.random.uniform(
                low=self._object_position_low, high=self._object_position_high)
            object_position_candidate = np.reshape(
                object_position_candidate, (1,3))
            min_distance = scipy.spatial.distance.cdist(
                object_position_candidate, object_positions)
            if (min_distance > min_distance_threshold).any():
                object_positions = np.concatenate(
                    (object_positions, object_position_candidate), axis=0)

        assert len(self.object_ids) >= self._num_objects
        import random
        indexes = list(range(self._num_objects))
        random.shuffle(indexes)
        for idx in indexes:
            key_idx = self.object_ids.index(self.object_ids[idx])
            self._objects[key_idx] = load_shapenet_object(
                self.object_list[self.object_ids[idx]], self.scaling,
                object_positions[idx], scale_local=self._scaling_local[idx])
            for _ in range(10):
                bullet.step()

    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [v for k,v in self._objects.items() if not bullet.has_fixed_root(v)]
        ## position and orientation of link
        links = [(self._franka, self._end_effector)]
        ## position and velocity of prismatic joint
        joints = [(self._franka, None)]
        self._state_query = bullet.format_sim_query(bodies, links, joints)

    def _format_action(self, *action):
        if len(action) == 1:
            delta_pos, gripper = action[0][:-1], action[0][-1]
        elif len(action) == 2:
            delta_pos, gripper = action[0], action[1]
        else:
            raise RuntimeError('Unrecognized action: {}'.format(action))
        # print(delta_pos, gripper)
        return np.array(delta_pos), gripper

    # def get_observation(self):
        # observation = bullet.get_sim_state(*self._state_query)
        # return observation

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._franka, self._end_effector, 'pos')
        
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        # if gripper < 0:
        #     gripper = -1
        # else:
        #     gripper = 1


        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        image_observation = self.render_obs()
        reward = self.get_reward(observation)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._franka, self._end_effector, 'pos')
        return observation, reward, done, {}

    def _simulate(self, pos, theta, gripper):
        for _ in range(self._action_repeat):
            bullet.sawyer_position_ik(
                self._franka, self._end_effector, 
                pos, theta,
                gripper, gripper_bounds=self._gripper_bounds, 
                discrete_gripper=False, gripper_name=('panda_finger_joint1','panda_finger_joint2'), max_force=self._max_force
            )
            bullet.step_ik(self._gripper_range)

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def get_termination(self, observation):
        return False
    
    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._franka, 'panda_finger_joint1', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._franka, 'panda_finger_joint2', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()
        end_effector_theta = bullet.get_link_state(
            self._franka, self._end_effector, 'theta', quat_to_deg=False)

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
            image_observation = np.float32(image_observation.flatten())/255.0
            # image_observation = np.zeros((48, 48, 3), dtype=np.uint8)
            observation = {
                'state': np.concatenate(
                    (end_effector_pos, gripper_tips_distance)),
                'image': image_observation
            }
        elif self._observation_mode == 'pixels_debug':
            # This mode passes in all the true state information + images
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
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
        # import ipdb; ipdb.set_trace()
        return observation

    def get_reward(self, info):
        object_list = self._objects.keys()
        reward = REWARD_NEGATIVE
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
        return reward

    def visualize_targets(self, pos):
        bullet.add_debug_line(self._prev_pos, pos)

    def save_state(self, *save_path):
        state_id = bullet.save_state(*save_path)
        return state_id

    def load_state(self, load_path):
        bullet.load_state(load_path)
        obs = self.get_observation()
        return obs

    '''
        prevents always needing a gym adapter in softlearning
        @TODO : remove need for this method
    '''
    def convert_to_active_observation(self, obs):
        return obs


if __name__ == "__main__":
    fr = FrankaBaseGraspEnv(img_dim=256, gui=True, action_scale=1, action_repeat=10, timestep=1./120, solver_iterations=150,
     gripper_bounds=[-1,1],max_force=1000., visualize=False,)
    
    # print(fr.reset())
    from time import sleep
    sleep(2)
    import numpy as np
    
    pos = np.array(bullet.get_link_state(fr._franka, fr._end_effector, 'pos'))

    pos_arr = []
    obs_arr = []
    pos[2] += 0.1
    for t in range(50):
        act = np.zeros(4)
        act[0:3] = np.random.uniform(-1,1,(3,)) * 0.3
        # act[2] = pos[2] + np.sin(t) * 0.5 - bullet.get_link_state(fr._franka, fr._end_effector, 'pos')[2]
        act[3] = (-1)**t
        print('action', act)
        print('loc', bullet.get_link_state(fr._franka, fr._end_effector, 'pos'))
        pos_arr.append(bullet.get_link_state(fr._franka, fr._end_effector, 'pos'))
        obs_arr.append(fr.step(act))
        sleep(.05)    
    import ipdb; ipdb.set_trace()