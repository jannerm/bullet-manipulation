import roboverse.bullet as bullet
import numpy as np
from roboverse.envs.sawyer_base import SawyerBaseEnv


class SawyerGraspOneEnv(SawyerBaseEnv):

    def __init__(self,
                 goal_pos=(0.75, 0.2, -0.1),
                 reward_type='shaped',
                 reward_min=-2.5,
                 randomize=True,
                 observation_mode='state',
                 obs_img_dim=48,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param goal_pos: xyz coordinate of desired goal
        :param reward_type: one of 'shaped', 'sparse'
        :param reward_min: minimum possible reward per timestep
        :param randomize: whether to randomize the object position or not
        :param observation_mode: state, pixel, pixel_debug
        :param obs_img_dim: image dimensions for the observations
        """
        self._goal_pos = np.asarray(goal_pos)
        self._reward_type = reward_type
        self._reward_min = reward_min


        self._randomize = randomize
        self._multi_obj = True
        self._obj_list = ['lego', 'duck', 'cube']
        self._object = self._obj_list[0]


        self._observation_mode = observation_mode

        # self._object_position_low = (0.49999162183937296 + 0.3, -1.760392168808169e-05, -.36)#(.65, .10, -.36)
        # self._object_position_high = (0.49999162183937296 + 0.1, -1.760392168808169e-05, -.36)#(.8, .25, -.36)

        self._object_position_low = (0.49999162183937296, -1.760392168808169e-05 - 0.7, -.36)#(.65, .10, -.36)
        self._object_position_high = (0.49999162183937296, -1.760392168808169e-05 + 0.5, -.36)#(.8, .25, -.36)  # make trimodal a continuous distribution

        self._fixed_object_position = (0.49999162183937296 + 0.2, -1.760392168808169e-05, -.36)#-4.563183413075489e-08)#(.75, .2, -.36)
        self._trimodal_position1 = (0.49999162183937296, -1.760392168808169e-05 - 0.7, -.36)
        self._trimodal_position2 = (0.49999162183937296, -1.760392168808169e-05 - 0.1, -.36)
        self._trimodal_position3 = (0.49999162183937296, -1.760392168808169e-05 + 0.5, -.36)
        self._trimodal_positions = [self._trimodal_position1,self._trimodal_position2,self._trimodal_position3]

        self.obs_img_dim = obs_img_dim
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.75, +.15, -0.2], distance=0.3,
            yaw=90, pitch=-45, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)

        self.dt = 0.1
        super().__init__(*args, **kwargs)

    def _load_meshes(self):
        super()._load_meshes()

        if self._multi_obj:
            if self._randomize:
                object_positions = [np.random.uniform(low=self._object_position_low, high=self._object_position_high),
                                    np.random.uniform(low=self._object_position_low, high=self._object_position_high),
                                    np.random.uniform(low=self._object_position_low, high=self._object_position_high)]
            else:
                object_positions = self._trimodal_positions
            self._objects = {
                'lego': bullet.objects.lego(pos=object_positions[0]),
                'duck': bullet.objects.lego(pos=object_positions[1]),
                'cube': bullet.objects.lego(pos=object_positions[2])
            }
            choice = np.random.randint(3)
            self._object = self._obj_list[choice]
            print(self._object)
            object_position = object_positions[choice]
        else:
            if self._randomize:
                object_position = np.random.uniform(low=self._object_position_low, high=self._object_position_high)
            else:
                object_position = self._fixed_object_position
            self._objects = {
                'lego': bullet.objects.lego(pos=object_position)
            }
            self._object = 'lego'



        # if self._randomize:
        #     choice = np.random.randint(3)
        #     print("---------------choice: ", choice)
        #     object_position = self._trimodal_positions[choice]
        #     #object_position = np.random.uniform(
        #     #    low=self._object_position_low, high=self._object_position_high)
        # if self._uniform:
        #     object_position = np.random.uniform(low=self._object_position_low, high=self._object_position_high)
        #     print("obj position: ", object_position)
        #
        # else:
        #     object_position = self._fixed_object_position
        # self._objects = {
        #     'lego': bullet.objects.lego(pos=object_position)
        # }

        # set goal_pos to be directly above the randomized position 
        # of lego, rather than a fixed position
        self._goal_pos = np.copy(object_position)
        self._goal_pos[2] = self._goal_pos[2]

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        return observation, reward, done, info

    def get_info(self):
        object_pos = np.asarray(self.get_object_midpoint(self._object))  ## generalized lego
        object_goal_distance = np.linalg.norm(object_pos - self._goal_pos)
        end_effector_pos = self.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(
            object_pos - end_effector_pos)
        gripper_goal_distance = np.linalg.norm(
            self._goal_pos - end_effector_pos)
        object_goal_distance_z = max(self._goal_pos[2] - object_pos[2], 0)

        info = {
            'object_goal_distance': object_goal_distance,
            'object_gripper_distance': object_gripper_distance,
            'gripper_goal_distance': gripper_goal_distance,
            'object_goal_distance_z': object_goal_distance_z
        }
        return info

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs, self._projection_matrix_obs)
        return img

    def get_reward(self, info):

        if self._reward_type == 'sparse':
            if info['object_goal_distance'] < 0.05:
                reward = 1
            else:
                reward = 0
        elif self._reward_type == 'shaped':
            reward = -1*(4*info['object_goal_distance']
                         + info['object_gripper_distance'])
            reward = max(reward, self._reward_min)
        elif self._reward_type == 'grasp_only':
            reward = -1*(4*info['object_goal_distance_z']
                         + info['object_gripper_distance'])
            reward = max(reward, self._reward_min)
        else:
            raise NotImplementedError

        return reward

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

        if self._observation_mode == 'state':
            object_info = bullet.get_body_info(self._objects[self._object],
                                               quat_to_deg=False)
            object_pos = object_info['pos']
            object_theta = object_info['theta']
            observation = np.concatenate(
                (end_effector_pos, gripper_tips_distance,
                 object_pos, object_theta))
        elif self._observation_mode == 'pixels':
            image_observation = self.render_obs()
            observation = {
                'state': np.concatenate(
                    (end_effector_pos, gripper_tips_distance)),
                'image': image_observation
            }
        elif self._observation_mode == 'pixels_debug':
            # This mode passes in all the true state information + images
            image_observation = self.render_obs()

            object_info = bullet.get_body_info(self._objects[self._object],
                                               quat_to_deg=False)
            all_obj_pos = []
            for obj in self._obj_list:
                all_obj_pos.append(bullet.get_body_info(self._objects[obj],
                                               quat_to_deg=False)["pos"])

            object_pos = object_info['pos']
            object_theta = object_info['theta']
            state = np.concatenate(
                (end_effector_pos,gripper_tips_distance,
                 object_pos, object_theta))
            observation = {
                'state': state,
                'image': image_observation,
                'all_obj_pos': all_obj_pos
            }
        else:
            raise NotImplementedError

        return observation
