from roboverse.envs.widow200_grasp_v6_drawer_open_v0 import (
    Widow200GraspV6DrawerOpenV0Env, drawer_open_policy)
from roboverse.envs.rand_obj import RandObjEnv
from roboverse.utils.shapenet_utils import load_shapenet_object, \
    import_shapenet_metadata
import roboverse.bullet as bullet
from roboverse.bullet.objects import lifted_long_box_open_top_center_pos
import roboverse.utils as utils
import numpy as np
import time
import roboverse
from PIL import Image
import skimage.io as skii
import skimage.transform as skit
import itertools

obj_path_map, path_scaling_map = import_shapenet_metadata()

class Widow200GraspV6DrawerPlaceThenOpenV0Env(Widow200GraspV6DrawerOpenV0Env):
    """
    Setup: blocking_obj blocking the drawer from being opened. 
    obj in closed drawer.
    Task: grasp blocking_obj, put in box above drawer.
    open drawer. grasp obj.
    """

    def __init__(self,
                 *args,
                 object_name_scaling=("ball", 0.5),
                 blocking_object_name_scaling=("shed", 0.4),
                 success_dist_threshold=0.04,
                 noisily_open_drawer=False,
                 randomize_blocking_obj_quat=False,
                 open_grasp_only=False,
                 place_only=False,
                 **kwargs):
        self.noisily_open_drawer = noisily_open_drawer
        self.randomize_blocking_obj_quat = randomize_blocking_obj_quat
        self.object_name = object_name_scaling[0]
        self.blocking_object_name = blocking_object_name_scaling[0]

        assert self.object_name != self.blocking_object_name
        object_names = (self.object_name, self.blocking_object_name)
        scaling_local_list = [
            object_name_scaling[1], blocking_object_name_scaling[1]]
        num_objects = 2

        self._object_position_high = (.82, -.08, -.29)
        self._object_position_low = (.82, -.09, -.29)

        self.place_only = place_only
        self.open_grasp_only = open_grasp_only
        assert not (self.place_only and self.open_grasp_only)
        self.box_high = lifted_long_box_open_top_center_pos + np.array([0.0525, 0.04, 0.035])
        self.box_low = lifted_long_box_open_top_center_pos + np.array([-0.0525, -0.04, -0.01])

        if not self.open_grasp_only:
            # Blocking = Obstruction object
            # Drop object blocking drawer
            blocking_object_offset = np.array([0, -0.03, -0.05])
            self._blocking_object_position_high = self._object_position_high + blocking_object_offset
            self._blocking_object_position_low = self._object_position_low + blocking_object_offset
            self._success_dist_threshold = success_dist_threshold
        else:
            # Drop the object in the box.
            margin = np.array([0.03, 0.03, 0])
            drop_height = self.box_high[2]
            self._blocking_object_position_high = list(self.box_high[:2]) + [drop_height]
            self._blocking_object_position_high -= margin
            self._blocking_object_position_low = list(self.box_low[:2]) + [drop_height]
            self._blocking_object_position_low += margin

        super().__init__(*args,
            object_names=object_names,
            scaling_local_list=scaling_local_list,
            num_objects=num_objects, **kwargs)

        self._env_name = "Widow200GraspV6DrawerPlaceThenOpenV0Env"

        if self.place_only:
            self.scripted_traj_len = 30
        elif self.open_grasp_only:
            self.scripted_traj_len = 50
        else:
            self.scripted_traj_len = 80

    def get_random_quat(self):
        quat_possible_vals = [-1, 0, 1]
        vals = [quat_possible_vals] * 4
        possible_quats = list(itertools.product(*vals))
        possible_quats.remove((0, 0, 0, 0))
        random_idx = np.random.random_integers(0, len(possible_quats) - 1)
        random_quat = possible_quats[random_idx]
        return random_quat

    def _load_meshes(self):
        self._robot_id = bullet.objects.widowx_200()
        self._table = bullet.objects.table()
        self._workspace = bullet.Sensor(self._robot_id,
                                        xyz_min=self._pos_low,
                                        xyz_max=self._pos_high,
                                        visualize=False, rgba=[0, 1, 0, .1])
        self._tray = bullet.objects.widow200_hidden_tray()
        self._objects = {}
        self._drawer = bullet.objects.drawer()
        bullet.open_drawer(self._drawer, noisy_open=self.noisily_open_drawer)
        object_position = np.random.uniform(
            self._object_position_low, self._object_position_high)
        blocking_object_position = np.random.uniform(
            self._blocking_object_position_low,
            self._blocking_object_position_high)
        object_positions = np.concatenate(
            (object_position, blocking_object_position), axis=0)

        self.load_object(self.object_name, object_position)

        bullet.close_drawer(self._drawer)

        if self.open_grasp_only or self.randomize_blocking_obj_quat:
            self.blocking_obj_quat = self.get_random_quat()
        else:
            self.blocking_obj_quat = [1, -1, 0, 0]

        self.load_object(
            self.blocking_object_name, blocking_object_position, quat=self.blocking_obj_quat)

        self._box = bullet.objects.lifted_long_box_open_top()

    def load_object(self, name, pos, quat=[1, -1, 0, 0]):
        self._objects[name] = load_shapenet_object(
            obj_path_map[name], self.scaling,
            pos, scale_local=self._scaling_local[name], quat=quat)

    def get_box_pos(self):
        box_open_top_info = bullet.get_body_info(self._box, quat_to_deg=False)
        return np.asarray(box_open_top_info['pos'])

    def is_object_grasped(self):
        object_info = bullet.get_body_info(
            self._objects[self.object_name], quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])
        object_gripper_distance = np.linalg.norm(object_pos - self.get_end_effector_pos())
        return (object_pos[2] > self._reward_height_thresh) and (object_gripper_distance < 0.1)

    def get_reward(self, info):
        reward = float(self.is_object_grasped())
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

    def get_info(self):
        info = {}

        # object pos
        assert self._num_objects == 2
        object_info = bullet.get_body_info(
            self._objects[self.object_name], quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])

        # Blocking object pos
        blocking_object_info = bullet.get_body_info(
            self._objects[self.blocking_object_name], quat_to_deg=False)
        blocking_object_pos = np.asarray(blocking_object_info['pos'])

        ee_pos = np.array(self.get_end_effector_pos())
        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        info['object_gripper_dist'] = object_gripper_dist
        blocking_object_gripper_dist = np.linalg.norm(blocking_object_pos - ee_pos)
        info['blocking_object_gripper_dist'] = blocking_object_gripper_dist

        info['is_drawer_opened'] = int(self.is_drawer_opened())
        info['is_drawer_opened_widely'] = int(self.is_drawer_opened(widely=True))
        info['drawer_y_pos'] = self.get_drawer_bottom_pos()[1] # y coord

        # gripper-handle dist
        ee_pos = np.array(self.get_end_effector_pos())
        gripper_handle_dist = np.linalg.norm(ee_pos - self.get_handle_pos())
        info['gripper_handle_dist'] = gripper_handle_dist

        # Blocking object and box info
        box_pos = self.get_box_pos()

        blocking_object_box_dist = np.linalg.norm(blocking_object_pos - box_pos)

        blocking_object_box_dist_success = int(blocking_object_box_dist < self._success_dist_threshold)

        blocking_object_within_box_bounds = ((self.box_low <= blocking_object_pos)
            & (blocking_object_pos <= self.box_high))
        blocking_object_in_box_success = int(np.all(blocking_object_within_box_bounds))

        blocking_object_xy_in_box_xy = int(np.all(blocking_object_within_box_bounds[:2]))
        blocking_object_z_above_box_z = int(blocking_object_pos[2] >= self.box_low[2])
        blocking_object_above_box_sucess = (
            blocking_object_xy_in_box_xy and blocking_object_z_above_box_z)

        info['blocking_object_box_dist'] = blocking_object_box_dist
        info['blocking_object_box_dist_success'] = blocking_object_box_dist_success
        info['blocking_object_in_box_success'] = blocking_object_in_box_success
        info['blocking_object_above_box_success'] = blocking_object_above_box_sucess

        return info

class Widow200GraspV6DrawerPlaceThenOpenV0PickPlaceOnlyEnv(Widow200GraspV6DrawerPlaceThenOpenV0Env):
    """
    Setup: blocking_obj blocking the drawer from being opened.
    obj in closed drawer.
    Task: grasp blocking_obj, put in box.
    """

    def __init__(self,
                 *args,
                 object_name_scaling=("ball", 0.5),
                 blocking_object_name_scaling=("shed", 0.4),
                 success_dist_threshold=0.04,
                 noisily_open_drawer=False,
                 randomize_blocking_obj_quat=False,
                 open_grasp_only=False,
                 place_only=True,
                 **kwargs):
        super().__init__(*args,
            object_name_scaling=object_name_scaling,
            blocking_object_name_scaling=blocking_object_name_scaling,
            success_dist_threshold=success_dist_threshold,
            randomize_blocking_obj_quat=randomize_blocking_obj_quat,
            place_only=place_only,
            **kwargs)

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        reward = float(info['blocking_object_in_box_success'])
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

def drawer_place_then_open_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    blocking_object_ind = 1
    for i in range(200):
        obs = env.reset()
        print("env.scripted_traj_len", env.scripted_traj_len)
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        box_dist_thresh = 0.035 + np.random.normal(scale=0.01)
        box_dist_thresh = np.clip(box_dist_thresh, 0.025, 0.05)

        drawer_never_opened = True

        images, images_for_gif = [], [] # new video at the start of each trajectory

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            box_pos = env.get_box_pos()
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            blocking_object_pos = obj_obs[
                blocking_object_ind * 7 : blocking_object_ind * 7 + 3]
            handle_pos = env.get_handle_pos()
            object_lifted_with_margin = object_pos[2] > (
                env._reward_height_thresh + margin)
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            blocking_object_gripper_dist = np.linalg.norm(
                blocking_object_pos - ee_pos)
            blocking_object_box_dist = np.linalg.norm(blocking_object_pos - box_pos)
            gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
            theta_action = 0.

            blocking_object_pos_offset = np.array([0, -0.01, 0])

            info = env.get_info()

            z_diff = abs(blocking_object_pos[2] + blocking_object_pos_offset[2] - ee_pos[2])

            if ((blocking_object_gripper_dist > dist_thresh or z_diff > 0.015) and
                env._gripper_open and not info['blocking_object_above_box_success']):
                # print('approaching')
                action = ((blocking_object_pos +
                    blocking_object_pos_offset) - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.03:
                    action[2] *= 0.3
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif (env._gripper_open and blocking_object_box_dist > box_dist_thresh and
                not info['blocking_object_in_box_success']):
                action = (blocking_object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif (blocking_object_gripper_dist > 2 * dist_thresh and
                not info['blocking_object_above_box_success']):
                # Open gripper
                # Remove this case for scripted_collect.
                break
            elif (blocking_object_box_dist > box_dist_thresh and
                not info['blocking_object_above_box_success']):
                action = (box_pos - blocking_object_pos)*7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if "DrawerPlaceThenOpen" in env._env_name:
                    # print("don't droop down until xy-close to box")
                    action[2] = 0.2
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
                # print("blocking_object_pos", blocking_object_pos)
            elif not info['blocking_object_in_box_success']:
                # object is now above the box.
                action = (box_pos - blocking_object_pos)*7.0
                action[2] = 0.2
                action = np.concatenate(
                    (action, np.asarray([0., 0.7, 0.])))
            elif (gripper_handle_dist > dist_thresh
                and not env.is_drawer_opened(widely=drawer_never_opened)):
                # print('approaching handle')
                handle_pos_offset = np.array([0.0, -0.01, -0.01])
                action = ((handle_pos + handle_pos_offset) - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.75 * dist_thresh:
                    action[2] = 0.2 # force upward action to avoid upper box
                action = np.concatenate((action, np.asarray([theta_action,0.7,0.])))
            elif not env.is_drawer_opened(widely=drawer_never_opened):
                # print("opening drawer")
                action = np.array([0, -1.0, 0])
                # action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif (object_gripper_dist > dist_thresh
                and env._gripper_open and gripper_handle_dist < 1.5 * dist_thresh):
                # print("Lift upward")
                drawer_never_opened = False
                action = np.array([0, 0, 0.7]) # force upward action to avoid upper box
                action = np.concatenate(
                    (action, np.asarray([theta_action, 0., 0.])))
            elif object_gripper_dist > dist_thresh and env._gripper_open:
                # print("Move toward object")
                action = (object_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > dist_thresh:
                    action[2] = 0.3
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif env._gripper_open:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif not object_lifted_with_margin:
                # print('raise object upward')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif object_gripper_dist > 2 * dist_thresh:
                # Open gripper
                # Remove this case for scripted_collect.
                action = np.array([0, 0, 0, 0, 0.7, 0])
            else:
                # Move above tray's xy-center.
                tray_info = roboverse.bullet.get_body_info(
                    env._tray, quat_to_deg=False)
                tray_center = np.asarray(tray_info['pos'])
                action = (tray_center - ee_pos)[:2]
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0., 0.])))

            noise_scalings = [noise] * 3 + [0.1 * noise] + [noise] * 2
            action += np.random.normal(scale=noise_scalings)
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print(action)
            obs, rew, done, info = env.step(action)
            # print("block object in box." , info['blocking_object_in_box_success'])
            # print("rew", rew)

            img = env.render_obs()
            if save_video:
                images.append(img)
                # img = np.transpose(img, (1, 2, 0))
                img_side = 48
                img = skit.resize(img, (img_side * 3, img_side * 3, 3)) # middle dimensions should be 48
                images_for_gif.append(Image.fromarray(np.uint8(img * 255)))

            time.sleep(0.05)

        if rew > 0:
            print("i", i)
            print('reward: {}'.format(rew))
            print('--------------------')

        if save_video:
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)
            images_for_gif[0].save('data/grasp_place_{}.gif'.format(i),
                save_all=True, append_images=images_for_gif[1:],
                duration=env.scripted_traj_len * 2, loop=0)

def drawer_place_only_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    blocking_object_ind = 1
    for i in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        box_dist_thresh = 0.035 + np.random.normal(scale=0.01)
        box_dist_thresh = np.clip(box_dist_thresh, 0.025, 0.05)

        images, images_for_gif = [], [] # new video at the start of each trajectory

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            box_pos = env.get_box_pos()
            blocking_object_pos = obj_obs[
                blocking_object_ind * 7 : blocking_object_ind * 7 + 3]
            ending_target_pos = np.array([0.73822169, -0.03909928, -0.25635483]) # Effective neutral pos.
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            blocking_object_gripper_dist = np.linalg.norm(
                blocking_object_pos - ee_pos)
            blocking_object_box_dist = np.linalg.norm(
                blocking_object_pos - box_pos)
            theta_action = 0.

            blocking_object_pos_offset = np.array([0, -0.01, 0.0])

            info = env.get_info()

            z_diff = abs(blocking_object_pos[2] + blocking_object_pos_offset[2] - ee_pos[2])

            if ((blocking_object_gripper_dist > dist_thresh or z_diff > 0.015) and
                env._gripper_open and not info['blocking_object_above_box_success']):
                # print('approaching')
                action = ((blocking_object_pos +
                    blocking_object_pos_offset) - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.03:
                    action[2] *= 0.3
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif (env._gripper_open and blocking_object_box_dist > box_dist_thresh and
                not info['blocking_object_in_box_success']):
                action = (blocking_object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif (blocking_object_box_dist > box_dist_thresh and
                not info['blocking_object_above_box_success']):
                action = (box_pos - blocking_object_pos)*7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
                # print("blocking_object_pos", blocking_object_pos)
            elif not info['blocking_object_in_box_success']:
                # object is now above the box.
                action = (box_pos - blocking_object_pos)*7.0
                action[2] = 0.2
                action = np.concatenate(
                    (action, np.asarray([0., 0.7, 0.])))
            else:
                # print('move to neutral')
                action = (ending_target_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > dist_thresh:
                    action[2] = 0.2 # force upward action to avoid upper box
                action = np.concatenate((action, np.asarray([theta_action,0.7,0.])))

            noise_scalings = [noise] * 3 + [0.1 * noise] + [noise] * 2
            action += np.random.normal(scale=noise_scalings)
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print(action)
            obs, rew, done, info = env.step(action)
            # print("block object in box." , info['blocking_object_in_box_success'])
            # print("rew", rew)

            img = env.render_obs()
            if save_video:
                images.append(img)
                # img = np.transpose(img, (1, 2, 0))
                img_side = 48
                img = skit.resize(img, (img_side * 3, img_side * 3, 3)) # middle dimensions should be 48
                images_for_gif.append(Image.fromarray(np.uint8(img * 255)))

            time.sleep(0.05)

        print("i", i)
        if rew > 0:
            print('blocking object in box')
            print('--------------------')

        if save_video:
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)
            images_for_gif[0].save('data/grasp_place_{}.gif'.format(i),
                save_all=True, append_images=images_for_gif[1:],
                duration=env.scripted_traj_len * 2, loop=0)

if __name__ == "__main__":
    EPSILON = 0.05
    noise = 0.2
    margin = 0.025
    save_video = True

    mode = "PickPlaceOnly"

    gui = True
    reward_type = "sparse"
    obs_mode = "pixels_debug"
    if mode == "PlaceThenOpen":
        env = roboverse.make("Widow200GraspV6DrawerPlaceThenOpenV0-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_place_then_open_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "PickPlaceOnly":
        env = roboverse.make("Widow200GraspV6DrawerPlaceThenOpenV0PickPlaceOnly-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_place_only_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "OpenGraspOnly":
        env = roboverse.make("Widow200GraspV6DrawerPlaceThenOpenV0OpenGraspOnly-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_open_policy(EPSILON, noise, margin, save_video, env)
    else:
        raise NotImplementedError
