from roboverse.envs.widow200_grasp_v6_drawer_open_v0 import (
    Widow200GraspV6DrawerOpenV0Env)
from roboverse.envs.rand_obj import RandObjEnv
import roboverse.bullet as bullet
import roboverse.utils as utils
import numpy as np
import time
import roboverse
from PIL import Image
import skimage.io as skii
import skimage.transform as skit


class Widow200GraspV6DoubleDrawerV0Env(Widow200GraspV6DrawerOpenV0Env):
    """Task is to open drawer, then grasp object inside it."""
    def __init__(self,
                 *args,
                 object_names=('ball',),
                 scaling_local_list=[0.5],
                 success_dist_threshold=0.04,
                 noisily_open_drawer=False,
                 task="CloseOpenGrasp",
                 **kwargs):
        self.noisily_open_drawer = noisily_open_drawer

        assert task in ["CloseOpenGrasp", "Close", "OpenGrasp",
            "CloseOpen", "Grasp"]
        self.task = task

        super().__init__(*args,
            object_names=object_names,
            scaling_local_list=scaling_local_list,
            **kwargs)
        self._env_name = "Widow200GraspV6DoubleDrawerV0Env"
        self._object_position_high = (.84, -.05, -.29)
        self._object_position_low = (.84, -.06, -.29)
        self._success_dist_threshold = success_dist_threshold

        self.task_to_traj_len_map = {
            'Close': 30,
            'OpenGrasp': 50,
            'Grasp': 25,
            'CloseOpen': 60,
            'CloseOpenGrasp': 80,
        }
        self.scripted_traj_len = self.task_to_traj_len_map[self.task]

    def _load_meshes(self):
        self._robot_id = bullet.objects.widowx_200()
        self._table = bullet.objects.table()
        self._workspace = bullet.Sensor(self._robot_id,
                                        xyz_min=self._pos_low,
                                        xyz_max=self._pos_high,
                                        visualize=False, rgba=[0, 1, 0, .1])
        self._tray = bullet.objects.widow200_hidden_tray()
        self._objects = {}
        self._bottom_drawer = bullet.objects.drawer_bottom()
        bullet.open_drawer(self._bottom_drawer)
        object_positions = self._generate_object_positions()
        self._load_objects(object_positions)
        bullet.close_drawer(self._bottom_drawer)

        self._top_drawer = bullet.objects.drawer_no_handle()

        if not self.task in ["OpenGrasp", "Grasp"]:
            # Open Top drawer only if it is not openGrasp or Grasp.
            # (both assume top already closed)
            bullet.open_drawer(
                self._top_drawer, noisy_open=self.noisily_open_drawer, half_open=True)

        if self.task == "Grasp":
            # Open bottom drawer
            bullet.open_drawer(self._bottom_drawer, noisy_open=self.noisily_open_drawer)

        self.drawers = {"top": self._top_drawer, "bottom": self._bottom_drawer}

    def get_drawer_bottom_pos(self, drawer_name):
        drawer = self.drawers[drawer_name]
        link_names = [bullet.get_joint_info(drawer, j, 'link_name')
            for j in range(bullet.p.getNumJoints(drawer))]
        drawer_bottom_link_idx = link_names.index('base')
        drawer_bottom_pos = bullet.get_link_state(
            drawer, drawer_bottom_link_idx, "pos")
        return np.array(drawer_bottom_pos)

    def get_bottom_drawer_handle_pos(self):
        link_names = [bullet.get_joint_info(self._bottom_drawer, j, 'link_name')
            for j in range(bullet.p.getNumJoints(self._bottom_drawer))]
        handle_link_idx = link_names.index('handle_r')
        handle_pos = bullet.get_link_state(
            self._bottom_drawer, handle_link_idx, "pos")
        return np.array(handle_pos)

    def is_drawer_opened(self, drawer_name, widely=False):
        opened_thresh = 0.03 if not widely else -0.02
        return self.get_drawer_bottom_pos(drawer_name)[1] < opened_thresh

    def is_drawer_closed(self, drawer_name):
        closed_thresh = 0.06
        return self.get_drawer_bottom_pos(drawer_name)[1] > closed_thresh

    def get_info(self):
        info = {}

        # object gripper dist
        assert self._num_objects == 1
        object_name = list(self._objects.keys())[0]
        object_info = bullet.get_body_info(self._objects[object_name],
                                           quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])
        ee_pos = np.array(self.get_end_effector_pos())
        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        info['object_gripper_dist'] = object_gripper_dist

        for drawer_name in ['top', 'bottom']:
            info['is_{}_drawer_opened'.format(drawer_name)] = int(
                self.is_drawer_opened(drawer_name))
            info['is_{}_drawer_opened_widely'.format(drawer_name)] = int(
                self.is_drawer_opened(drawer_name, widely=True))
            info['{}_drawer_y_pos'.format(drawer_name)] = (
                self.get_drawer_bottom_pos(drawer_name)[1]) # y coord

        # gripper-handle dist
        ee_pos = np.array(self.get_end_effector_pos())
        gripper_bottom_drawer_handle_dist = np.linalg.norm(
            ee_pos - self.get_bottom_drawer_handle_pos())
        info['gripper_bottom_drawer_handle_dist'] = gripper_bottom_drawer_handle_dist
        return info

class Widow200GraspV6DoubleDrawerV0CloseEnv(Widow200GraspV6DoubleDrawerV0Env):
    """Task is to open drawer, then grasp object inside it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="Close", **kwargs)

    def get_reward(self, info):
        reward = float(self.is_drawer_closed("top"))
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

class Widow200GraspV6DoubleDrawerV0CloseOpenEnv(Widow200GraspV6DoubleDrawerV0Env):
    """Task is to open drawer, then grasp object inside it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="CloseOpen", **kwargs)

    def get_reward(self, info):
        reward = float(self.is_drawer_opened("bottom", widely=True))
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

class Widow200GraspV6DoubleDrawerV0GraspEnv(Widow200GraspV6DoubleDrawerV0Env):
    """Task is to open drawer, then grasp object inside it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="Grasp", **kwargs)

    def get_reward(self, info):
        reward = float(self.is_drawer_opened("bottom", widely=True))
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

def close_open_grasp_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    for i in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)
        drawer_never_opened = True
        reached_pushing_region = False

        images, images_for_gif = [], [] # new video at the start of each trajectory.

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            bottom_drawer_handle_pos = env.get_bottom_drawer_handle_pos()
            object_lifted_with_margin = object_pos[2] > (
                env._reward_height_thresh + margin)
            top_drawer_pos = env.get_drawer_bottom_pos("top")
            top_drawer_push_target_pos = (top_drawer_pos +
                np.array([0, -0.15, 0.02]))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            gripper_handle_dist = np.linalg.norm(bottom_drawer_handle_pos - ee_pos)
            is_gripper_ready_to_push = (ee_pos[1] < top_drawer_push_target_pos[1] and
                ee_pos[2] < top_drawer_push_target_pos[2])
            theta_action = 0.

            if (not env.is_drawer_closed("top") and not reached_pushing_region and
                not is_gripper_ready_to_push):
                # print("move up and left")
                action = np.concatenate(
                    ([-0.2, -0.4, -0.2], np.array([theta_action, 0, 0])))
            elif not env.is_drawer_closed("top"):
                # print("close top drawer")
                reached_pushing_region = True
                action = (top_drawer_pos - ee_pos) * 7.0
                action[0] *= 3
                action[1] *= 0.6
                action = np.concatenate((action, np.array([theta_action, 0, 0])))
            elif (gripper_handle_dist > dist_thresh
                and not env.is_drawer_opened("bottom", widely=drawer_never_opened)):
                # print('approaching handle')
                handle_pos_offset = np.array([0, 0, 0])
                action = (bottom_drawer_handle_pos + handle_pos_offset- ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > dist_thresh:
                    action[2] = 0.4 # force upward action
                action = np.concatenate((action, np.asarray([theta_action,0.7,0.])))
            elif not env.is_drawer_opened("bottom", widely=drawer_never_opened):
                # print("opening drawer")
                action = np.array([0, -1.0, 0])
                # action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0.01, 0., -0.01])))
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
                if xy_diff > 0.75 * dist_thresh:
                    action[2] = 0.5
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif env._gripper_open:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif object_gripper_dist > 2 * dist_thresh:
                # Open gripper to retry
                action = np.array([0, 0, 0, 0, 0.7, 0])
            elif not object_lifted_with_margin:
                # print('raise object upward')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
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
            # print("rew", rew)

            img = env.render_obs()
            # save_video:
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
            # print("i", i)
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)
            images_for_gif[0].save('data/grasp_place_{}.gif'.format(i),
                save_all=True, append_images=images_for_gif[1:],
                duration=env.scripted_traj_len * 2, loop=0)

        # print('object pos: {}'.format(object_pos))
        # print('reward: {}'.format(rew))
        # print('--------------------')

        if save_video:
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)

def close_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    for i in range(50):
        obs = env.reset()

        reached_pushing_region = False

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            top_drawer_pos = env.get_drawer_bottom_pos("top")
            top_drawer_push_target_pos = (top_drawer_pos +
                np.array([0, -0.15, 0.02]))

            is_gripper_ready_to_push = (ee_pos[1] < top_drawer_push_target_pos[1] and
                ee_pos[2] < top_drawer_push_target_pos[2])
            theta_action = 0.

            if (not env.is_drawer_closed("top") and not reached_pushing_region and
                not is_gripper_ready_to_push):
                # print("move up and left")
                action = np.concatenate(
                    ([-0.2, -0.4, -0.2], np.array([theta_action, 0, 0])))
            elif not env.is_drawer_closed("top"):
                # print("close top drawer")
                reached_pushing_region = True
                action = (top_drawer_pos - ee_pos) * 7.0
                action[0] *= 3
                action[1] *= 0.6
                action = np.concatenate((action, np.array([theta_action, 0, 0])))
            else:
                # print("Move toward neutral")
                action = (ending_target_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))

            noise_scalings = [noise] * 3 + [0.1 * noise] + [noise] * 2
            action += np.random.normal(scale=noise_scalings)
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print(action)
            obs, rew, done, info = env.step(action)
            # print("rew", rew)

            time.sleep(0.05)

        if rew > 0:
            print("i", i)
            print('reward: {}'.format(rew))
            print('--------------------')

def open_grasp_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    for i in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)
        drawer_never_opened = True

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            bottom_drawer_handle_pos = env.get_bottom_drawer_handle_pos()
            object_lifted_with_margin = object_pos[2] > (
                env._reward_height_thresh + margin)

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            gripper_handle_dist = np.linalg.norm(bottom_drawer_handle_pos - ee_pos)
            theta_action = 0.

            if (gripper_handle_dist > dist_thresh
                and not env.is_drawer_opened("bottom", widely=drawer_never_opened)):
                # print('approaching handle')
                handle_pos_offset = np.zeros((3,))
                if np.abs(ee_pos[0] - bottom_drawer_handle_pos[0]) > dist_thresh:
                    handle_pos_offset = np.array([0, -0.03, 0])
                action = (bottom_drawer_handle_pos + handle_pos_offset - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > dist_thresh:
                    action[2] = 0.4 # force upward action
                action = np.concatenate((action, np.asarray([theta_action,0.7,0.])))
            elif not env.is_drawer_opened("bottom", widely=drawer_never_opened):
                # print("opening drawer")
                action = np.array([0, -1.0, 0])
                # action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0, 0., 0])))
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
                if xy_diff > 0.75 * dist_thresh:
                    action[2] = 0.5
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif env._gripper_open:
                # print('gripper closing')
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif object_gripper_dist > 2 * dist_thresh:
                # Open gripper to retry
                action = np.array([0, 0, 0, 0, 0.7, 0])
            elif not object_lifted_with_margin:
                # print('raise object upward')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
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
            # print("rew", rew)

            time.sleep(0.05)

        if rew > 0:
            print("i", i)
            print('reward: {}'.format(rew))
            print('--------------------')

if __name__ == "__main__":
    EPSILON = 0.05
    noise = 0.2
    margin = 0.025
    save_video = True

    mode = ["CloseOpenGrasp", "Close", "OpenGrasp"][2]

    gui = True
    reward_type = "sparse"
    obs_mode = "pixels_debug"

    ending_target_pos = np.array([0.73822169, -0.03909928, -0.25635483])
    if mode == "CloseOpenGrasp":
        env = roboverse.make("Widow200GraspV6DoubleDrawerV0CloseOpenGrasp-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        close_open_grasp_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "Close":
        env = roboverse.make("Widow200GraspV6DoubleDrawerV0Close-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        close_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "OpenGrasp":
        env = roboverse.make("Widow200GraspV6DoubleDrawerV0OpenGrasp-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        open_grasp_policy(EPSILON, noise, margin, save_video, env)
