from roboverse.envs.widow200_grasp_v6_drawer_open_v0 import (
    Widow200GraspV6DrawerOpenV0Env, drawer_open_policy)
from roboverse.envs.rand_obj import RandObjEnv
from roboverse.utils.shapenet_utils import load_shapenet_object, \
    import_shapenet_metadata
import roboverse.bullet as bullet
from roboverse.bullet.objects import small_tray_center_pos
import roboverse.utils as utils
import numpy as np
import time
import roboverse
from PIL import Image
import skimage.io as skii
import skimage.transform as skit
import itertools

obj_path_map, path_scaling_map = import_shapenet_metadata()

class Widow200GraspV6DrawerOpenThenPlaceV0Env(Widow200GraspV6DrawerOpenV0Env):
    """
    Setup: blocking_obj blocking the drawer from being opened. 
    obj in closed drawer.
    Task: grasp blocking_obj, put in box above drawer.
    open drawer. grasp obj.
    """

    def __init__(self,
                 *args,
                 object_names=("ball",),
                 scaling_local_list=[0.5],
                 success_dist_threshold=0.04,
                 noisily_open_drawer=False,
                 task_type="OpenPickPlace",
                 **kwargs):
        assert len(object_names) == 1
        self.object_name = object_names[0]
        self.noisily_open_drawer = noisily_open_drawer

        self.task_type = task_type
        assert self.task_type in ["OpenPickPlace", "Open", "PickPlace"]
        self.box_high = small_tray_center_pos + np.array([0.0525, 0.04, 0.035])
        self.box_low = small_tray_center_pos + np.array([-0.0525, -0.04, -0.01])

        self._success_dist_threshold = success_dist_threshold

        super().__init__(*args, object_names=object_names,
            scaling_local_list=scaling_local_list, **kwargs)

        margin = 0.0225
        self._object_position_high = np.array(list(self.box_high[:2] - margin) + [-0.2])
        self._object_position_low = np.array(list(self.box_low[:2] + margin) + [-0.2])

        drawer_urdf_size = np.array([1.4, 1.2, 0])
        half_drawer_dims = 0.5 * 0.1 * drawer_urdf_size
        half_drawer_z_dim = 0.5 * np.array([0, 0, drawer_urdf_size[2]])
        # drawer high and low offsets from the drawer bottom.
        self.drawer_high_offset = half_drawer_dims + np.array([0, 0, 0.05])
        self.drawer_low_offset = -1 * half_drawer_dims + np.array([0, 0, -0.05])

        self._env_name = "Widow200GraspV6DrawerOpenThenPlaceV0Env"
        task_scripted_traj_len_map = {
            "Open": 30,
            "PickPlace": 30,
            "OpenPickPlace": 60,
        }
        self.scripted_traj_len = task_scripted_traj_len_map[self.task_type]

    def _load_meshes(self):
        self._robot_id = bullet.objects.widowx_200()
        self._table = bullet.objects.table()
        self._workspace = bullet.Sensor(self._robot_id,
                                        xyz_min=self._pos_low,
                                        xyz_max=self._pos_high,
                                        visualize=False, rgba=[0, 1, 0, .1])
        self._tray = bullet.objects.widow200_hidden_tray()
        self._objects = {}
        self._drawer = bullet.objects.drawer_with_tray_inside()
        object_position = np.random.uniform(
            self._object_position_low, self._object_position_high)

        # For some reason this is important for the drawer to be openable by the robot.
        bullet.close_drawer(self._drawer)
        # End

        if self.task_type == "PickPlace":
            bullet.open_drawer(self._drawer, noisy_open=self.noisily_open_drawer)

        self.object_tray = bullet.objects.small_object_tray()
        self.load_object(self.object_name, object_position)

    def load_object(self, name, pos, quat=[1, -1, 0, 0]):
        self._objects[name] = load_shapenet_object(
            obj_path_map[name], self.scaling,
            pos, scale_local=self._scaling_local[name], quat=quat)

    def get_box_pos(self):
        box_open_top_info = bullet.get_body_info(self._box, quat_to_deg=False)
        return np.asarray(box_open_top_info['pos'])

    def get_object_pos(self):
        object_info = bullet.get_body_info(
            self._objects[self.object_name], quat_to_deg=False)
        return np.asarray(object_info['pos'])

    def is_object_grasped(self):
        object_gripper_distance = np.linalg.norm(
            self.get_object_pos() - self.get_end_effector_pos())
        return (self.get_object_pos()[2] > self._reward_height_thresh) and (object_gripper_distance < 0.1)

    def object_within_drawer_bounds(self):
        """Returns a boolean vector."""
        object_pos = self.get_object_pos()
        drawer_low = self.get_drawer_bottom_pos() + self.drawer_low_offset
        drawer_high = self.get_drawer_bottom_pos() + self.drawer_high_offset
        object_within_drawer_bounds = ((drawer_low <= object_pos)
            & (object_pos <= drawer_high))
        return object_within_drawer_bounds

    def is_object_in_drawer(self):
        return np.all(self.object_within_drawer_bounds())

    def get_reward(self, info):
        reward = float(self.is_object_in_drawer())
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

    def get_info(self):
        info = {}

        # object pos
        object_info = bullet.get_body_info(
            self._objects[self.object_name], quat_to_deg=False)
        object_pos = np.asarray(object_info['pos'])

        ee_pos = np.array(self.get_end_effector_pos())
        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        info['object_gripper_dist'] = object_gripper_dist

        info['is_drawer_opened'] = int(self.is_drawer_opened())
        info['is_drawer_opened_widely'] = int(self.is_drawer_opened(widely=True))
        info['drawer_y_pos'] = self.get_drawer_bottom_pos()[1] # y coord

        # gripper-handle dist
        ee_pos = np.array(self.get_end_effector_pos())
        gripper_handle_dist = np.linalg.norm(ee_pos - self.get_handle_pos())
        info['gripper_handle_dist'] = gripper_handle_dist

        # object above drawer:
        object_above_drawer = np.all(self.object_within_drawer_bounds()[:2])
        info['object_above_drawer_success'] = float(object_above_drawer)

        # object in drawer:
        info['object_in_drawer_success'] = float(self.is_object_in_drawer())

        return info


class Widow200GraspV6DrawerOpenThenPlaceV0PickPlaceOnlyEnv(Widow200GraspV6DrawerOpenThenPlaceV0Env):
    """
    Setup: blocking_obj blocking the drawer from being opened.
    obj in closed drawer.
    Task: grasp blocking_obj, put in box.
    """

    def __init__(self, *args, task_type="PickPlace", **kwargs):
        super().__init__(*args, task_type=task_type, **kwargs)


class Widow200GraspV6DrawerOpenThenPlaceV0OpenOnlyEnv(Widow200GraspV6DrawerOpenThenPlaceV0Env):
    """
    Setup: blocking_obj blocking the drawer from being opened.
    obj in closed drawer.
    Task: grasp blocking_obj, put in box.
    Same as PickPlaceOnly, but with 40 timesteps.
    """

    def __init__(self, *args, task_type="Open", **kwargs):
        super().__init__(*args, task_type=task_type, **kwargs)

    def get_reward(self, info):
        reward = float(self.is_drawer_opened(widely=True))
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

def drawer_open_then_place_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    for i in range(200):
        obs = env.reset()
        # print("env.scripted_traj_len", env.scripted_traj_len)

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        drawer_dist_thresh = 0.035 + np.random.normal(scale=0.01)
        drawer_dist_thresh = np.clip(drawer_dist_thresh, 0.025, 0.05)

        object_thresh = 0.04 + np.random.normal(scale=0.01)
        object_thresh = np.clip(object_thresh, 0.030, 0.050)

        drawer_never_opened = True

        images, images_for_gif = [], [] # new video at the start of each trajectory

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            drawer_pos = env.get_drawer_bottom_pos()
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            handle_pos = env.get_handle_pos()
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
            object_drawer_xy_dist = np.linalg.norm(object_pos[:2] - drawer_pos[:2])
            theta_action = 0.

            info = env.get_info()

            if (gripper_handle_dist > dist_thresh
                and not env.is_drawer_opened(widely=drawer_never_opened)):
                # print('approaching handle')
                action = (handle_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.75 * dist_thresh:
                    action[2] = 0.0 # force upward action to avoid upper box
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif not env.is_drawer_opened(widely=drawer_never_opened):
                # print("opening drawer")
                action = np.array([0, -1.0, 0])
                # action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif (object_gripper_dist > object_thresh
                and env._gripper_open and gripper_handle_dist < 1.5 * dist_thresh):
                # print("Lift upward")
                drawer_never_opened = False
                if ee_pos[2] < -.15:
                    action = env.gripper_goal_location - ee_pos
                    action[2]  = 0.7  # force upward action to avoid upper box
                else:
                    action = env.gripper_goal_location - ee_pos
                    action *= 7.0
                    action[2]  *= 0.5  # force upward action to avoid upper box
                action = np.concatenate(
                    (action, np.asarray([theta_action, 0., 0.])))
            elif ((object_gripper_dist > dist_thresh) and
                env._gripper_open and not info['object_above_drawer_success']):
                # print('approaching')
                # print("object_pos", object_pos)
                action = (object_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > dist_thresh:
                    action[2] = 0.3
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif (env._gripper_open and object_drawer_xy_dist > dist_thresh and
                not env.is_object_in_drawer()):
                # print("close gripper")
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif object_drawer_xy_dist > drawer_dist_thresh:
                # print("move_to_drawer")
                action = (drawer_pos - object_pos)*7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                action[2] = 0.2
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
                # print("object_pos", object_pos)
            elif not env.is_object_in_drawer():
                # object is now above the drawer.
                # print("gripper opening")
                action = np.array([0., 0., 0., 0., 0.7, 0.])
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


def drawer_pick_place_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    for i in range(200):
        obs = env.reset()
        print("env.scripted_traj_len", env.scripted_traj_len)

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        drawer_dist_thresh = 0.035 + np.random.normal(scale=0.01)
        drawer_dist_thresh = np.clip(drawer_dist_thresh, 0.025, 0.05)

        object_thresh = 0.04 + np.random.normal(scale=0.01)
        object_thresh = np.clip(object_thresh, 0.030, 0.050)

        images, images_for_gif = [], [] # new video at the start of each trajectory

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            drawer_pos = env.get_drawer_bottom_pos()
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            handle_pos = env.get_handle_pos()
            object_lifted_with_margin = object_pos[2] > (
                env._reward_height_thresh + margin)
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
            object_drawer_dist = np.linalg.norm(object_pos - drawer_pos)
            theta_action = 0.

            info = env.get_info()
            z_diff = abs(object_pos[2] - ee_pos[2])

            # if (object_gripper_dist > object_thresh
            #     and env._gripper_open and gripper_handle_dist < 1.5 * dist_thresh):
            #     # print("Lift upward")
            #     drawer_never_opened = False
            #     if ee_pos[2] < -.15:
            #         action = env.gripper_goal_location - ee_pos
            #         action[2]  = 0.7  # force upward action to avoid upper box
            #     else:
            #         action = env.gripper_goal_location - ee_pos
            #         action *= 7.0
            #         action[2]  *= 0.5  # force upward action to avoid upper box
            #     action = np.concatenate(
            #         (action, np.asarray([theta_action, 0., 0.])))
            if ((object_gripper_dist > dist_thresh or z_diff > 0.015) and
                env._gripper_open and not info['object_above_drawer_success']):
                # print('approaching')
                action = (object_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.03:
                    action[2] *= 0.3
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif (env._gripper_open and object_drawer_dist > dist_thresh and
                not env.is_object_in_drawer()):
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif (object_drawer_dist > dist_thresh and
                not info['object_above_drawer_success']):
                action = (drawer_pos - object_pos)*7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if "DrawerPlaceThenOpen" or "DrawerOpenThenPlace" in env._env_name:
                    # print("don't droop down until xy-close to box")
                    action[2] = 0.2
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
                # print("object_pos", object_pos)
            elif not env.is_object_in_drawer():
                # object is now above the drawer.
                # print("gripper opening")
                action = np.array([0., 0., 0., 0., 0.7, 0.])
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


if __name__ == "__main__":
    EPSILON = 0.05
    noise = 0.2
    margin = 0.025
    save_video = True

    mode = "OpenThenPlace"

    gui = False
    reward_type = "sparse"
    obs_mode = "pixels_debug"
    if mode == "OpenThenPlace":
        env = roboverse.make("Widow200GraspV6DrawerOpenThenPlaceV0-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_open_then_place_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "Open":
        NotImplementedError
    elif mode == "PickPlaceOnly":
        env = roboverse.make("Widow200GraspV6DrawerOpenThenPlaceV0PickPlaceOnly-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_pick_place_policy(EPSILON, noise, margin, save_video, env)
    else:
        raise NotImplementedError
