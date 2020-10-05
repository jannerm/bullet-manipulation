from roboverse.envs.widow200_grasp_v6_box_v0 import Widow200GraspV6BoxV0Env
import roboverse.bullet as bullet
import roboverse.utils as utils
import numpy as np
import time
import roboverse
from PIL import Image
import skimage.io as skii
import skimage.transform as skit


class Widow200GraspV6DrawerOpenV0Env(Widow200GraspV6BoxV0Env):
    """Task is to open drawer, then grasp object inside it."""
    def __init__(self,
                 *args,
                 object_names=('ball',),
                 scaling_local_list=[0.5],
                 success_dist_threshold=0.04,
                 noisily_open_drawer=False,
                 close_drawer_on_reset=True,
                 open_only=False,
                 **kwargs):
        camera_target_pos = [1.05, -0.05, -0.1]
        camera_pitch = -50
        self.noisily_open_drawer = noisily_open_drawer
        # When True, drawer does not open all the way
        self.close_drawer_on_reset = close_drawer_on_reset

        super().__init__(*args,
            object_names=object_names,
            scaling_local_list=scaling_local_list,
            camera_target_pos=camera_target_pos,
            camera_pitch=camera_pitch,
            **kwargs)
        self._env_name = "Widow200GraspV6DrawerOpenV0Env"

        self.open_only = open_only

        object_pos_offset = np.zeros((3,))
        if not self.close_drawer_on_reset:
            # Grasp only.
            object_pos_offset[1] = np.random.uniform(0, 0.02)
            # drop object more to the right
            # because of noisy open positions
        self._object_position_high = (.84, -.08, -.29) + object_pos_offset
        self._object_position_low = (.84, -.09, -.29) + object_pos_offset

        self._success_dist_threshold = success_dist_threshold
        # self._scaling_local_list = scaling_local_list
        # self.set_scaling_dicts()
        # self.obs_img_dim = 228

        self.scripted_traj_len = 50

        if not self.close_drawer_on_reset:
            # self._object_position_high = (.82, -.07, -.29)
            # self._object_position_low = (.82, -.07, -.29)
            self.scripted_traj_len = 25 # Give less time if drawer starts opened.
        if self.open_only:
            assert self.close_drawer_on_reset
            # If task is only to open drawer, the drawer better start out closed
            self.scripted_traj_len = 30

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
        bullet.open_drawer(self._drawer, noisy_open=self.noisily_open_drawer)

        object_positions = self._generate_object_positions()
        self._load_objects(object_positions)

        if self.close_drawer_on_reset:
            bullet.close_drawer(self._drawer)

        self._box = bullet.objects.lifted_long_box_open_top()

    def get_drawer_bottom_pos(self):
        link_names = [bullet.get_joint_info(self._drawer, j, 'link_name')
            for j in range(bullet.p.getNumJoints(self._drawer))]
        drawer_bottom_link_idx = link_names.index('base')
        drawer_bottom_pos = bullet.get_link_state(
            self._drawer, drawer_bottom_link_idx, "pos")
        return np.array(drawer_bottom_pos)

    def get_handle_pos(self):
        link_names = [bullet.get_joint_info(self._drawer, j, 'link_name')
            for j in range(bullet.p.getNumJoints(self._drawer))]
        handle_link_idx = link_names.index('handle_r')
        handle_pos = bullet.get_link_state(
            self._drawer, handle_link_idx, "pos")
        return np.array(handle_pos)

    def is_drawer_opened(self, widely=False):
        opened_thresh = 0.0 if not widely else -0.05
        return self.get_drawer_bottom_pos()[1] < opened_thresh

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

        info['is_drawer_opened'] = int(self.is_drawer_opened())
        info['is_drawer_opened_widely'] = int(self.is_drawer_opened(widely=True))
        info['drawer_y_pos'] = self.get_drawer_bottom_pos()[1] # y coord

        # gripper-handle dist
        ee_pos = np.array(self.get_end_effector_pos())
        gripper_handle_dist = np.linalg.norm(ee_pos - self.get_handle_pos())
        info['gripper_handle_dist'] = gripper_handle_dist
        return info


def drawer_open_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    for i in range(300):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)
        drawer_never_opened = True

        images, images_for_gif = [], [] # new video at the start of each trajectory.

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            handle_pos = env.get_handle_pos()
            object_lifted_with_margin = object_pos[2] > (
                env._reward_height_thresh + margin)
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
            theta_action = 0.

            if (gripper_handle_dist > dist_thresh
                and not env.is_drawer_opened(widely=drawer_never_opened)):
                # print('approaching handle')
                action = (handle_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.75 * dist_thresh:
                    action[2] = 0.5 # force upward action to avoid upper box
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
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

        # if rew > 0:
        #     print("i", i)
        #     print('reward: {}'.format(rew))
        #     print('--------------------')

        if save_video:
            print("i", i)
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)
            images_for_gif[0].save('data/grasp_place_{}.gif'.format(i),
                save_all=True, append_images=images_for_gif[1:],
                duration=env.scripted_traj_len * 2, loop=0)

        # print('object pos: {}'.format(object_pos))
        # print('reward: {}'.format(rew))
        # print('--------------------')

        if save_video:
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)

def drawer_open_only_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    for i in range(50):
        obs = env.reset()
        print("obs", obs)
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)
        drawer_never_opened = True

        images = [] # new video at the start of each trajectory.

        for _ in range(env.scripted_traj_len):
            state_obs = obs[env.fc_input_key]
            obj_obs = obs[env.object_obs_key]
            ee_pos = state_obs[:3]
            object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
            handle_offset = np.array([0, -0.01, 0])
            handle_pos = env.get_handle_pos() + handle_offset
            # Make robot aim a little to the left of the handle
            ending_target_pos = np.array([0.73822169, -0.03909928, -0.25635483]) # Effective neutral pos.
            object_lifted_with_margin = object_pos[2] > (
                env._reward_height_thresh + margin)
            # object_pos += np.random.normal(scale=0.02, size=(3,))

            gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
            theta_action = 0.

            if (gripper_handle_dist > dist_thresh
                and not env.is_drawer_opened(widely=drawer_never_opened)):
                # print('approaching handle')
                action = (handle_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.75 * dist_thresh:
                    action[2] = 0.5 # force upward action to avoid upper box
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif not env.is_drawer_opened(widely=drawer_never_opened):
                # print("opening drawer")
                action = np.array([0, -1.0, 0])
                # action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif np.abs(ee_pos[2] - ending_target_pos[2]) > dist_thresh:
                # print("Lift upward")
                drawer_never_opened = False
                action = np.array([0, 0, 0.7]) # force upward action to avoid upper box
                action = np.concatenate(
                    (action, np.asarray([theta_action, 0., 0.])))
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

            img = env.render_obs()
            if save_video:
                images.append(img)

            time.sleep(0.05)

        print('object pos: {}'.format(object_pos))
        print('reward: {}'.format(rew))
        print('--------------------')

        if save_video:
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)

def drawer_grasping_only_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    margin = 0.025

    for i in range(50):
        obs = env.reset()

        dist_thresh = 0.045 + np.random.normal(scale=0.01)
        dist_thresh = np.clip(dist_thresh, 0.035, 0.060)

        for _ in range(env.scripted_traj_len):

            if isinstance(obs, dict):
                object_pos = obs[env.object_obs_key][
                             object_ind * 7 : object_ind * 7 + 3]
                ee_pos = obs[env.fc_input_key][:3]
            else:
                object_pos = obs[
                             object_ind * 7 + 8: object_ind * 7 + 8 + 3]
                ee_pos = obs[:3]

            object_lifted_with_margin = object_pos[2] > (env._reward_height_thresh + margin)

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.

            if object_gripper_dist > dist_thresh and env._gripper_open:
                # print('approaching')
                action = (object_pos - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2] / 7.0)
                if "Drawer" in env._env_name:
                    if xy_diff > dist_thresh:
                        action[2] = 0.4 # force upward action to avoid upper box
                else:
                    if xy_diff > 0.02:
                        action[2] = 0.0
                action = np.concatenate(
                    (action, np.asarray([theta_action, 0., 0.])))
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
            else:
                # Move above tray's xy-center.
                # print("done")
                tray_info = roboverse.bullet.get_body_info(
                    env._tray, quat_to_deg=False)
                tray_center = np.asarray(tray_info['pos'])
                action = (tray_center - ee_pos)[:2]
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0., 0.])))

            # action += np.random.normal(scale=noise, size=(6,))
            action[:3] += np.random.normal(scale=noise, size=(3,))
            action[3] += np.random.normal(scale=noise*0.1)
            action[4:] += np.random.normal(scale=noise, size=(2,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            # print("action", action)

            obs, reward, done, info = env.step(action)

            if done:
                break

        print("reward:", reward)

if __name__ == "__main__":
    EPSILON = 0.05
    noise = 0.2
    margin = 0.025
    save_video = True

    mode = "Open"

    gui = True
    reward_type = "sparse"
    obs_mode = "pixels_debug"
    if mode == "Open":
        env = roboverse.make("Widow200GraspV6DrawerOpenV0-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_open_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "OpenOnly":
        env = roboverse.make("Widow200GraspV6DrawerOpenOnlyV0-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_open_only_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "GraspOnly":
        env = roboverse.make("Widow200GraspV6DrawerGraspOnlyV0-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode,
                             noisily_open_drawer=True)
        drawer_grasping_only_policy(EPSILON, noise, margin, save_video, env)
    else:
        raise NotImplementedError