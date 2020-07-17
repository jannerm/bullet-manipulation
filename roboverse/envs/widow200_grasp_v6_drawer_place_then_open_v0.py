from roboverse.envs.widow200_grasp_v6_drawer_open_v0 import (
    Widow200GraspV6DrawerOpenV0Env, drawer_open_policy)
from roboverse.envs.rand_obj import RandObjEnv
from roboverse.utils.shapenet_utils import load_shapenet_object, \
    import_shapenet_metadata
import roboverse.bullet as bullet
import roboverse.utils as utils
import numpy as np
import time
import roboverse

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
                 blocking_object_name_scaling=("shed", 0.3),
                 success_dist_threshold=0.04,
                 noisily_open_drawer=False,
                 close_drawer_on_reset=True,
                 open_only=False,
                 **kwargs):
        self.noisily_open_drawer = noisily_open_drawer
        self.close_drawer_on_reset = close_drawer_on_reset
        self.object_name = object_name_scaling[0]
        self.blocking_object_name = blocking_object_name_scaling[0]

        assert self.object_name != self.blocking_object_name
        object_names = (self.object_name, self.blocking_object_name)
        scaling_local_list = [
            object_name_scaling[1], blocking_object_name_scaling[1]]
        print("scaling_local_list", scaling_local_list)
        num_objects = 2

        self._object_position_high = (.82, -.08, -.29)
        self._object_position_low = (.82, -.09, -.29)

        # Blocking = Obstruction object
        blocking_object_offset = np.array([0, -0.025, -0.05])
        self._blocking_object_position_high = self._object_position_high + blocking_object_offset
        self._blocking_object_position_low = self._object_position_low + blocking_object_offset
        self._success_dist_threshold = success_dist_threshold
        # self._scaling_local_list = scaling_local_list
        # self.set_scaling_dicts()
        # self.obs_img_dim = 228

        super().__init__(*args,
            object_names=object_names,
            scaling_local_list=scaling_local_list,
            num_objects=num_objects, **kwargs)
        self._env_name = "Widow200GraspV6DrawerOpenV0Env"

        self.box_high = np.array([0.895, .09, -.26]) # Double check this!!!
        self.box_low = np.array([0.79, 0.01, -.305])

        self.scripted_traj_len = 80

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
        print("pre generate object pos")
        object_position = np.random.uniform(
            self._object_position_low, self._object_position_high)
        blocking_object_position = np.random.uniform(
            self._blocking_object_position_low,
            self._blocking_object_position_high)
        object_positions = np.concatenate(
            (object_position, blocking_object_position), axis=0)

        self.load_object(self.object_name, object_position)

        if self.close_drawer_on_reset:
            bullet.close_drawer(self._drawer)

        self.load_object(self.blocking_object_name, blocking_object_position)

        self._box = bullet.objects.lifted_long_box_open_top()

    def load_object(self, name, pos):
        self._objects[name] = load_shapenet_object(
            obj_path_map[name], self.scaling,
            pos, scale_local=self._scaling_local[name])

    def get_box_pos(self):
        box_open_top_info = bullet.get_body_info(self._box, quat_to_deg=False)
        return np.asarray(box_open_top_info['pos'])

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

def drawer_place_then_open_policy(EPSILON, noise, margin, save_video, env):
    object_ind = 0
    blocking_object_ind = 1
    for i in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        box_dist_thresh = 0.035 + np.random.normal(scale=0.01)
        box_dist_thresh = np.clip(box_dist_thresh, 0.025, 0.05)

        drawer_never_opened = True

        images = [] # new video at the start of each trajectory.

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

            if (blocking_object_gripper_dist > dist_thresh and
                env._gripper_open and not info['blocking_object_above_box_success']):
                # print('approaching')
                action = ((blocking_object_pos +
                    blocking_object_pos_offset) - ee_pos) * 7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if "Drawer" in env._env_name:
                    if xy_diff > dist_thresh:
                        action[2] = 0.4 # force upward action to avoid upper box
                else:
                    if xy_diff > 0.02:
                        action[2] = 0.0
                action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
            elif env._gripper_open and blocking_object_box_dist > box_dist_thresh:
                # print('gripper closing')
                action = (blocking_object_pos - ee_pos) * 7.0
                action = np.concatenate(
                    (action, np.asarray([0., -0.7, 0.])))
            elif blocking_object_box_dist > box_dist_thresh:
                action = (box_pos - blocking_object_pos)*7.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if "DrawerPlaceThenOpen" in env._env_name:
                    print("don't droop down until xy-close to box")
                    if xy_diff > dist_thresh:
                        action[2] = 0.0
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0.])))
            elif not info['blocking_object_in_box_success']:
                # object is now above the box.
                action = (box_pos - blocking_object_pos)*7.0
                action = np.concatenate(
                    (action, np.asarray([0., 0.7, 0.])))
            elif (gripper_handle_dist > dist_thresh
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
            if save_video:
                images.append(img)

            time.sleep(0.05)

        print('object pos: {}'.format(object_pos))
        print('reward: {}'.format(rew))
        print('--------------------')

        if save_video:
            utils.save_video('data/grasp_place_{}.avi'.format(i), images)

if __name__ == "__main__":
    EPSILON = 0.05
    noise = 0.2
    margin = 0.025
    save_video = True

    mode = "PlaceThenOpen"

    gui = True
    reward_type = "sparse"
    obs_mode = "pixels_debug"
    if mode == "PlaceThenOpen":
        env = roboverse.make("Widow200GraspV6DrawerPlaceThenOpenV0-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_place_then_open_policy(EPSILON, noise, margin, save_video, env)
    else:
        raise NotImplementedError
