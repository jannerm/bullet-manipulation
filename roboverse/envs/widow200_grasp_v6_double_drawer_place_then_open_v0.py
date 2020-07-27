from roboverse.envs.widow200_grasp_v6_drawer_place_then_open_v0 import (
    Widow200GraspV6DrawerPlaceThenOpenV0Env, drawer_open_policy)
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

class Widow200GraspV6DoubleDrawerPlaceThenOpenV0Env(Widow200GraspV6DrawerPlaceThenOpenV0Env):
    """
    Setup: blocking_obj blocking the bottom drawer from being opened. 
    obj in closed drawer.
    Task: grasp blocking_obj, put in box above drawer.
    open drawer. grasp obj.
    """

    def __init__(self,
                 *args,
                 task_type="PickPlaceOpenGrasp",
                 **kwargs):
        super().__init__(*args, task_type=task_type, **kwargs)
        self._object_position_high = (.84, -.05, -.29)
        self._object_position_low = (.84, -.06, -.29)

        if self.task_type in ["OpenGrasp", "Grasp", "Open"]:
            # Only open and/or grasping required. So blocking object is dropped in box.
            # Drop the object in the box.
            margin = np.array([0.03, 0.03, 0])
            drop_height = self.box_high[2]
            self._blocking_object_position_high = list(self.box_high[:2]) + [drop_height]
            self._blocking_object_position_high -= margin
            self._blocking_object_position_low = list(self.box_low[:2]) + [drop_height]
            self._blocking_object_position_low += margin
        else:
            # The task starts with needing to place blocking object in the box.
            # Blocking = Obstruction object
            # Drop object blocking drawer
            # blocking_object_offset = np.array([0.02, -0.05, -0.05])
            self._blocking_object_position_high = np.array([0.86, -0.1, -0.34])
            self._blocking_object_position_low = np.array([0.86, -0.11, -0.34])

        self._env_name = "Widow200GraspV6DoubleDrawerPlaceThenOpenV0Env"

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
        self._bottom_drawer = bullet.objects.drawer_bottom()
        self._top_drawer = bullet.objects.drawer_no_handle()
        self.drawers = {"top": self._top_drawer, "bottom": self._bottom_drawer}
        bullet.open_drawer(self._bottom_drawer, noisy_open=self.noisily_open_drawer)
        object_position = np.random.uniform(
            self._object_position_low, self._object_position_high)
        blocking_object_position = np.random.uniform(
            self._blocking_object_position_low,
            self._blocking_object_position_high)
        object_positions = np.concatenate(
            (object_position, blocking_object_position), axis=0)

        self.load_object(self.object_name, object_position)

        if self.task_type not in ["Grasp"]:
            bullet.close_drawer(self._bottom_drawer)

        if self.task_type in ["OpenGrasp", "Grasp", "Open"] or self.randomize_blocking_obj_quat:
            self.blocking_obj_quat = self.get_random_quat()
        else:
            self.blocking_obj_quat = [1, -1, 0, 0]

        self.load_object(
            self.blocking_object_name, blocking_object_position, quat=self.blocking_obj_quat)

        self._box = bullet.objects.lifted_long_box_open_top()

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

        # Drawer
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

class Widow200GraspV6DoubleDrawerPlaceThenOpenV0PickPlaceOnlyEnv(Widow200GraspV6DoubleDrawerPlaceThenOpenV0Env):
    """
    Setup: blocking_obj blocking the drawer from being opened.
    obj in closed drawer.
    Task: grasp blocking_obj, put in box.
    """

    def __init__(self, *args, task_type="PickPlace", **kwargs):
        super().__init__(*args, task_type=task_type, **kwargs)

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        reward = float(info['blocking_object_in_box_success'])
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

class Widow200GraspV6DoubleDrawerPlaceThenOpenV0OpenOnlyEnv(Widow200GraspV6DoubleDrawerPlaceThenOpenV0Env):
    def __init__(self, *args, task_type="Open", **kwargs):
        super().__init__(*args, task_type=task_type, **kwargs)

    def get_reward(self, info):
        reward = float(self.is_drawer_opened(widely=True))
        reward = self.adjust_rew_if_use_positive(reward)
        return reward

class Widow200GraspV6DoubleDrawerPlaceThenOpenV0GraspOnlyEnv(Widow200GraspV6DoubleDrawerPlaceThenOpenV0Env):
    def __init__(self, *args, task_type="Grasp", **kwargs):
        super().__init__(*args, task_type=task_type, **kwargs)

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
        env = roboverse.make("Widow200GraspV6DoubleDrawerPlaceThenOpenV0-v0",
                             gui=gui,
                             reward_type=reward_type,
                             observation_mode=obs_mode)
        drawer_place_then_open_policy(EPSILON, noise, margin, save_video, env)
    elif mode == "PickPlaceOnly":
        raise NotImplementedError
    elif mode == "PickPlace40Only":
        raise NotImplementedError
    elif mode == "OpenGraspOnly":
        raise NotImplementedError
    else:
        raise NotImplementedError
