from roboverse.envs.widow200_grasp_v7 import Widow200GraspV7Env
from roboverse.envs.rand_obj import RandObjEnv
from roboverse.envs.env_object_list import (
    POSSIBLE_TRAIN_OBJECTS, POSSIBLE_TRAIN_SCALINGS,
    POSSIBLE_TEST_OBJECTS, POSSIBLE_TEST_SCALINGS)
import roboverse.bullet as bullet
import roboverse.utils as utils
import numpy as np


class Widow200GraspV7BoxV0Env(Widow200GraspV7Env):
    """
    Deterministic object, non-terminating grasping env with
    a box in the tray. Like GraspV6, the goal is still only
    to grasp the object and lift it above a certain height.
    """

    def __init__(self,
                 *args,
                 object_names=('gatorade',),
                 scaling_local_list=[0.5],
                 success_dist_threshold=0.04,
                 **kwargs):
        super().__init__(*args,
            object_names=object_names,
            scaling_local_list=scaling_local_list,
            **kwargs)
        self._object_position_high = (.82, -.07, -.20)
        self._object_position_low = (.78, -.125, -.20)
        self._success_dist_threshold = success_dist_threshold

    def _load_meshes(self):
        super()._load_meshes()
        self._box = bullet.objects.long_box_open_top()