from roboverse.envs.sawyer_grasp import SawyerGraspOneEnv
import roboverse.bullet as bullet
import numpy as np

class SawyerReachEnv(SawyerGraspOneEnv):

    def get_reward(self, info):

        if self._reward_type == 'sparse':
            if info['object_gripper_distance'] < 0.03:
                reward = 1
            else:
                reward = 0
        elif self._reward_type == 'shaped':
            reward = -1*info['object_gripper_distance']
            reward = max(reward, self._reward_min)
        else:
            raise NotImplementedError

        return reward


class SawyerReachEnvMultiObj(SawyerReachEnv):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        self.cur_obj_index = 0
        super().__init__(*args, **kwargs)


    def _load_meshes(self):
        self._fixed_object_position = [(.75, .2, -.36), (.65, .25, -.33), (.55, .1, -.4)]

        self._sawyer = bullet.objects.sawyer_finger_visual_only()
        self._table = bullet.objects.table()
        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._sawyer,
                                        xyz_min=self._pos_low, xyz_max=self._pos_high,
                                        visualize=False, rgba=[0, 1, 0, .1])
        if self._randomize:
            object_position = np.random.uniform(
                low=self._object_position_low, high=self._object_position_high, size = [3, 3])
        else:
            object_position = self._fixed_object_position
        self._objects = {
            'lego': bullet.objects.lego(pos=object_position[0]),
            'duck':bullet.objects.duck(pos=object_position[1]),
            'cube':bullet.objects.cube(pos=object_position[2])
        }
        self._objects_list = ['lego', 'duck', 'cube']

    def change_obj(self, index):
        self.cur_obj_index = index