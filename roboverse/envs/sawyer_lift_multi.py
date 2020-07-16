import numpy as np
import pdb

import pybullet as p
import roboverse.bullet as bullet
from roboverse.envs.sawyer_base import SawyerBaseEnv

class SawyerLiftMultiEnv(SawyerBaseEnv):

    def __init__(self, goal_pos=None, *args, goal_mult=4, bonus=0, min_reward=-3.,
                 num_obj=2, obj_urdf='spam', max_joint_velocity=None, bowl_type=None,
                 **kwargs):
        self.record_args(locals())
        self._goal_pos = goal_pos
        self._goal_mult = goal_mult
        self._bonus = bonus
        self._min_reward = min_reward
        self._id = 'SawyerLiftMultiEnv'
        self.num_obj = num_obj
        self._obj_urdf = obj_urdf
        self._max_joint_velocity = max_joint_velocity
        assert bowl_type in ['fixed', 'light', 'heavy']
        self._bowl_type = bowl_type
        if self._obj_urdf in ['spam', 'spam_long']:
            self._clip_obj_pos = True
        elif self._obj_urdf == 'spam_2d':
            self._clip_obj_pos = False
        else:
            raise NotImplementedError
        super().__init__(*args, **kwargs)

    def get_params(self):
        params = super().get_params()
        return params

    def _load_meshes(self):
        super()._load_meshes()

        # lid_obj = bullet.objects.lid_2d()
        # self._objects['lid'] = lid_obj
        # numJoints = p.getNumJoints(lid_obj)
        # p.setJointMotorControl2(lid_obj, numJoints - 2, p.VELOCITY_CONTROL, force=0)

        if self._bowl_type == 'fixed':
            self._objects['bowl'] = bullet.objects.bowl()
        elif self._bowl_type == 'light':
            self._objects['bowl'] = bullet.objects.bowl_sliding()
        elif self._bowl_type == 'heavy':
            self._objects['bowl'] = bullet.objects.bowl_sliding_heavy()
        else:
            raise NotImplementedError

        colors = [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 1],
        ]

        positions = [
            [0.5, -0.3, -0.35],
            [0.5, -0.2, -0.35],
            [0.5, -0.1, -0.35],
            [0.5, 0.0, -0.35],
            [0.5, 0.1, -0.35],
            [0.5, 0.2, -0.35],
            [0.5, 0.3, -0.35],
        ]

        for obj_id in range(self.num_obj):
            obj_name = self.get_obj_name(obj_id)

            if self._obj_urdf == 'spam':
                obj_loader_func = bullet.objects.spam
            elif self._obj_urdf == 'spam_long':
                obj_loader_func = bullet.objects.spam_long
            elif self._obj_urdf == 'spam_2d':
                obj_loader_func = bullet.objects.spam_2d
            else:
                raise NotImplementedError

            obj = obj_loader_func(
                pos=positions[obj_id],
            )

            if self._obj_urdf in ['spam', 'spam_long']:
                p.changeVisualShape(obj, -1, rgbaColor=colors[obj_id])

                if self._max_joint_velocity is not None:
                    p.changeDynamics(
                        obj,
                        -1,
                        maxJointVelocity=self._max_joint_velocity,
                    )
            elif self._obj_urdf == 'spam_2d':
                numJoints = p.getNumJoints(obj)
                p.changeVisualShape(obj, numJoints - 1, rgbaColor=colors[obj_id])
                p.setJointMotorControl2(obj, numJoints-1, p.VELOCITY_CONTROL, force=0)

            self._objects[obj_name] = obj

    def get_object_positions(self):
        bodies = sorted([
            v for k, v in self._objects.items()
            if not bullet.has_fixed_root(v) and 'cube' in k
        ])
        obj_pos = []
        for body in bodies:
            if self._obj_urdf in ['spam', 'spam_long']:
                state = bullet.get_body_info(body, ['pos', 'theta'])
            elif self._obj_urdf == 'spam_2d':
                link = bullet.get_index_by_attribute(body, 'link_name', 'spam')
                self._format_state_query()
                state = bullet.get_link_state(body, link, ['pos', 'theta'])
            else:
                raise NotImplementedError
            obj_pos.append(state['pos'])
        return obj_pos

    def get_bowl_position(self):
        if self._bowl_type in ['light', 'heavy']:
            link = bullet.get_index_by_attribute(self._objects['bowl'], 'link_name', 'base')
            self._format_state_query()
            state = bullet.get_link_state(self._objects['bowl'], link, ['pos', 'theta'])
        elif self._bowl_type == 'fixed':
            state = bullet.get_body_info(self._objects['bowl'], ['pos', 'theta'])
        else:
            raise NotImplementedError
        return state['pos']

    def set_bowl_position(self, pos):
        bullet.set_body_state(self._objects['bowl'], pos, deg=[0, 0, 0])

        if self._bowl_type in ['light', 'heavy']:
            # set the joint position to 0
            link = bullet.get_index_by_attribute(self._objects['bowl'], 'link_name', 'base')
            p.resetJointState(self._objects['bowl'], link, 0)

    def get_reward(self, observation):
        """Dummy reward for sawyer lift multi env
        """
        reward = 1
        return reward


    def get_obj_name(self, cube_id):
        return 'cube_{cube_id}'.format(cube_id=cube_id)
