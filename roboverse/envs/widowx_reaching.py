from roboverse.envs.widow200_grasp_v6 import Widow200GraspV6Env
import roboverse.bullet as bullet
import numpy as np
import gym

class WidowXReaching(Widow200GraspV6Env):
    def __init__(self,
        eps = 0.1,
        sparse = True,
        scaling_local_list=[0.5],
        img_dim=48,
        gui=True,
        action_scale=.2,
        action_repeat=1,
        timestep=1. / 120,
        solver_iterations=150,
        gripper_bounds=[-1, 1],
        pos_high=[0.13, 0.13, 0.45],
        pos_low=[-0.13, -0.13,  0.35],
        max_force=1000.,
        visualize=True,
        downwards=True,
        observation_mode='pixels',
        transpose_image=False,
        reward_height_threshold=-0.25,
        num_objects=1,
        object_names=('beer_bottle',),
        ):

        pos_init = np.random.uniform(low=-1, high=1, size=(3,))
        pos_init[0] *= 0.13
        pos_init[1] *= 0.13
        pos_init[2] = pos_init[2] * 0.4 + 0.05

        gp = np.random.uniform(low=-1, high=1, size=(3,))
        gp *= 0.13
        gp *= 0.13
        gp = gp * 0.4 + 0.05

        # TODO Camera orientaition
        super().__init__(scaling_local_list=[0.5],
        img_dim=img_dim,
        gui=gui,
        action_scale=action_scale,
        action_repeat=action_repeat,
        timestep=timestep,
        solver_iterations=solver_iterations,
        gripper_bounds=gripper_bounds,
        pos_init=pos_init,
        pos_high=pos_high,
        pos_low=pos_low,
        max_force=max_force,
        visualize=visualize,
        downwards=downwards,
        observation_mode=observation_mode,
        transpose_image=transpose_image,
        reward_height_threshold=reward_height_threshold,
        num_objects=num_objects,
        object_names=object_names,)

        self.gp = gp
        self.eps = eps
        self.sparse = sparse
    
    def get_sparse_rew(self, pos):
        dist = np.linalg.norm(self.gp-pos[:3])
        return 5 if dist < self.eps else -1

    def get_dist_rew(self, pos):
        return -np.linalg.norm(self.gp-pos[:3])

    def get_reward(self, info):
        pos = np.asarray(self.get_end_effector_pos())
        if self.sparse:
            return self.get_sparse_rew(pos)
        return self.get_dist_rew(pos)