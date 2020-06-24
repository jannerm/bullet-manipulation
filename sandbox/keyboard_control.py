"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys

import numpy as np

import roboverse as rv

import pygame
from pygame.locals import QUIT, KEYDOWN

pygame.init()
screen = pygame.display.set_mode((400, 300))


char_to_action = {
    'w': np.array([0, 0, 1, 0]),
    'a': np.array([0, -1, 0, 0]),
    's': np.array([0, 0, -1, 0]),
    'd': np.array([0, 1, 0, 0]),
    # 'q': np.array([1, -1, 0, 0]),
    # 'e': np.array([-1, -1, 0, 0]),
    # 'z': np.array([1, 1, 0, 0]),
    # 'c': np.array([-1, 1, 0, 0]),
    # 'k': np.array([0, 0, 1, 0]),
    # 'j': np.array([0, 0, -1, 0]),
    'h': 'close',
    'l': 'open',
    'x': 'toggle',
    'r': 'reset',
    'p': 'put obj in hand',
    'g': 'goal',
}

# env_id = 'SawyerBase2d-v0'
# env_id = 'SawyerGraspOne2d-v0'
env_id = 'SawyerLid2d-v0'
# env_id = 'SawyerSoup2d-v0'
# env_id = 'SawyerMultiSoup2d-v0'

# env = rv.make(
#     env_id,
#     action_scale=.05,
#     action_repeat=50,
#     timestep=1. / 120,
#
#     solver_iterations=250,
#
#     max_force=100,
#
#     gui=True
# )

# env = rv.make(
#     env_id,
#
#     # action_scale=.05,
#     # action_repeat=50,
#     # timestep=1. / 120,
#     # solver_iterations=250,
#     # max_force=100,
#
#     action_scale=.06,
#     action_repeat=10,
#     timestep=1. / 120, #1. / 240
#     solver_iterations=150,
#     max_force=1000,
#
#     gui=True
# )

from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC
env_kwargs={
    'action_scale': .06, #.06
    'action_repeat': 10, #5
    'timestep': 1./120, #1./240 1/120
    'max_force': 1000, #1000
    'solver_iterations': 500, #150

    'gui': True,  # False,
    'goal_mult': 0,
    'pos_init': [.75, -.3, 0],
    'pos_high': [.75, .4, .3], #[.75, .4, .3],
    'pos_low': [.75, -.4, -.36], #[.75, -.4, -.36],
    'reset_obj_in_hand_rate': 0.0, #0.0
    'img_dim': 48,
    'goal_mode': 'obj_in_bowl',
    'num_obj': 4, #2

    # 'use_rotated_gripper': False, #False
    # 'use_wide_gripper': False, #False
    # 'soft_clip': True,
    #
    # 'obj_urdf': 'spam_long',
    # 'max_joint_velocity': 1.0,

    'use_rotated_gripper': True,  # False
    'use_wide_gripper': True,  # False
    'soft_clip': True,
    'obj_urdf': 'spam',
    'max_joint_velocity': None,
}
env = SawyerLiftEnvGC(**env_kwargs)


print(env.observation_space, env.action_space)

NDIM = env.action_space.low.size
lock_action = False
obs = env.reset()
action = np.zeros(10)
while True:
    done = False
    if not lock_action:
        action[:3] = 0
    for event in pygame.event.get():
        event_happened = True
        if event.type == QUIT:
            sys.exit()
        if event.type == KEYDOWN:
            char = event.dict['key']
            new_action = char_to_action.get(chr(char), None)
            # print(new_action)
            if new_action == 'toggle':
                lock_action = not lock_action
            elif new_action == 'reset':
                done = True
            elif new_action == 'goal':
                ob = env.reset()
                env.set_to_goal({"state_desired_goal": ob["state_desired_goal"]})
            elif new_action == 'close':
                action[3] = 1
            elif new_action == 'open':
                action[3] = -1
            elif new_action == 'put obj in hand':
                print("putting obj in hand")
                env.put_obj_in_hand()
                action[3] = 1
            elif new_action is not None:
                action[:3] = new_action[:3]
            else:
                action = np.zeros(10)
            if action.any():
                env.step(action[:4])
                print(env.get_observation())
    if done:
        obs = env.reset()
    env.render()
