import roboverse
import numpy as np
import time
import math
import roboverse.utils as utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_video", action="store_true")
args = parser.parse_args()

env = roboverse.make('WidowGraspDownwardsOne-v0', gui=True)
obj_key = 'lego'
num_grasps = 0

env.reset()
# target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
images = []

print(env.get_end_effector_pos())

episode_reward = 0.
holding = False
rotate_object = False
rotate_bowl = False

def angle(p0, p1):
    v0 = np.array(p0) - np.array(p1)

    angle_radians = math.atan2(v0[1], v0[0])
    while angle_radians < 0:
        angle_radians += 2 * math.pi
    while angle_radians > 2 * math.pi:
        angle_radians -= 2 * math.pi
    return angle_radians

for i in range(1000):
    ee_pos = np.array(env.get_end_effector_pos())
    object_pos = np.array(env.get_object_midpoint(obj_key))
    bowl_pos = np.array(env.get_object_midpoint('bowl'))

    xyz_diff = object_pos - ee_pos
    xy_diff = xyz_diff[:2]
    xy_goal_diff = (env._goal_pos - object_pos)[:2]

    if abs(angle(object_pos[:2], np.array([0.7, 0])) - angle(ee_pos[:2], np.array([0.7, 0]))) > 0.1\
            and not holding and not rotate_object:
        a = angle(ee_pos[:2], np.array([0.7, 0]))
        a += 0.1
        print(angle(object_pos[:2], np.array([0.7, 0])),  angle(ee_pos[:2], np.array([0.7, 0])), '*')
        if angle(object_pos[:2], np.array([0.7, 0])) > angle(ee_pos[:2], np.array([0.7, 0])):
            action = -np.array([math.sin(a), -math.cos(a), 0])
        else:
            action = np.array([math.sin(a), -math.cos(a), 0])
        grip = 0
        print('Rotating')
    elif np.linalg.norm(xyz_diff) > 0.042 and not holding:
        action = object_pos - ee_pos
        action /= np.linalg.norm(object_pos - ee_pos)
        action /= 3
        grip=0.
        rotate_object = True
        print('Approaching')
    elif o[3] > 0.05 and not holding:
        # o[3] is gripper tip distance
        action = np.zeros((3,))
        action[2] = -0.01
        grip=0.8
        print('Grasping')
    elif env._goal_pos[2] - object_pos[2] > 0.01 and not holding:
        action = env._goal_pos - object_pos
        grip = 1.0
        action[0] = 0
        action[1] = 0
        action *= 3.0
        print('Lifting')
    elif abs(angle(bowl_pos[:2], np.array([0.7, 0])) - angle(ee_pos[:2], np.array([0.7, 0]))) > 0.1\
            and not holding and not rotate_bowl:
        a = angle(ee_pos[:2], np.array([0.7, 0]))
        a += 0.1
        if angle(bowl_pos[:2], np.array([0.7, 0])) > angle(ee_pos[:2], np.array([0.7, 0])):
            action = -np.array([math.sin(a), -math.cos(a), 0])
        else:
            action = np.array([math.sin(a), -math.cos(a), 0])
        action[2] = 0.05
        grip = 1.0
        print('Rotating')
    elif np.linalg.norm((bowl_pos - object_pos)[:2]) > 0.02:
        action = bowl_pos - object_pos
        grip = 1.0
        action *= 2.0
        action[2] = 0.05
        holding = True
        rotate_bowl = True
        print("Moving to Bowl")
    else:
        action = np.zeros((3,))
        grip=0.
        holding = True
        print('Dropping')



    action = np.append(action, [grip])

    if args.save_video:
        img = env.render()
        images.append(img)

    time.sleep(0.05)
    o, r, d, info = env.step(action)
    print(action)
    print(o[3])
    print(r)
    print('object to goal: {}'.format(info['object_goal_distance']))
    print('object to gripper: {}'.format(info['object_gripper_distance']))
    episode_reward += r

print('Episode reward: {}'.format(episode_reward))
object_pos = env.get_object_midpoint(obj_key)
if object_pos[2] > -0.1:
    num_grasps += 1

if args.save_video:
    utils.save_video('data/lego_test_{}.avi'.format(0), images)
