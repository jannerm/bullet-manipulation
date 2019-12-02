import roboverse
import numpy as np
from tqdm import tqdm
import roboverse.utils as utils
import math
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_save_directory", type=str)
parser.add_argument("--num_trajectories", type=int, default=2000)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
parser.add_argument("--gui", dest="gui", action="store_true", default=False)

args = parser.parse_args()
timestamp = roboverse.utils.timestamp()
data_save_path = os.path.join(__file__, "../..", 'data',
                              args.data_save_directory)
data_save_path = os.path.abspath(data_save_path)
video_save_path = os.path.join(data_save_path, timestamp, "videos")
trajectory_save_path = os.path.join(data_save_path, "trajectories")

env = roboverse.make('WidowGraspDownwardsOne-v0', gui=args.gui)
object_name = 'lego'
num_grasps = 0
image_data = []

obs_dim = env.observation_space.shape
assert(len(obs_dim) == 1)
obs_dim = obs_dim[0]
act_dim = env.action_space.shape[0]

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
if not os.path.exists(trajectory_save_path):
    os.makedirs(trajectory_save_path)
if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
    os.makedirs(video_save_path)


pool = roboverse.utils.DemoPool()

for j in tqdm(range(args.num_trajectories)):
    env.reset()
    target_pos = env.get_object_midpoint(object_name)
    offset = np.random.uniform(low=-0.05, high=0.05, size=(3,))
    offset[2] = np.random.uniform(low=-0.01, high=0.05, size=(1,))
    starting_pos = np.array(env.get_end_effector_pos())
    starting_pos[0] = np.random.uniform(low=-0.01, high=0.01, size=(1,))

    for i in range(20):
        action = np.random.uniform(low=-0.5, high=0.5, size=(3,))
        grip = 0
        action = np.append(action, [grip])
        env.step(action)
    for i in range(10):
        env.step([0, 0, 0, 0])
    target_pos += offset
    # the object is initialized above the table, so let's compensate for it
    images = []
    trajectory = roboverse.utils.Trajectory()
    holding = False
    rotate_object = False
    rotate_goal = False
    grasping = False

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        object_pos = env.get_object_midpoint(object_name)
        target_pos = env.get_object_midpoint(object_name) + offset

        xyz_diff = target_pos - ee_pos
        xy_diff = xyz_diff[:2]
        xy_goal_diff = (env._goal_pos - object_pos)[:2]

        if abs(utils.angle(target_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0]))) > 0.1 \
                and not holding and not rotate_object and not grasping:
            a = utils.angle(ee_pos[:2], np.array([0.7, 0]))
            a += 0.1
            diff = utils.angle(target_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0]))
            diff = diff - 2 * math.pi if diff > 2 * math.pi else diff + 2 * math.pi if diff < 0 else diff
            if diff > math.pi:
                action = np.array([math.sin(a), -math.cos(a), 0])
            else:
                action = -np.array([math.sin(a), -math.cos(a), 0])
            grip = 0#if not grasping else 1
            action /= 2.0
            #print('Rotating')
        elif np.linalg.norm(xyz_diff) > 0.05 and not holding and not grasping:
            action = target_pos - ee_pos
            action /= np.linalg.norm(target_pos - ee_pos)
            action /= 3
            grip = 0.# if not grasping else 1
            rotate_object = True
            #print('Approaching')
        elif next_state[3] > 0.05 and not holding:
            # o[3] is gripper tip distance
            action = np.zeros((3,))
            action[2] = -0.01
            grip = 1.0
            grasping = True
            #print('Grasping')
        elif -0.27 - object_pos[2] > 0.01 and not holding:
            action = env._goal_pos - object_pos
            grip = 1.0
            action[0] = 0
            action[1] = 0
            action *= 3.0
            #print('Lifting')
        elif abs(utils.angle(env._goal_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0]))) > 0.1 \
                and not holding and not rotate_goal:
            a = utils.angle(ee_pos[:2], np.array([0.7, 0]))
            a += 0.1
            diff = utils.angle(env._goal_pos[:2], np.array([0.7, 0])) - utils.angle(ee_pos[:2], np.array([0.7, 0]))
            diff = diff - 2 * math.pi if diff > 2 * math.pi else diff + 2 * math.pi if diff < 0 else diff
            if diff > math.pi:
                action = np.array([math.sin(a), -math.cos(a), 0])
            else:
                action = -np.array([math.sin(a), -math.cos(a), 0])
            action[2] = 0.02
            action /= 3.0
            grip = 1.0
            #print('Rotating')
        elif np.linalg.norm((env._goal_pos - object_pos)[:2]) > 0.02:
            action = env._goal_pos - object_pos
            grip = 1.0
            action *= 3.0
            action[2] = 0.05
            holding = True
            rotate_goal = True
            #print("Moving to Goal")
        elif np.linalg.norm((env._goal_pos - object_pos)) > 0.01:
            action = env._goal_pos - object_pos
            grip = 1.0
            action *= 3.0
            holding = True
            rotate_goal = True
            #print("Lifting")
        else:
            action = np.zeros((3,))
            grip = 1.
            holding = True
            #print('Holding')

        action = np.append(action, [grip])

        if args.video_save_frequency > 0 and j % args.video_save_frequency == 0:
            img = env.render()
            images.append(Image.fromarray(np.uint8(img)))

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        trajectory.add_sample(observation, action, next_state, reward, done)

    pool.add_trajectory(trajectory)
    object_pos = env.get_object_midpoint(object_name)
    if info['object_goal_distance'] < 0.05:
        num_grasps += 1
    print('Num grasps: {0}/{1}'.format(num_grasps, j + 1))

    if args.video_save_frequency > 0 and j % args.video_save_frequency == 0:
        images[0].save('{}/{}.gif'.format(video_save_path, j),
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)

params = env.get_params()
pool.save(params, trajectory_save_path,
          '{}_pool_{}.pkl'.format(timestamp, pool.size))
