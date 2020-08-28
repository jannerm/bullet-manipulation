import numpy as np
import pdb 
import pybullet as p
import roboverse
import roboverse.bullet as bullet
import math
from roboverse.bullet.misc import load_urdf, load_obj
import os

from tqdm import tqdm
from PIL import Image
import argparse
import time

print(p.SHARED_MEMORY)
# connect to the VR server (by p.SHARED_MEMORY)
bullet.connect()
bullet.setup(real_time=False)

# query event which is a tuple of changes in controller. 
# Index corresponds to POSITION, ORIENTATION, BUTTTON etc
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6
OBJECT_NAME = "lego"

# set the environment
reward_type = "grasp_only"
#env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type)

controllers = [e[0] for e in p.getVREvents()]

trigger = 0.8
num_grasps = 0
ORIENTATION_ENABLED = True
data = []

min_traj_length = 30

EPSILON = 0.005

def collect_one_trajectory(env, env2, pool, render_images):
    global min_traj_length

    # get VR controller output at one timestamp
    def get_VR_output():
        global trigger
        ee_pos = env.get_end_effector_pos()

        events = p.getVREvents()

        # detect input from controllers
        if events:
            e = events[0]
        else:
            return np.zeros((4,))

        # Detect change in button, and change trigger state
        if e[BUTTONS][33] & p.VR_BUTTON_IS_DOWN:
            trigger = -0.8
        elif e[BUTTONS][33] & p.VR_BUTTON_WAS_RELEASED:
            trigger = 0.8
        else:
            trigger = 0.0

        # pass controller position and orientation into the environment
        # currently orientation is not used yet
        cont_pos = e[POSITION]
        cont_orient = bullet.deg_to_quat([180, 0, 0])
        if ORIENTATION_ENABLED:
            cont_orient = e[ORIENTATION]
            cont_orient = bullet.quat_to_deg(list(cont_orient))

        action = [cont_pos[0] - ee_pos[0], cont_pos[1] - ee_pos[1], cont_pos[2] - ee_pos[2]]
        action = np.array(action)
        action *= 3

        grip = trigger

        ee_theta = list(bullet.get_link_state(env._robot_id, env._end_effector, 'theta'))

        action = np.append(action, np.random.normal(scale=0.01))
        action = np.append(action, grip)
        action = np.append(action, np.random.normal(scale=0.1))

        noise = 0.1
        noise_scalings = [noise] * 3 + [0.1 * noise] + [noise] * 2
        action += np.random.normal(scale=noise_scalings)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        return action

    o = env.reset()

    time.sleep(1.5)
    images = []

    accept = False

    print(env.original_object_positions)

    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
        original_object_positions=env.original_object_positions,
    )

    # Collect a fixed length of trajectory
    for i in range(args.num_timesteps):

        # for _ in range(3):
        action = get_VR_output()

        if True:
            img = env.render_obs()
            images.append(Image.fromarray(np.uint8(img)))

        observation = env.get_observation()
        traj["observations"].append(observation)
        next_state, reward, done, info = env.step(action)
        traj["next_observations"].append(next_state)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)

        pool.add_sample(observation, action, next_state, reward, done)
        time.sleep(0.03)

    if reward > 0:
        accept = "y"

    return accept, images, traj


def main(args):

    timestamp = roboverse.utils.timestamp()
    data_save_path = os.path.join(__file__, "../..", 'data',
                                  args.data_save_directory, timestamp)
    data_save_path = os.path.abspath(data_save_path)
    video_save_path = os.path.join(data_save_path, "videos")
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
        os.makedirs(video_save_path)

    reward_type = 'sparse'
    env = roboverse.make('Widow200GraspTwoAndPushV6-v0', reward_type=reward_type,
                         randomize=True, observation_mode='pixels_debug', num_objects=2)
    env.reset()
    print(env.render_obs())

    num_grasps = 0
    pool = roboverse.utils.DemoPool()
    success_pool = roboverse.utils.DemoPool()

    for j in tqdm(range(args.num_trajectories)):
        render_images = args.video_save_frequency > 0 and \
                        j % args.video_save_frequency == 0

        #success, images
        success, images, traj = collect_one_trajectory(env, None, pool, render_images)

        while success != 'y' and success != 'Y':
            print("failed for trajectory {}, collect again".format(j))
            success, images, traj = collect_one_trajectory(env, None, pool, render_images)

        top = pool._size
        bottom = top - len(traj["actions"])
        for i in range(bottom, top):
            success_pool.add_sample(
                pool._fields['observations'][i],
                pool._fields['actions'][i],
                pool._fields['next_observations'][i],
                pool._fields['rewards'][i],
                pool._fields['terminals'][i]
            )

        data.append(traj)
        if render_images:
            images[0].save('{}/{}.gif'.format(video_save_path, j),
                           format='GIF', append_images=images[1:],
                           save_all=True, duration=100, loop=0)

        if j % 50 == 0:
            path = os.path.join(__file__, "../..", "vr_demos_success_{}_checkpoint_{}.npy".format(timestamp, j))
            np.save(path, data)


    params = env.get_params()
    pool.save(params, data_save_path, '{}_pool_{}.pkl'.format(timestamp, pool.size))
    success_pool.save(params, data_save_path, '{}_pool_{}_success_only.pkl'.format(timestamp, pool.size))

    path = os.path.join(__file__, "../..", "vr_demos_success_{}.npy".format(timestamp))
    np.save(path, data)

    print("min_traj_length: ", min_traj_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-save-directory", type=str, default="vr_expert_demos")
    parser.add_argument("-n", "--num-trajectories", type=int, default=400)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=30)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--video_save_frequency", type=int,
                        default=1, help="Set to zero for no video saving")
    parser.add_argument("--randomize", dest="randomize",
                        action="store_true", default=True)
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("-o", "--observation-mode", type=str, default='pixels_debug')

    args = parser.parse_args()

    main(args)

