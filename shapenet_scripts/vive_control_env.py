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

# connect to the VR server (by p.SHARED_MEMORY)
bullet.connect()
bullet.setup()

# query event which is a tuple of changes in controller. 
# Index corresponds to POSITION, ORIENTATION, BUTTTON etc
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6
OBJECT_NAME = "lego"

# set the environment
reward_type = "grasp_only"
env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type)

controllers = [e[0] for e in p.getVREvents()]

trigger = 0
num_grasps = 0
ORIENTATION_ENABLED = True
data = []

def collect_one_trajectory(env, pool, render_images):

    # get VR controller output at one timestamp
    def get_VR_output():
        global trigger
        ee_pos = env.get_end_effector_pos()

        events = p.getVREvents()

        # detect input from controllers
        if events:
            e = events[0]
        else:
            return

        # Detect change in button, and change trigger state
        if e[BUTTONS][33] & p.VR_BUTTON_WAS_TRIGGERED:
            trigger = 1
        if e[BUTTONS][33] & p.VR_BUTTON_WAS_RELEASED:
            trigger = 0


        # pass controller position and orientation into the environment
        # currently orientation is not used yet
        cont_pos = e[POSITION]
        cont_orient = bullet.deg_to_quat([180, 0, 0])
        if ORIENTATION_ENABLED:
            cont_orient = e[ORIENTATION]
            cont_orient = list(cont_orient)


        action = [cont_pos[0] - ee_pos[0], cont_pos[1] - ee_pos[1], cont_pos[2] - ee_pos[2]]
        grip = trigger
        action = np.append(action, [grip])

        #action = np.append(action, cont_orient)
        return action

    o = env.reset()
    images = []

    accept = False

    traj = dict(
        observations=[o],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    # Collect a fixed length of trajectory
    for i in range(args.num_timesteps):
        
        action = get_VR_output()

        if render_images:
            img = env.render()
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

        if info["object_goal_distance_z"] < 0.01:
            accept = True

    # Collect each trajectory besides adding to the DemoPool, for 
    # purpose of  
    if accept:
        data.append(traj)
    return accept, images


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

    reward_type = 'sparse' if args.sparse else 'grasp_only'
    env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type,
                         gui=args.gui, randomize=args.randomize,
                         observation_mode=args.observation_mode)
    num_grasps = 0
    pool = roboverse.utils.DemoPool()
    success_pool = roboverse.utils.DemoPool()

    for j in tqdm(range(args.num_trajectories)):
        render_images = args.video_save_frequency > 0 and \
                        j % args.video_save_frequency == 0


        success, images = collect_one_trajectory(env, pool, render_images)

        if success:
            num_grasps += 1
            print('Num grasps: {}'.format(num_grasps))
            top = pool._size
            bottom = top - args.num_timesteps
            for i in range(bottom, top):
                success_pool.add_sample(
                    pool._fields['observations'][i],
                    pool._fields['actions'][i],
                    pool._fields['next_observations'][i],
                    pool._fields['rewards'][i],
                    pool._fields['terminals'][i]
                )
        if render_images:
            images[0].save('{}/{}.gif'.format(video_save_path, j),
                           format='GIF', append_images=images[1:],
                           save_all=True, duration=100, loop=0)

    params = env.get_params()
    pool.save(params, data_save_path,
              '{}_pool_{}.pkl'.format(timestamp, pool.size))
    success_pool.save(params, data_save_path,
                      '{}_pool_{}_success_only.pkl'.format(
                          timestamp, pool.size))

    path = os.path.join(__file__, "../..", "vr_demos_success + {}.npy".format(timestamp))
    np.save(path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-save-directory", type=str, default="vr_expert_demos")
    parser.add_argument("-n", "--num-trajectories", type=int, default=150)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=250)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--video_save_frequency", type=int,
                        default=0, help="Set to zero for no video saving")
    parser.add_argument("--randomize", dest="randomize",
                        action="store_true", default=True)
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("-o", "--observation-mode", type=str, default='state')

    args = parser.parse_args()

    main(args)

