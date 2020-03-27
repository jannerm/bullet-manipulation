import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time

import roboverse

OBJECT_NAME = 'lego'
EPSILON = 1e-8


def scripted_non_markovian(env, pool, render_images):
    env.reset()
    target_pos = env.get_object_midpoint(OBJECT_NAME)
    target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
    # the object is initialized above the table, so let's compensate for it
    target_pos[2] += -0.05
    images = []

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        if i == 0:
            print("initial end effector pos: ", ee_pos)
        if i < 25:
            action = target_pos - ee_pos
            action[2] = 0.
            action *= 5.0
            grip = 0.
        elif i < 35:
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 5.0
            grip = 0.
        elif i < 42:
            action = np.zeros((3,))
            grip = 0.5
        else:
            action = np.zeros((3,))
            action[2] = 1.0
            grip = 1.

        action = np.append(action, [grip])
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = env.render()
            images.append(Image.fromarray(np.uint8(img)))

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)

    success = info['object_goal_distance'] < 0.05
    return success, images


def scripted_markovian(env, pool, render_images):
    observation = env.reset()
    if args.randomize:
        target_pos = env.get_object_midpoint(OBJECT_NAME)
        #target_pos = np.random.uniform(low=env._object_position_low,
        #                              high=env._object_position_high)
        #target_pos[:2] += np.random.uniform(low=-0.03, high=0.03, size=(2,))
        #target_pos[2] += np.random.uniform(low=-0.02, high=0.02, size=(1,))
    else:
        target_pos = env.get_object_midpoint(OBJECT_NAME)
        #target_pos[:2] += np.random.uniform(low=-0.05, high=0.05, size=(2,))
        #target_pos[2] += np.random.uniform(low=-0.01, high=0.01, size=(1,))
        #target_pos[2] += 0.05 #np.random.uniform(low=-0.01, high=0.01, size=(1,))
    # the object is initialized above the table, so let's compensate for it
    # target_pos[2] += -0.01
    print("target_pos: ", target_pos)
    images = []
    grip_open = -1
    grip_close = 1
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        if i == 0:
            print("initial end effector pos: ", ee_pos)
        xyz_diff = target_pos - ee_pos
        xy_diff = xyz_diff[:2]
        # print(observation[3])
        # print(xyz_diff)
        if isinstance(observation, dict):
            gripper_tip_distance = observation['state'][3]
        else:
            gripper_tip_distance = observation[3]
        grip = 0

        if i < 10:
            diff = target_pos - ee_pos
            action = [0,0,diff[2]*0.8]
        else:
            action = target_pos - ee_pos
            action *= 5.0
        """
        elif env.get_info()['gripper_goal_distance'] > 0.02 and gripper_tip_distance > 0.025:
            action = target_pos - ee_pos
            action *= 5.0
            if np.linalg.norm(xy_diff) > 0.05:
                action[2] *= 0.5
            #grip = grip_open
            # print('Approaching')
        #elif gripper_tip_distance > 0.025:
            # o[3] is gripper tip distance
        #    action = np.zeros((3,))
            #if grip == grip_open:
            #    grip = 0.
            #else:
            #    grip = grip_close
            # print('Grasping')
        elif env.get_info()['gripper_goal_distance'] > 0.01:
            action = target_pos - ee_pos #env._goal_pos - ee_pos
            #action *= 5.0
            #grip = grip_close
            # print('Moving')
        else:
            action = np.zeros((3,))
            #grip = grip_close
            # print('Holding')
        """
        action[0] = 0
        action = np.append(action, [grip])
        #action += np.random.normal(scale=args.noise_std)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = env.render()
            images.append(Image.fromarray(np.uint8(img)))

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)
        # time.sleep(0.2)
        observation = next_state
        traj["observations"].append(observation)
        next_state, reward, done, info = env.step(action)
        #print(observation[0:3])
        traj["next_observations"].append(next_state)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)

    success = info['object_gripper_distance'] < 0.05
    #success = info['object_goal_distance'] < 0.05
    return True, images, traj


def main(args):

    timestamp = roboverse.utils.timestamp()
    data_save_path = os.path.join(__file__, "../..", 'data',
                                  args.data_save_directory, timestamp)
    data_save_path = os.path.abspath(data_save_path)
    video_save_path = os.path.join(data_save_path, "videos")

    image_save_path = os.path.join(data_save_path, "images")

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
        os.makedirs(video_save_path)

    reward_type = 'sparse' if args.sparse else 'shaped'
    env = roboverse.make('SawyerGraspOne-v0', reward_type=reward_type,
                         gui=args.gui, randomize=args.randomize,
                         observation_mode=args.observation_mode)
    num_grasps = 0
    pool = roboverse.utils.DemoPool()
    success_pool = roboverse.utils.DemoPool()
    all_trajs = []

    num_success = 0
    for j in tqdm(range(args.num_trajectories)):
        render_images = args.video_save_frequency > 0 and \
                        j % args.video_save_frequency == 0

        if args.non_markovian:
            success, images = scripted_non_markovian(env, pool, render_images)
        else:
            success, images, traj = scripted_markovian(env, pool, render_images)

        if success:
            num_success += 1
            print('Num success: {}'.format(num_success))
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
            all_trajs.append(traj)

            new_image_save_path = os.path.join(image_save_path, str(num_success))
            for i in range(len(images)):
                if not os.path.exists(new_image_save_path):
                    os.makedirs(new_image_save_path)
                import imageio;
                print('{}/t_{}.png'.format(new_image_save_path, i))
                imageio.imwrite(
                    '{}/t_{}.png'.format(new_image_save_path, i), images[i])

            """
            
            images_recreated = []
            env.reset()
            actions = traj["actions"]
            for k in range(len(actions)):
                a = actions[k]
                env.step(a)
                img = env.render()
                images_recreated.append(Image.fromarray(np.uint8(img)))
            """
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
    path = "/home/gaoyuezhou/Desktop/images-bullet/bullet-manipulation/data/reaching_success_fixed_{}.npy".format(timestamp)
    np.save(path, all_trajs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=500)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--video_save_frequency", type=int,
                        default=1, help="Set to zero for no video saving")
    parser.add_argument("--randomize", dest="randomize",
                        action="store_true", default=False)
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("--non-markovian", dest="non_markovian",
                        action="store_true", default=False)
    parser.add_argument("-o", "--observation-mode", type=str, default='state',
                        choices=('state', 'pixels', 'pixels_debug'))

    args = parser.parse_args()

    main(args)