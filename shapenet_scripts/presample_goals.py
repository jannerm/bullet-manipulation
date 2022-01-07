import argparse
import os

import numpy as np
import pickle as pkl
import roboverse
from tqdm import tqdm

import matplotlib.pyplot as plt  # NOQA


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--num_trajectories', type=int, default=8)
parser.add_argument('--num_pre_steps', type=int, default=70)
parser.add_argument('--num_post_steps', type=int, default=1)
parser.add_argument('--downsample', action='store_true')
parser.add_argument("--full_open_close", action="store_true")
parser.add_argument('--test_env_seeds', nargs='+', type=int)
parser.add_argument('--gui', dest='gui', action='store_true', default=False)
parser.add_argument('--video_save_frequency', type=int,
                    default=0, help='Set to zero for no video saving')
parser.add_argument('--debug', action='store_true')

# Environment arguments.
parser.add_argument('--drawer_sliding', action='store_true')
parser.add_argument('--fix_drawer_orientation', action='store_true')
parser.add_argument('--fix_drawer_orientation_semicircle', action='store_true')
parser.add_argument('--new_view', action='store_true')
parser.add_argument('--close_view', action='store_true')
parser.add_argument('--red_drawer_base', action='store_true')

args = parser.parse_args()

# data_save_path = '/2tb/home/patrickhaoy/data/affordances/data/reset_free_v5_rotated_top_drawer/top_drawer_goals.pkl'  # NOQA


def presample_goal(args, test_env_seed):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    kwargs = {
        'drawer_sliding': args.drawer_sliding,
        'fix_drawer_orientation': args.fix_drawer_orientation,
        'fix_drawer_orientation_semicircle': (
            args.fix_drawer_orientation_semicircle),
        'new_view': args.new_view,
        'close_view': args.close_view,
        'red_drawer_base': args.red_drawer_base,

        'full_open_close_init_and_goal': args.full_open_close,
        'expl': True if args.full_open_close else False,
    }

    print(kwargs)

    if test_env_seed is None:
        output_path = os.path.join(args.output_dir, 'goals.pkl')
    else:
        output_path = os.path.join(args.output_dir,
                                   'goals_seed%d.pkl' % (test_env_seed))
        kwargs['test_env_seed'] = test_env_seed

    if args.downsample:
        kwargs['downsample'] = True
        kwargs['env_obs_img_dim'] = 196

    env = roboverse.make('SawyerRigAffordances-v1', test_env=True, **kwargs)

    obs_dim = env.observation_space.spaces['state_achieved_goal'].low.size
    imlength = env.obs_img_dim * env.obs_img_dim * 3

    dataset = {
        # 'test_env_seed': test_env_seed,
        'initial_latent_state': np.zeros(
            (args.num_trajectories * args.num_post_steps, 720),
            dtype=np.float),
        'latent_desired_goal': np.zeros(
            (args.num_trajectories * args.num_post_steps, 720),
            dtype=np.float),
        'state_desired_goal': np.zeros(
            (args.num_trajectories * args.num_post_steps, obs_dim),
            dtype=np.float),
        'image_desired_goal': np.zeros(
            (args.num_trajectories * args.num_post_steps, imlength),
            dtype=np.float),
        'initial_image_observation': np.zeros(
            (args.num_trajectories * args.num_post_steps, imlength),
            dtype=np.float),
    }

    if args.debug:
        plt.figure()

    for i in tqdm(range(args.num_trajectories)):
        print('reset')
        env.demo_reset()
        init_img = np.uint8(env.render_obs()).transpose() / 255.0

        # if args.debug:
        #     print('pre-step')
        #     img = np.uint8(env.render_obs()).transpose() / 255.0
        #     img = img.transpose(2, 1, 0)
        #     plt.imshow(img)
        #     plt.show()

        for t in range(args.num_pre_steps):
            action = env.get_demo_action()
            obs, reward, done, info = env.step(action)

            if args.debug:
                if t % 10 == 0:
                    print('pre-step')
                    img = np.uint8(env.render_obs()).transpose() / 255.0
                    img = img.transpose(2, 1, 0)
                    plt.imshow(img)
                    plt.show()

        for t in range(args.num_post_steps):
            action = env.get_demo_action()
            obs, reward, done, info = env.step(action)

            if args.debug:
                if t % 10 == 0:
                    print('post-step')
                    img = np.uint8(env.render_obs()).transpose() / 255.0
                    img = img.transpose(2, 1, 0)
                    plt.imshow(img)
                    plt.show()

            img = np.uint8(env.render_obs()).transpose() / 255.0
            ind = i * args.num_post_steps + t
            dataset['initial_image_observation'][ind] = init_img.flatten()
            dataset['state_desired_goal'][ind] = obs['state_achieved_goal']
            dataset['image_desired_goal'][ind] = img.flatten()

    file = open(output_path, 'wb')
    pkl.dump(dataset, file)
    file.close()


test_env_seeds = args.test_env_seeds

if test_env_seeds is None:
    presample_goal(args, None)
else:
    for test_env_seed in test_env_seeds:
        presample_goal(args, test_env_seed)
