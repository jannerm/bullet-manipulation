import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt  # NOQA

import roboverse

from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands  # NOQA

########################################
# Args.
########################################
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--num_trajectories', type=int, default=32)
parser.add_argument('--max_steps_per_stage', type=int, default=160)
parser.add_argument('--num_subgoals', type=int, default=16)
parser.add_argument('--subgoal_interval', type=int, default=10)
parser.add_argument('--test_env_seeds', nargs='+', type=int)
parser.add_argument('--debug',
                    dest='debug',
                    action='store_true',
                    default=False)

args = parser.parse_args()

num_trajectories = args.num_trajectories
max_steps_per_stage = args.max_steps_per_stage
subgoal_interval = args.subgoal_interval
num_subgoals = args.num_subgoals
debug = args.debug
timeout_k_steps_after_done = 20
mix_timeout_k = False

for test_env_seed in args.test_env_seeds:  # NOQA
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(
        args.output_dir,
        'td_pnp_push_scripted_goals_seed{}.pkl'.format(str(test_env_seed)))
    print(output_path)
    command = drawer_pnp_push_commands[test_env_seed]

    ########################################
    # Environment.
    ########################################
    kwargs = {
        'test_env_command': command,
        'downsample': True,
        'env_obs_img_dim': 196,
        'random_init_gripper_pos': False, #True,
        'random_init_gripper_yaw': False,
    }
    env = roboverse.make('SawyerRigAffordances-v6',
                                    test_env=True,
                                    expl=True,
                                    **kwargs)

    ########################################
    # Rollout in Environment and Collect Data.
    ########################################
    obs_dim = env.observation_space.spaces['state_achieved_goal'].low.size
    imlength = env.obs_img_dim * env.obs_img_dim * 3
    num_stages = len(command['command_sequence'])

    dataset = {
        'initial_latent_state':
        np.zeros((num_trajectories, 720), dtype=np.float),  # NOQA
        'latent_desired_goal':
        np.zeros((num_trajectories, 720), dtype=np.float),  # NOQA
        'state_desired_goal':
        np.zeros((num_trajectories, obs_dim), dtype=np.float),  # NOQA
        'image_desired_goal':
        np.zeros((num_trajectories, imlength), dtype=np.float),  # NOQA
        'initial_image_observation':
        np.zeros((num_trajectories, imlength), dtype=np.float),  # NOQA
        'image_plan':
        np.zeros((num_trajectories, num_subgoals, imlength),
                 dtype=np.float),  # NOQA
    }

    if mix_timeout_k:
        curr_timeout_k = 0
    for i in tqdm(range(num_trajectories)):
        is_done = False
        # If task isn't done, scripted policy failed, so recollect samples.
        while True:
            print('Traj: %d' % (i))
            valid_traj = True
            plan_imgs = []
            t = 0

            for stage_i in range(num_stages):
                print('-- stage: %d' % (stage_i))
                # env.reset()
                env.demo_reset()
                has_done = False

                if stage_i == 0:
                    init_img = np.uint8(env.render_obs()).transpose() / 255.0
                    init_img = init_img.flatten()

                    # Debug
                    if debug:
                        print('t: ', 0, 'pos: ', env.get_end_effector_pos(),
                              'theta: ', env.get_end_effector_theta())
                        plt.figure()
                        _img = np.reshape(init_img, [3, 48, 48])
                        _img = np.transpose(_img, [2, 1, 0])
                        plt.imshow(_img)
                        plt.show()

                timeout = max_steps_per_stage - 1
                for t_i in range(max_steps_per_stage):
                    action, done = env.get_demo_action(
                        first_timestep=(t_i == 0),
                        # final_timestep=(t_i == max_steps_per_stage - 1),
                        return_done=True)
                    obs, reward, _, info = env.step(action)
                    t += 1

                    if (t % subgoal_interval == 0
                            and len(plan_imgs) < num_subgoals - 1):
                        plan_img = np.uint8(
                            env.render_obs()).transpose() / 255.0
                        plan_img = plan_img.flatten()
                        plan_imgs.append(plan_img)

                        # Debug
                        if debug:
                            print('t: ', t, 'pos: ',
                                  env.get_end_effector_pos(), 'theta: ',
                                  env.get_end_effector_theta())
                            plt.figure()
                            _img = np.reshape(plan_img, [3, 48, 48])
                            _img = np.transpose(_img, [2, 1, 0])
                            plt.imshow(_img)
                            plt.show()

                    if done and not has_done:
                        print('done')
                        if mix_timeout_k:
                            timeout_k = curr_timeout_k
                        else:
                            timeout_k = timeout_k_steps_after_done
                        timeout = t_i + timeout_k  # Lift up the gripper.
                        has_done = True
                    
                    if t_i >= timeout:
                        print('timeout')
                        break

                if not has_done or not done:
                    print('valid_traj = False')
                    valid_traj = False
                    # break

                if not done and has_done:
                    print('Warning: The success condition turned failed after '
                          'task is done.')

            goal_img = np.uint8(env.render_obs()).transpose() / 255.0
            goal_img = goal_img.flatten()
            # Debug
            if debug:
                print('pos: ', env.get_end_effector_pos(), 'theta: ',
                      env.get_end_effector_theta())
                plt.figure()
                _img = np.reshape(goal_img, [3, 48, 48])
                _img = np.transpose(_img, [2, 1, 0])
                plt.imshow(_img)
                plt.show()

            print('number of subgoals: ', len(plan_imgs))
            plan_imgs += [goal_img] * (num_subgoals - len(plan_imgs))

            if not valid_traj:
                print('continue')
                # env.reset()
                continue
            else:
                if mix_timeout_k:
                    curr_timeout_k = (curr_timeout_k + 1) % (timeout_k_steps_after_done + 1)
                dataset['state_desired_goal'][i] = obs['state_achieved_goal']
                dataset['image_desired_goal'][i] = goal_img
                dataset['initial_image_observation'][i] = init_img
                dataset['image_plan'][i] = np.stack(plan_imgs, 0)
                print('break')
                break

    file = open(output_path, 'wb')
    pkl.dump(dataset, file)
    file.close()