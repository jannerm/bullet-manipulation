import numpy as np
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import argparse
import pickle

import roboverse
import skvideo.io
import roboverse.bullet as bullet

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

from roboverse.envs.env_list import *

NFS_PATH = '/nfs/kun1/users/avi/batch_rl_datasets/'
EPSILON = 0.05


def scripted_grasping_V6_drawer_closed_placing_V0(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    blocking_object_ind = 1
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.045 + np.random.normal(scale=0.01)
    dist_thresh = np.clip(dist_thresh, 0.035, 0.050)

    box_dist_thresh = 0.035 + np.random.normal(scale=0.01)
    box_dist_thresh = np.clip(box_dist_thresh, 0.025, 0.05)

    reset_never_taken = True

    box_pos = np.random.uniform(
        low=(0.6925, - 0.25, - 0.345), high=(0.9, -0.05, - 0.345))

    for t_ind in range(env.scripted_traj_len):

        if isinstance(observation, dict):
            blocking_object_pos = observation[env.object_obs_key][
                         blocking_object_ind * 7 : blocking_object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            blocking_object_pos = observation[
                         blocking_object_ind * 7 + 8: blocking_object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        # box_pos = env.get_box_pos()


        blocking_object_gripper_dist = np.linalg.norm(
            blocking_object_pos - ee_pos)
        blocking_object_box_dist = np.linalg.norm(
            blocking_object_pos[:2] - box_pos[:2])
        theta_action = 0.

        blocking_object_pos_offset = np.array([0.01, -0.01, 0])

        info = env.get_info()
        z_diff = abs(blocking_object_pos[2] + blocking_object_pos_offset[2] - ee_pos[2])

        currJointStates = bullet.get_joint_positions(
            env._robot_id)[1][:len(env.RESET_JOINTS)]
        joint_norm_dev_from_neutral = np.linalg.norm(currJointStates - env.RESET_JOINTS)

        eligible_for_reset = ((args.one_reset_per_traj and reset_never_taken) or
            (not args.one_reset_per_traj))

        if (blocking_object_gripper_dist > dist_thresh ) and \
                env._gripper_open and not info['blocking_object_in_box_success']:
            # print('approaching')
            action = ((blocking_object_pos +
                blocking_object_pos_offset) - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2]/7.0)
            if xy_diff > 0.02:
                action[2] *= 0.3
            action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
        elif (env._gripper_open and blocking_object_box_dist > box_dist_thresh and
            not info['blocking_object_in_box_success']):
            # print('gripper closing')
            action = (blocking_object_pos - ee_pos)
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        elif blocking_object_box_dist > box_dist_thresh and \
                not info['blocking_object_in_box_success']:
            action = (box_pos - blocking_object_pos)*7.0
            xy_diff = np.linalg.norm(action[:2]/7.0)
            if "DrawerPlaceThenOpen" in env._env_name:
                # print("don't droop down until xy-close to box")
                action[2] = 0.0
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
            # print("blocking_object_pos", blocking_object_pos)
        elif not info['blocking_object_in_box_success']:
            # object is now above the box.
            action = (box_pos - blocking_object_pos)*7.0
            action[2] = 0.2
            action = np.concatenate(
                (action, np.asarray([0., 0.7, 0.])))
        elif ((joint_norm_dev_from_neutral > args.joint_norm_thresh) and
            eligible_for_reset):
            # print("Move toward neutral")
            action = np.asarray([0., 0., 0., 0., 0., 0.7])
            # 0.7 = move to reset.
            reset_never_taken = False
        else:
            action = np.zeros((6,))

        noise_scalings = [noise] * 3 + [0.1 * noise] + [noise] * 2
        action += np.random.normal(scale=noise_scalings)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        next_observation, reward, done, info = env.step(action)
        actions.append(action)
        observations.append(observation)
        rewards.append(reward)
        terminals.append(done)
        infos.append(info)
        next_observations.append(next_observation)

        observation = next_observation

        if done:
            break

        if not reset_never_taken and args.end_at_neutral:
            break


    path = dict(
        actions=actions,
        rewards=np.asarray(rewards).reshape((-1, 1)),
        terminals=np.asarray(terminals).reshape((-1, 1)),
        infos=infos,
        observations=observations,
        next_observations=next_observations,
    )

    if not isinstance(observation, dict):
        path_length = len(rewards)
        path['agent_infos'] = np.asarray([{} for i in range(path_length)])
        path['env_infos'] = np.asarray([{} for i in range(path_length)])

    pool.add_path(path)
    if rewards[-1] > 0:
        success_pool.add_path(path)
        if args.end_at_neutral:
            return 1 # Only return 1 if end_at_neutral == True and last timestep was success.

def scripted_grasping_V6_double_drawer_close_open_V0(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    object_ind = 0
    margin = 0.025
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.045 + np.random.normal(scale=0.01)
    dist_thresh = np.clip(dist_thresh, 0.040, 0.060)
    drawer_never_opened = True
    reached_pushing_region = False
    reset_never_taken = True

    random_dir = np.random.uniform(-1, 1, 1)

    for _ in range(env.scripted_traj_len):

        if isinstance(observation, dict):
            object_pos = observation[env.object_obs_key][
                         object_ind * 7 : object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        top_drawer_pos = env.get_drawer_bottom_pos("top")
        top_drawer_push_target_pos = (top_drawer_pos +
            np.array([0, -0.15, 0.02]))
        is_gripper_ready_to_push = (ee_pos[1] < top_drawer_push_target_pos[1] and
                ee_pos[2] < top_drawer_push_target_pos[2])
        bottom_drawer_handle_pos = env.get_bottom_drawer_handle_pos()
        object_lifted_with_margin = object_pos[2] > (
            env._reward_height_thresh + margin)

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        gripper_handle_dist = np.linalg.norm(bottom_drawer_handle_pos - ee_pos)
        theta_action = 0.

        currJointStates = bullet.get_joint_positions(
            env._robot_id)[1][:len(env.RESET_JOINTS)]
        joint_norm_dev_from_neutral = np.linalg.norm(currJointStates - env.RESET_JOINTS)

        eligible_for_reset = ((args.one_reset_per_traj and reset_never_taken) or
            (not args.one_reset_per_traj))

        if (gripper_handle_dist > dist_thresh
            and not env.is_drawer_opened("bottom", widely=drawer_never_opened)):
            # print('approaching handle')
            handle_offset = np.array([0.02, 0, 0])
            action = (bottom_drawer_handle_pos + handle_offset - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2]/7.0)
            if xy_diff > 0.75 * dist_thresh:
                action[1] = -1 * 0.5 # move left, screw up.
                action[2] = -0.2 # force upward action
            action = np.concatenate((action, np.asarray([theta_action,0.7,0.])))
        elif not env.is_drawer_opened("bottom", widely=drawer_never_opened):
            # print("opening drawer")
            action = np.array([0, -1.0, 0])
            # action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0.7, 0.])))
        elif (object_gripper_dist > dist_thresh
            and env._gripper_open and gripper_handle_dist < 1.5 * dist_thresh):
            # print("Lift upward")
            drawer_never_opened = False
            if ee_pos[2] < -.15:
                action = env.gripper_goal_location - ee_pos
                action[2]  = 0.7  # force upward action to avoid upper box
            else:
                action = env.gripper_goal_location - ee_pos
                action *= 7.0
                action[2]  *= 0.5  # force upward action to avoid upper box
            action = np.concatenate(
                (action, np.asarray([theta_action, 0.7, 0.])))
        elif ((joint_norm_dev_from_neutral > args.joint_norm_thresh) and
            eligible_for_reset):
            # print("Take neutral action")
            action = np.asarray([0., 0., 0., 0., 0., 0.7])
            # 0.7 = move to reset.
            reset_never_taken = False
        else:
            if not eligible_for_reset:
                action = (ending_target_pos - ee_pos) * 7.0
                action = np.concatenate((action, np.zeros((3,))))
            else:
                action = np.zeros((6,))


        noise_scalings = [noise] * 3 + [0.1 * noise] + [noise] * 2
        action += np.random.normal(scale=noise_scalings)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        next_observation, reward, done, info = env.step(action)

        actions.append(action)
        observations.append(observation)
        rewards.append(reward)
        terminals.append(done)
        infos.append(info)
        next_observations.append(next_observation)

        observation = next_observation

        if done or (not reset_never_taken and args.end_at_neutral):
            break

    path = dict(
        actions=actions,
        rewards=np.asarray(rewards).reshape((-1, 1)),
        terminals=np.asarray(terminals).reshape((-1, 1)),
        infos=infos,
        observations=observations,
        next_observations=next_observations,
    )

    if not isinstance(observation, dict):
        path_length = len(rewards)
        path['agent_infos'] = np.asarray([{} for i in range(path_length)])
        path['env_infos'] = np.asarray([{} for i in range(path_length)])

    pool.add_path(path)
    if rewards[-1] > 0:
        success_pool.add_path(path)
        if args.end_at_neutral:
            return 1


def main(args):

    timestamp = roboverse.utils.timestamp()
    if osp.exists(NFS_PATH):
        data_save_path = osp.join(NFS_PATH, args.data_save_directory, timestamp)
    else:
        data_save_path = osp.join(os.path.dirname(__file__), "..", 'data',
                                  args.data_save_directory,  timestamp)
    data_save_path = os.path.abspath(data_save_path)
    video_save_path = os.path.join(data_save_path, "videos")
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
        os.makedirs(video_save_path)

    if args.sparse:
        reward_type = 'sparse'

    if args.env in PROXY_ENVS_MAP:
        roboverse_env_name = PROXY_ENVS_MAP[args.env]
    else:
        roboverse_env_name = args.env

    env = roboverse.make(roboverse_env_name, reward_type=reward_type,
                         gui=args.gui, randomize=args.randomize,
                         observation_mode=args.observation_mode,
                         transpose_image=True)

    obs_keys = (env.cnn_input_key, env.fc_input_key)
    assert 'pixels' in args.observation_mode

    pool_size = args.num_trajectories * env.scripted_traj_len + 1
    railrl_pool = ObsDictReplayBuffer(pool_size, env,
                                          observation_keys=obs_keys)
    railrl_success_pool = ObsDictReplayBuffer(pool_size, env,
                                              observation_keys=obs_keys)

    end_at_neutral_num_successes = 0 # used if args.end_at_neutral == True.

    for j in tqdm(range(args.num_trajectories)):
        if args.env in (V6_GRASPING_V0_DRAWER_CLOSED_PLACING_ENV +
            V6_GRASPING_V0_DOUBLE_DRAWER_PICK_PLACE_OPEN_ENVS):
            success = False
            result = scripted_grasping_V6_drawer_closed_placing_V0(
                env, railrl_pool, railrl_success_pool, noise=args.noise_std)
            end_at_neutral_num_successes += (result == 1)
        elif args.env in V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_ENVS:
            success = False
            result = scripted_grasping_V6_double_drawer_close_open_V0(
                env, railrl_pool, railrl_success_pool, noise=args.noise_std)
            end_at_neutral_num_successes += (result == 1)
        else:
            raise NotImplementedError

    path = osp.join(data_save_path,
                    '{}_pool_{}.pkl'.format(timestamp, pool_size))
    pickle.dump(railrl_pool, open(path, 'wb'), protocol=4)
    path = osp.join(data_save_path,
                    '{}_pool_{}_success_only.pkl'.format(
                        timestamp, pool_size))
    pickle.dump(railrl_success_pool, open(path, 'wb'), protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str)
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=2000)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--video_save_frequency", type=int,
                        default=0, help="Set to zero for no video saving")
    parser.add_argument("--randomize", dest="randomize",
                        action="store_true", default=False)
    parser.add_argument("--random-actions", dest="random_actions",
                        action="store_true", default=False)
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("--semisparse", dest="semisparse", action="store_true",
                        default=False)
    parser.add_argument("--non-markovian", dest="non_markovian",
                        action="store_true", default=False)
    parser.add_argument("-o", "--observation-mode", type=str, default='pixels',
                        choices=('state', 'pixels', 'pixels_debug'))
    parser.add_argument("--allow-grasp-retries", dest="allow_grasp_retries",
                        action="store_true", default=False)
    parser.add_argument("-j", "--joint-norm-thresh", type=float, default=0.05)
    parser.add_argument("--one-reset-per-traj", dest="one_reset_per_traj",
                        action="store_true", default=False)
    parser.add_argument("--end-at-neutral", dest="end_at_neutral",
                        action="store_true", default=False)

    args = parser.parse_args()

    assert args.semisparse != args.sparse

    main(args)
