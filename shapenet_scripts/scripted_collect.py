import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import argparse
import pickle

import roboverse
import skvideo.io

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

from roboverse.envs.env_list import *

OBJECT_NAME = 'lego'
EPSILON = 0.05

NFS_PATH = '/nfs/kun1/users/avi/batch_rl_datasets/'

def scripted_non_markovian_grasping(env, pool, render_images):
    env.reset()
    target_pos = env.get_object_midpoint(OBJECT_NAME)
    target_pos += np.random.uniform(low=-0.05, high=0.05, size=(3,))
    # the object is initialized above the table, so let's compensate for it
    target_pos[2] += -0.05
    images = []

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()

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
            images.append(img)

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)

    success = info['object_goal_distance'] < 0.05
    return success, images


def scripted_markovian_grasping(env, pool, render_images):
    observation = env.reset()
    if args.randomize:
        target_pos = np.random.uniform(low=env._object_position_low,
                                      high=env._object_position_high)
        target_pos[:2] += np.random.uniform(low=-0.03, high=0.03, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.02, high=0.02, size=(1,))
    else:
        target_pos = env.get_object_midpoint(OBJECT_NAME)
        target_pos[:2] += np.random.uniform(low=-0.05, high=0.05, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.01, high=0.01, size=(1,))

    # the object is initialized above the table, so let's compensate for it
    # target_pos[2] += -0.01
    images = []
    grip_open = -0.8
    grip_close = 0.8

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        xyz_diff = target_pos - ee_pos
        xy_diff = xyz_diff[:2]
        # print(observation[3])
        # print(xyz_diff)
        if isinstance(observation, dict):
            gripper_tip_distance = observation['state'][3]
        else:
            gripper_tip_distance = observation[3]

        if np.linalg.norm(xyz_diff) > 0.02 and gripper_tip_distance > 0.025:
            action = target_pos - ee_pos
            action *= 5.0
            if np.linalg.norm(xy_diff) > 0.05:
                action[2] *= 0.5
            grip = grip_open
            # print('Approaching')
        elif gripper_tip_distance > 0.025:
            # o[3] is gripper tip distance
            action = np.zeros((3,))
            if grip == grip_open:
                grip = 0.
            else:
                grip = grip_close
            # print('Grasping')
        elif info['gripper_goal_distance'] > 0.01:
            action = env._goal_pos - ee_pos
            action *= 5.0
            grip = grip_close
            # print('Moving')
        else:
            action = np.zeros((3,))
            grip = grip_close
            # print('Holding')

        action = np.append(action, [grip])
        action += np.random.normal(scale=args.noise_std)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = observation['image']
            images.append(img)

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)
        # time.sleep(0.2)
        observation = next_state

    success = info['object_goal_distance'] < 0.05
    return success, images


def scripted_grasping_V2(env, pool, success_pool, random_actions=False):
    """

    :param env:
    :param pool:
    :param success_pool:
    :param random_actions: When set to True, executes random actions instead
    of following the object(s).
    :return:
    """

    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    for _ in range(args.num_timesteps):

        if not random_actions:
            if isinstance(observation, dict):
                object_pos = observation['state'][
                             object_ind*7 + 8: object_ind*7 + 8 + 3]
                ee_pos = observation['state'][:3]
            else:
                object_pos = observation[
                             object_ind * 7 + 8: object_ind * 7 + 8 + 3]
                ee_pos = observation[:3]

            action = object_pos - ee_pos
            action = action*4.0
            action += np.random.normal(scale=args.noise_std, size=(3,))
        else:
            action = np.random.uniform(low=-1.0, high=1.0, size=(3,))
            if np.random.uniform() < 0.9:
                action[2] = -1

        theta_action = np.random.uniform(low=-1 + EPSILON, high=1-EPSILON)
        action = np.concatenate((action, np.asarray([theta_action])))

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


def scripted_grasping_V4(env, pool, success_pool):
    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.04 + np.random.normal(scale=0.01)

    for _ in range(args.num_timesteps):

        if isinstance(observation, dict):
            object_pos = observation['state'][
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation['state'][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        # print(object_ind)
        # print(object_pos)
        # print(ee_pos)

        #ee_pos = observation[:3]
        # object_pos = observation[object_ind * 7 + 8: object_ind * 7 + 8 + 3]
        # object_pos += np.random.normal(scale=0.02, size=(3,))

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        theta_action = 0.
        # theta_action = np.random.uniform()

        if object_gripper_dist > dist_thresh and env._gripper_open:
            # print('approaching')
            action = (object_pos - ee_pos) * 4.0
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif env._gripper_open:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 4.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        elif object_pos[2] < env._reward_height_thresh:
            # print('lifting')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
        else:
            # print('terminating')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.7])))

        action += np.random.normal(scale=0.1, size=(6,))
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

def scripted_grasping_V5(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.04 + np.random.normal(scale=0.01)

    for _ in range(args.num_timesteps):

        if isinstance(observation, dict):
            object_pos = observation[env.object_obs_key][
                         object_ind * 7 : object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        theta_action = 0.

        if object_gripper_dist > dist_thresh and env._gripper_open:
            # print('approaching')
            action = (object_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2] / 7.0)
            if xy_diff > 0.02:
                action[2] = 0.0
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif env._gripper_open:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 4.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        else:
            # print('terminating')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.7])))

        action += np.random.normal(scale=noise, size=(6,))
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

def scripted_grasping_V6(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    margin = 0.025
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.045 + np.random.normal(scale=0.01)
    dist_thresh = np.clip(dist_thresh, 0.035, 0.060)

    for _ in range(args.num_timesteps):

        if isinstance(observation, dict):
            object_pos = observation[env.object_obs_key][
                         object_ind * 7 : object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        object_lifted_with_margin = object_pos[2] > (env._reward_height_thresh + margin)

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        theta_action = 0.

        if object_gripper_dist > dist_thresh and env._gripper_open:
            # print('approaching')
            action = (object_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2] / 7.0)
            if "Drawer" in env._env_name:
                if xy_diff > dist_thresh:
                    action[2] = 0.4 # force upward action to avoid upper box
            else:
                if xy_diff > 0.02:
                    action[2] = 0.0
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif env._gripper_open:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 7.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        elif not object_lifted_with_margin:
            # print('raise object upward')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
        else:
            # Move above tray's xy-center.
            tray_info = roboverse.bullet.get_body_info(
                env._tray, quat_to_deg=False)
            tray_center = np.asarray(tray_info['pos'])
            action = (tray_center - ee_pos)[:2]
            action = np.concatenate(
                (action, np.asarray([0., 0., 0., 0.])))

        # action += np.random.normal(scale=noise, size=(6,))
        action[:3] += np.random.normal(scale=noise, size=(3,))
        action[3] += np.random.normal(scale=noise*0.1)
        action[4:] += np.random.normal(scale=noise, size=(2,))
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

def scripted_grasping_V7(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    margin = 0.025
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.045 + np.random.normal(scale=0.01)
    dist_thresh = np.clip(dist_thresh, 0.035, 0.060)

    for _ in range(args.num_timesteps):

        if isinstance(observation, dict):
            object_pos = observation[env.object_obs_key][
                         object_ind * 7 : object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 5: object_ind * 7 + 5 + 3]
            ee_pos = observation[:3]

        # object_lifted_with_margin = object_pos[2] > (env._reward_height_thresh + margin)

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        theta_action = 0.

        if object_gripper_dist > dist_thresh and env._gripper_open:
            # print('approaching')
            action = (object_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2] / 7.0)
            if xy_diff > 0.02:
                action[2] = 0.0
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif env._gripper_open:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 7.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        else:
            action = env.gripper_goal_location - env.get_end_effector_pos()
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))

        # action += np.random.normal(scale=noise, size=(6,))
        action[:3] += np.random.normal(scale=noise, size=(3,))
        action[3] += np.random.normal(scale=noise*0.1)
        action[4:] += np.random.normal(scale=noise, size=(2,))
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


def scripted_grasping_V6_placing_V0(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    object_ind = 0
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.045 + np.random.normal(scale=0.01)
    dist_thresh = np.clip(dist_thresh, 0.035, 0.060)

    box_dist_thresh = 0.035 + np.random.normal(scale=0.01)
    box_dist_thresh = np.clip(box_dist_thresh, 0.025, 0.05)

    assert args.num_timesteps == env.scripted_traj_len, (
        "args.num_timesteps: {} != env.scripted_traj_len: {}".format(
        args.num_timesteps, env.scripted_traj_len))

    for t_ind in range(args.num_timesteps):

        if isinstance(observation, dict):
            object_pos = observation[env.object_obs_key][
                         object_ind * 7 : object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        object_box_dist = np.linalg.norm(env._goal_position[:2] - object_pos[:2])

        theta_action = 0.
        # theta_action = np.random.uniform()

        info = env.get_info()
        if (object_gripper_dist > dist_thresh and
            env._gripper_open and not info['object_above_box_success']):
            # print('approaching')
            action = (object_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2]/7.0)
            if "Drawer" in env._env_name:
                if xy_diff > dist_thresh:
                    action[2] = 0.4 # force upward action to avoid upper box
            else:
                if xy_diff > 0.02:
                    action[2] = 0.0
            action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
        elif env._gripper_open and object_box_dist > box_dist_thresh:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 7.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        elif object_box_dist > box_dist_thresh:
            action = (env._goal_position - object_pos)*7.0
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
        elif not info['object_in_box_success']:
            # object is now above the box.
            action = (env._goal_position - object_pos)*7.0
            action = np.concatenate(
                (action, np.asarray([0., 0.7, 0.])))
        else:
            action = np.zeros((6,))
            action[2] = 0.5

        action[:3] += np.random.normal(scale=noise, size=(3,))
        action[3] += np.random.normal(scale=noise*0.1)
        action[4:] += np.random.normal(scale=noise, size=(2,))

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

def scripted_grasping_V6_opening_V0(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    margin = 0.025
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.04 + np.random.normal(scale=0.01)
    dist_thresh = np.clip(dist_thresh, 0.035, 0.060)
    max_theta_action_magnitude = 0.2
    grasp_target_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
    drawer_never_opened = True

    for _ in range(args.num_timesteps):

        if isinstance(observation, dict):
            object_pos = observation[env.object_obs_key][
                         object_ind * 7 : object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        handle_pos = env.get_handle_pos()
        object_lifted_with_margin = object_pos[2] > (
            env._reward_height_thresh + margin)

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        theta = env.get_wrist_joint_angle() # -pi, pi

        if (gripper_handle_dist > dist_thresh
            and not env.is_drawer_opened(widely=drawer_never_opened)):
            # print('approaching handle')
            action = (handle_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2]/7.0)
            if xy_diff > dist_thresh:
                action[2] = 0.4 # force upward action to avoid upper box
            # Rotate Wrist toward theta = np/2:
            theta_action = np.clip(
                (np.pi / 2) - theta,
                -max_theta_action_magnitude,
                max_theta_action_magnitude
            )
            action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
        elif not env.is_drawer_opened(widely=drawer_never_opened):
            # print("opening drawer")
            action = np.array([0, -1.0, 0])
            # action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
        elif (object_gripper_dist > dist_thresh
            and env._gripper_open and gripper_handle_dist < 1.5 * dist_thresh):
            # print("Lift upward")
            drawer_never_opened = False
            action = np.array([0, 0, 0.7]) # force upward action to avoid upper box
            theta_action_pre_clip = grasp_target_theta - theta
            theta_action = np.clip(
                theta_action_pre_clip,
                -max_theta_action_magnitude,
                max_theta_action_magnitude
            )
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif object_gripper_dist > dist_thresh and env._gripper_open:
            # print("Move toward object")
            action = (object_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2]/7.0)
            if xy_diff > dist_thresh:
                action[2] = 0.1
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif env._gripper_open:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 7.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        elif not object_lifted_with_margin:
            # print('raise object upward')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
        else:
            # Move above tray's xy-center.
            tray_info = roboverse.bullet.get_body_info(
                env._tray, quat_to_deg=False)
            tray_center = np.asarray(tray_info['pos'])
            action = (tray_center - ee_pos)[:2]
            action = np.concatenate(
                (action, np.asarray([0., 0., 0., 0.])))


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

def scripted_grasping_V6_opening_only_V0(env, pool, success_pool, noise=0.2):
    observation = env.reset()
    object_ind = np.random.randint(0, env._num_objects)
    margin = 0.025
    actions, observations, next_observations, rewards, terminals, infos = \
        [], [], [], [], [], []

    dist_thresh = 0.04 + np.random.normal(scale=0.01)
    max_theta_action_magnitude = 0.2
    drawer_never_opened = True

    for _ in range(args.num_timesteps):

        if isinstance(observation, dict):
            object_pos = observation[env.object_obs_key][
                         object_ind * 7 : object_ind * 7 + 3]
            ee_pos = observation[env.fc_input_key][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        handle_offset = np.array([0, -0.01, 0])
        handle_pos = env.get_handle_pos() + handle_offset
        # Make robot aim a little to the left of the handle
        ending_target_pos = np.array([0.73822169, -0.03909928, -0.25635483])
        # Effective neutral pos.
        object_lifted_with_margin = object_pos[2] > (
            env._reward_height_thresh + margin)

        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        theta_action = 0.

        if (gripper_handle_dist > dist_thresh
            and not env.is_drawer_opened(widely=drawer_never_opened)):
            # print('approaching handle')
            action = (handle_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2]/7.0)
            if xy_diff > 0.75 * dist_thresh:
                action[2] = 0.5 # force upward action to avoid upper box
            action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
        elif not env.is_drawer_opened(widely=drawer_never_opened):
            # print("opening drawer")
            action = np.array([0, -1.0, 0])
            # action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
        elif np.abs(ee_pos[2] - ending_target_pos[2]) > dist_thresh:
            # print("Lift upward")
            drawer_never_opened = False
            action = np.array([0, 0, 0.7]) # force upward action to avoid upper box
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        else:
            # print("Move toward neutral")
            action = (ending_target_pos - ee_pos) * 7.0
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))

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

def scripted_markovian_reaching(env, pool, render_images):
    observation = env.reset()
    if args.randomize:
        target_pos = np.random.uniform(low=env._object_position_low,
                                      high=env._object_position_high)
        target_pos[:2] += np.random.uniform(low=-0.03, high=0.03, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.02, high=0.02, size=(1,))
    else:
        target_pos = env.get_object_midpoint(OBJECT_NAME)
        target_pos[:2] += np.random.uniform(low=-0.05, high=0.05, size=(2,))
        target_pos[2] += np.random.uniform(low=-0.01, high=0.01, size=(1,))

    images = []

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        xyz_diff = target_pos - ee_pos
        xy_diff = xyz_diff[:2]

        action = target_pos - ee_pos
        action *= 5.0
        if np.linalg.norm(xy_diff) > 0.05:
            action[2] *= 0.5
        grip = 0.0
        action = np.append(action, [grip])
        action += np.random.normal(scale=args.noise_std)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = observation['image']
            images.append(img)

        observation = env.get_observation()
        next_state, reward, done, info = env.step(action)
        pool.add_sample(observation, action, next_state, reward, done)
        # time.sleep(0.2)
        observation = next_state

    success = info['object_gripper_distance'] < 0.03
    return success, images


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
    elif args.semisparse:
        reward_type = 'semisparse'
        assert args.env in (V6_GRASPING_V0_PLACING_ENVS +
            V6_GRASPING_V0_PLACING_ONLY_ENVS +
            V6_GRASPING_V0_DRAWER_PLACING_ENVS +
            V6_GRASPING_V0_DRAWER_OPENING_ENVS)
    else:
        reward_type = 'shaped'

    assert args.env in (V2_GRASPING_ENVS +
        V4_GRASPING_ENVS + V5_GRASPING_ENVS +
        V6_GRASPING_V0_PLACING_ENVS + V6_GRASPING_ENVS +
        V6_GRASPING_V0_PLACING_ONLY_ENVS +
        V6_GRASPING_V0_DRAWER_PLACING_ENVS +
        V6_GRASPING_V0_DRAWER_OPENING_ENVS +
        V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS +
        V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS +
        V7_GRASPING_ENVS)

    if args.env in PROXY_ENVS_MAP:
        roboverse_env_name = PROXY_ENVS_MAP[args.env]
    else:
        roboverse_env_name = args.env

    env = roboverse.make(roboverse_env_name, reward_type=reward_type,
                         gui=args.gui, randomize=args.randomize,
                         observation_mode=args.observation_mode,
                         transpose_image=True)

    num_success = 0
    if args.env == 'SawyerGraspOne-v0' or args.env == 'SawyerReach-v0':
        pool = roboverse.utils.DemoPool()
        success_pool = roboverse.utils.DemoPool()
    else:
        if args.env in (V2_GRASPING_ENVS + V4_GRASPING_ENVS):
            observation_keys = ('image',)
        else:
            # grasp_v5 env or newer.
            obs_keys = (env.cnn_input_key, env.fc_input_key)
        if 'pixels' in args.observation_mode:
            pool_size = args.num_trajectories*args.num_timesteps + 1
            railrl_pool = ObsDictReplayBuffer(pool_size, env,
                observation_keys=obs_keys)
            railrl_success_pool = ObsDictReplayBuffer(pool_size, env,
                observation_keys=obs_keys)
        elif args.observation_mode == 'state':
            pool_size = args.num_trajectories*args.num_timesteps + 1
            railrl_pool = EnvReplayBuffer(pool_size, env)
            railrl_success_pool = EnvReplayBuffer(pool_size, env)

    for j in tqdm(range(args.num_trajectories)):
        render_images = args.video_save_frequency > 0 and \
                        j % args.video_save_frequency == 0

        if args.env == 'SawyerGraspOne-v0':
            if args.non_markovian:
                success, images = scripted_non_markovian_grasping(env, pool, render_images)
            else:
                success, images = scripted_markovian_grasping(env, pool, render_images)
        elif args.env == 'SawyerReach-v0':
            success, images = scripted_markovian_reaching(env, pool, render_images)
        elif args.env in V2_GRASPING_ENVS:
            assert not render_images
            success = False
            scripted_grasping_V2(env, railrl_pool, railrl_success_pool,
                                 random_actions=args.random_actions)
        elif args.env in V4_GRASPING_ENVS:
            assert not render_images
            success = False
            scripted_grasping_V4(env, railrl_pool, railrl_success_pool)
        elif args.env in V5_GRASPING_ENVS:
            assert not render_images
            success = False
            scripted_grasping_V5(env, railrl_pool, railrl_success_pool,
                                 noise=args.noise_std)
        elif args.env in (V6_GRASPING_ENVS +
            V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS):
            assert not render_images
            success = False
            scripted_grasping_V6(env, railrl_pool, railrl_success_pool,
                                 noise=args.noise_std)
        elif args.env in (V6_GRASPING_V0_PLACING_ENVS +
            V6_GRASPING_V0_PLACING_ONLY_ENVS +
            V6_GRASPING_V0_DRAWER_PLACING_ENVS):
            assert not render_images
            success = False
            scripted_grasping_V6_placing_V0(
                env, railrl_pool, railrl_success_pool, noise=args.noise_std)
        elif args.env in V6_GRASPING_V0_DRAWER_OPENING_ENVS:
            assert not render_images
            success = False
            scripted_grasping_V6_opening_V0(
                env, railrl_pool, railrl_success_pool, noise=args.noise_std)
        elif args.env in V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS:
            assert not render_images
            success = False
            scripted_grasping_V6_opening_only_V0(
                env, railrl_pool, railrl_success_pool, noise=args.noise_std)
        elif args.env in V7_GRASPING_ENVS:
            assert not render_images
            success = False
            scripted_grasping_V7(env, railrl_pool, railrl_success_pool,
                                 noise=args.noise_std)
        else:
            raise NotImplementedError

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
        if render_images:
            filename = '{}/{}.mp4'.format(video_save_path, j)
            writer = skvideo.io.FFmpegWriter(
                filename,
                inputdict={"-r": "10"},
                outputdict={
                    '-vcodec': 'libx264',
                })
            num_frames = len(images)
            for i in range(num_frames):
                writer.writeFrame(images[i])
            writer.close()

    if args.env == 'SawyerGraspOne-v0' or args.env == 'SawyerReach-v0':
        params = env.get_params()
        pool.save(params, data_save_path,
                  '{}_pool_{}.pkl'.format(timestamp, pool.size))
        success_pool.save(params, data_save_path,
                          '{}_pool_{}_success_only.pkl'.format(
                              timestamp, pool.size))
    else:
        path = osp.join(data_save_path,
                        '{}_pool_{}.pkl'.format(timestamp, pool_size))
        pickle.dump(railrl_pool, open(path, 'wb'), protocol=4)
        path = osp.join(data_save_path,
                        '{}_pool_{}_success_only.pkl'.format(
                            timestamp, pool_size))
        pickle.dump(railrl_success_pool, open(path, 'wb'), protocol=4)
        if args.env in (V6_GRASPING_ENVS + V7_GRASPING_ENVS +
            V6_GRASPING_V0_PLACING_ENVS +
            V6_GRASPING_V0_PLACING_ONLY_ENVS +
            V6_GRASPING_V0_DRAWER_PLACING_ENVS +
            V6_GRASPING_V0_DRAWER_OPENING_ENVS +
            V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS +
            V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS):
            # For non terminating envs: we reshape the rewards
            # array and count the number of trajectories with
            # a sucess in the last timestep.
            reshaped_rewards_pool = np.reshape(
                railrl_success_pool._rewards[:-1],
                (args.num_trajectories, args.num_timesteps))
            # print("reshaped_rewards_pool[:,-1]", reshaped_rewards_pool[:,-1])
            print('Num success: {}. Proxy_Env: {}'.format(
                np.sum(reshaped_rewards_pool[:,-1] > 0),
                args.env in PROXY_ENVS_MAP))
            # Num success has little meaning if it is a proxy env, since
            # the reward function corresponds to that of another env.
        else:
            print('Num success: {}'.format(
                np.sum(railrl_success_pool._rewards > 0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str)
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=2000)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=50)
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

    args = parser.parse_args()

    assert args.semisparse != args.sparse

    if args.env in V2_GRASPING_ENVS:
        args.num_timesteps = 20
        assert args.observation_mode != 'pixels'
    elif args.env in (V4_GRASPING_ENVS + V6_GRASPING_ENVS + V7_GRASPING_ENVS +
        V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS):
        args.num_timesteps = 25
        assert args.observation_mode != 'pixels'
    elif args.env in (V6_GRASPING_V0_PLACING_ENVS +
        V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS):
        args.num_timesteps = 30
    elif args.env in V6_GRASPING_V0_PLACING_ONLY_ENVS:
        args.num_timesteps = 10
    elif args.env in (V6_GRASPING_V0_DRAWER_PLACING_ENVS +
        V6_GRASPING_V0_DRAWER_OPENING_ENVS):
        args.num_timesteps = 50
    main(args)
