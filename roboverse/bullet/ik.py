import numpy as np
import pybullet as p
import pdb
import time

import roboverse
from roboverse.bullet.queries import (
    get_joint_info,
    get_joint_state,
    get_link_state,
)

from roboverse.bullet.misc import (
    quat_to_deg,
    l2_dist,
    rot_diff_deg,
)


############################
#### inverse kinematics ####
############################

def ik(body, link, pos, theta, damping):
    if type(damping) == float:
        ## if damping is a scalar, repeat for each degree of freedom
        n_dof = p.getNumJoints(body)
        damping = [damping for _ in range(n_dof)]
    ik_solution = p.calculateInverseKinematics(body, link, pos,
                                               # targetOrientation=theta,
                                               jointDamping=damping)
    return np.array(ik_solution)

def ee_approx_eq(pos_a, theta_a, pos_b, theta_b, pos_eps=1e-3, theta_eps=1):
    '''
        theta_a and theta_b : euler angles in degrees
        returns True if ||pos_a-pos_b||_2 <= pos_eps
        and ||theta_a-theta_b||_2 <= theta_eps
    '''
    pos_delta = l2_dist(pos_a, pos_b)
    theta_delta = rot_diff_deg(theta_a, theta_b)
    return pos_delta <= pos_eps and theta_delta <= theta_eps

def get_num_actuators(body):
    '''
        TODO : there is probably a better way to do this
    '''
    joint_indices, _ = get_joint_positions(body)
    return len(joint_indices)

def get_joint_positions(body):
    num_joints = p.getNumJoints(body)
    q_indices = [get_joint_info(body, j, 'q_index') for j in range(num_joints)]
    joint_indices = [j for j in range(num_joints) if q_indices[j] > -1]
    joint_positions = [get_joint_state(body, j, 'pos') for j in joint_indices]
    return np.array(joint_indices), np.array(joint_positions)

def ik_to_joint_vel(body, ik_solution):
    indices, current = get_joint_positions(body)
    velocities = ik_solution - current
    return indices, velocities


def velocity_control(body, joints, velocities):
    '''
        body : int
        joints : np.ndarray of ints
        velocities : np.ndarray of floats
    '''
    joints = joints.tolist()
    velocities = velocities.tolist()
    p.setJointMotorControlArray(body, joints, p.VELOCITY_CONTROL,
                                targetVelocities=velocities)


def position_control(body, link, pos, theta, damping=1e-3):
    ik_solution = ik(body, link, pos, theta, damping)
    joint_indices, _ = get_joint_positions(body)
    for joint_ind, pos in zip(joint_indices, ik_solution):
        p.resetJointState(body, joint_ind, pos)


def sawyer_ik(body, link, pos, theta, gripper, gripper_name=None, damping=1e-3,
              gripper_bounds=(-1,1), arm_vel_mult=3, gripper_vel_mult=10,
              discrete_gripper=True):
    gripper_state = get_gripper_state(body, gripper, gripper_bounds,
                                      discrete_gripper, gripper_name)
    #### ik
    ik_solution = ik(body, link, pos, theta, damping)
    ik_solution[-2:] = gripper_state
    #### velocities
    joints, velocities = ik_to_joint_vel(body, ik_solution)
    velocities[:-2] *= arm_vel_mult
    velocities[-2:] *= gripper_vel_mult
    #### check if end effector already at correct position and orientation
    link_pos, link_deg = get_link_state(body, link, ['pos', 'theta'],
                                        return_list=True)
    deg = quat_to_deg(theta)
    if ee_approx_eq(link_pos, link_deg, pos, deg):
        velocities[:-2] = 0
    #### apply velocities
    velocity_control(body, joints, velocities)


def sawyer_position_ik(body, link, pos, theta, gripper, gripper_name=None,
                       damping=1e-3, gripper_bounds=(-1,1),
                       discrete_gripper=True, max_force=1000.):
    gripper_state = get_gripper_state(body, gripper, gripper_bounds,
                                      discrete_gripper, gripper_name)
    #### ik
    ik_solution = ik(body, link, pos, theta, damping)
    ik_solution[-2:] = gripper_state
    joints, current = get_joint_positions(body)
    #### position control
    forces = [max_force for _ in range(len(joints))]
    p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                targetPositions=ik_solution, forces=forces)


def sawyer_position_theta_ik(body, link, pos, theta, gripper, wrist_theta,
                             gripper_name=None, damping=1e-3,
                             gripper_bounds=(-1,1), discrete_gripper=True,
                             max_force=1000.):
    """
    sawyer_position_ik, but allows for a wrist_theta argument to control
    wrist rotation
    """
    gripper_state = get_gripper_state(body, gripper, gripper_bounds,
                                      discrete_gripper, gripper_name)
    #### ik
    ik_solution = ik(body, link, pos, theta, damping)
    # print("ik_solution", ik_solution)
    ik_solution[-2:] = gripper_state
    joints, current = get_joint_positions(body)
    #### position control
    forces = [max_force for _ in range(len(joints))]
    ik_solution[4] += 3.0*wrist_theta
    # print("ik_solution[4]", ik_solution[4])
    p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                targetPositions=ik_solution, forces=forces)


def step_ik(gripper_range=range(20, 25), body=0):
    '''
        enforces joint limits for gripper fingers
    '''
    p.stepSimulation()
    for joint in gripper_range:
        low, high = get_joint_info(body, joint, ['low', 'high'],
                                   return_list=True)
        pos = get_joint_state(body, joint, 'pos')
        pos = np.clip(pos, low, high)
        p.resetJointState(body, joint, pos)

#################
#### gripper ####
#################


def get_gripper_state(body, gripper, gripper_bounds, discrete_gripper, gripper_name):
    if gripper_name:
        l_limits, r_limits = _get_gripper_limits(body, *gripper_name)
    else:
        l_limits, r_limits = _get_gripper_limits(body)

    if discrete_gripper:
        return _get_discrete_gripper_state(gripper, gripper_bounds, l_limits, r_limits)
    else:
        return _get_continuous_gripper_state(gripper, gripper_bounds, l_limits, r_limits)


def _get_gripper_limits(
        body,
        left_gripper_name='right_gripper_l_finger_joint',
        right_gripper_name='right_gripper_r_finger_joint'
    ):
    l_limits = get_joint_info(body, left_gripper_name,
                              ['low', 'high'])
    r_limits = get_joint_info(body, right_gripper_name,
                              ['low', 'high'])
    return l_limits, r_limits


def _get_discrete_gripper_state(gripper, gripper_bounds, l_limits, r_limits):
    low, high = gripper_bounds
    gripper_close_thresh = (low + high) / 2.
    if gripper > gripper_close_thresh:
        ## close gripper
        gripper_state = [l_limits['low'], r_limits['high']]
    else:
        ## open gripper
        gripper_state = [l_limits['high'], r_limits['low']]
    return gripper_state


def _get_continuous_gripper_state(gripper, gripper_bounds, l_limits, r_limits):
    low, high = gripper_bounds
    percent_closed = (gripper - low) / (high - low)
    l_state = l_limits['high'] + percent_closed * (l_limits['low'] - l_limits['high'])
    r_state = r_limits['low'] + percent_closed * (r_limits['high'] - r_limits['low'])
    return [l_state, r_state]


def restore_state(filename):
    p.restoreState(fileName=filename)

def move_to_neutral(RESET_JOINTS, robot_id):
    for i in range(len(RESET_JOINTS)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, RESET_JOINTS[i])
    joint_norm_dev_from_neutral = 1.0
    max_iters = 100
    iters = 0
    while joint_norm_dev_from_neutral > 0.01 and iters < max_iters:
        iters += 1
        p.stepSimulation()
        currJointStates = get_joint_positions(robot_id)[1][:len(RESET_JOINTS)]
        joint_norm_dev_from_neutral = np.linalg.norm(currJointStates - RESET_JOINTS)

def move_to_neutral_slow(RESET_JOINTS, robot_id, image_size, view_matrix, projection_matrix):
    images = []
    for i in range(len(RESET_JOINTS)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, RESET_JOINTS[i])
    joint_norm_dev_from_neutral = 1.0
    max_iters = 100
    max_images = 10
    iters = 0
    while joint_norm_dev_from_neutral > 0.01 and iters < max_iters:
        iters += 1
        p.stepSimulation()
        img, _, _ = roboverse.bullet.render(image_size, image_size,
                    view_matrix, projection_matrix)
        images.append(img)
        currJointStates = get_joint_positions(robot_id)[1][:len(RESET_JOINTS)]
        joint_norm_dev_from_neutral = np.linalg.norm(currJointStates - RESET_JOINTS)
    images_step_size = max(1, len(images) // (max_images - 1))
    return images[::images_step_size]

#################
####  drawer  ###
#################

def open_drawer(drawer, noisy_open=False, half_open=False):
    slide_drawer(drawer, -1, noisy_open, half_open)

def close_drawer(drawer):
    slide_drawer(drawer, 1)

def slide_drawer(drawer, direction, noisy_open=False, half_open=False):
    assert direction in [-1, 1]
    # -1 = open; 1 = close
    joint_names = [get_joint_info(drawer, j, 'joint_name') for j in range(p.getNumJoints(drawer))]
    drawer_frame_joint_idx = joint_names.index('base_frame_joint')

    num_ts = 10 if direction == -1 else 20

    command = np.clip(10 * direction,
            -10 * np.abs(direction), np.abs(direction))
    # enable fast opening; slow closing

    if noisy_open and direction == -1:
        rand = np.random.uniform(0.12, 0.2)
        command *= rand

    if half_open:
        command *= 0.06

    # Wait a little before closing
    wait_ts = 0 if direction == -1 else 20
    for i in range(wait_ts):
        time.sleep(0.01)
        roboverse.bullet.step()

    p.setJointMotorControl2(
        drawer,
        drawer_frame_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=command,
        force=10
    )

    for i in range(num_ts):
        time.sleep(0.01)
        roboverse.bullet.step()

    p.setJointMotorControl2(
        drawer,
        drawer_frame_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=10
    )

#################
#### pointmass###
#################


def set_pointmass_control(body_id):
    jointFrictionForce = 1
    for joint in range(p.getNumJoints(body_id)):
        p.setJointMotorControl2(body_id, joint, p.POSITION_CONTROL,
                                force=jointFrictionForce)


def pointmass_velocity_step_simulation(body_id, action, sim_steps=15):
    action[2] = 0.
    p.resetBaseVelocity(body_id, linearVelocity=action,
                        angularVelocity=[0, 0, 0])
    for _ in range(sim_steps):
        p.stepSimulation()


def pointmass_position_step_simulation(body_id, action, action_scale=0.1):
    current_pos = np.asarray(p.getBasePositionAndOrientation(body_id)[0])
    target_pos = current_pos
    target_pos[:2] = target_pos[:2] + action*action_scale
    p.resetBasePositionAndOrientation(body_id, target_pos,
                                      p.getQuaternionFromEuler([0., 0, 0]))
    p.resetBaseVelocity(body_id, linearVelocity=[0, 0, 0],
                        angularVelocity=[0, 0, 0])
    p.stepSimulation()
