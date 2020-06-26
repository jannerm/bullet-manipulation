import roboverse
import time
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm

def run_and_test_object_success():
    print("Remember to rename the csv if it is already in the dir.")
    EPSILON = 0.05
    noise = 0.2
    num_trials = 50
    objects_to_test = [
        'conic_bin',
        'jar',
        'gatorade',
        'bunsen_burner',
        'long_vase',
        'ringed_cup_oversized_base',
        'square_rod_embellishment',
        'elliptical_capsule',
        'aero_cylinder',
        'grill_trash_can',
    ]

    # scalings_to_try = [0.2]
    scalings = [0.2, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3]

    # obj_scaling_to_try = (list(itertools.product(objects_to_test, scalings_to_try)))
    obj_scaling_to_try = list(zip(objects_to_test, scalings))
    obj_scaling_to_try = [('gatorade', 0.5)]

    print("obj_scaling_to_try", obj_scaling_to_try)

    # data saving lists
    obj_names_data = []
    scalings_data = []
    final_success_data = []

    for obj, scaling in tqdm(obj_scaling_to_try):
        print("testing object", obj, scaling)
        env = roboverse.make("Widow200GraspV6BoxPlaceV0RandObj-v0",
                             # gui=True,
                             possible_train_objects=[obj],
                             train_scaling_list=[scaling],
                             reward_type='sparse',
                             observation_mode='pixels_debug',)
    
        object_ind = 0
        num_successes = 0
        for i in range(num_trials):
            obs = env.reset()
            # object_pos[2] = -0.30

            dist_thresh = 0.04 + np.random.normal(scale=0.01)

            for _ in range(env.scripted_traj_len):
                if isinstance(obs, dict):
                    state_obs = obs[env.fc_input_key]
                    obj_obs = obs[env.object_obs_key]
    
                ee_pos = state_obs[:3]
                object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
                # object_pos += np.random.normal(scale=0.02, size=(3,))
    
                object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
                theta_action = 0.

                info = env.get_info()
                if (object_gripper_dist > dist_thresh and
                        env._gripper_open and not info['object_above_box_success']):
                        # print('approaching')
                        action = (object_pos - ee_pos) * 7.0
                        xy_diff = np.linalg.norm(action[:2]/7.0)
                        if xy_diff > 0.02:
                            action[2] = 0.0
                        action = np.concatenate((action, np.asarray([theta_action,0.,0.])))
                elif env._gripper_open and not info['object_above_box_success']:
                    # print('gripper closing')
                    action = (object_pos - ee_pos) * 7.0
                    action = np.concatenate(
                        (action, np.asarray([0., -0.7, 0.])))
                elif not info['object_above_box_success']:
                    action = (env._goal_position - object_pos)*7.0
                    # action = np.asarray([0., 0., 0.7])
                    action = np.concatenate(
                        (action, np.asarray([0., 0., 0.])))
                elif not info['object_in_box_success']:
                    # object is now above the box.
                    action = (env._goal_position - object_pos)*7.0
                    action = np.concatenate(
                        (action, np.asarray([0., 0.7, 0.])))
                else:
                    action = np.zeros((6,))

                action[:3] += np.random.normal(scale=noise, size=(3,))
                action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
                obs, rew, done, info = env.step(action)
    
                time.sleep(0.05)
    
            num_successes += rew
            print('reward: {}'.format(rew))
            print('num_successes / i + 1: {}/{}'.format(num_successes, i + 1))
            # print('distance: {}'.format(info['object_goal_dist']))

        # Save data to lists
        obj_names_data.append(obj)
        scalings_data.append(scaling)
        final_success_data.append(num_successes / (i + 1))
        print('--------------------')

    return obj_names_data, scalings_data, final_success_data

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    # idx = df.groupby(['obj_name'])['final_success_avg'].transform(max) == df['final_success_avg']
    # df = df[idx]
    df = df.drop_duplicates(['obj_name'])
    df.to_csv('object_success_best_scaling.csv')

if __name__ == "__main__":
    obj_names_data, scalings_data, final_success_data = run_and_test_object_success()
    df_dict = {
        "obj_name": obj_names_data,
        "scaling": scalings_data,
        "final_success_avg": final_success_data,
    }
    df = pd.DataFrame(df_dict)
    df.to_csv('object_success_unsorted.csv')
    df = df.sort_values(by='final_success_avg', ascending=False)
    print(df)
    df.to_csv('object_success_sorted.csv')

    process_csv('object_success_sorted.csv')

