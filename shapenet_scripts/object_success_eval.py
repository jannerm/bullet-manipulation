import roboverse
import time
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm

def run_and_test_object_success():
    print("Remember to rename the csv if it is already in the dir.")
    EPSILON = 0.05
    save_video = False
    num_trials = 1
    objects_to_test = [
        'short_handle_cup',
        'curved_handle_cup',
        'flowery_half_donut',
        'semi_golf_ball_bowl',
        'passenger_airplane',
        'open_top_rect_box',
        'cookie_circular_lidless_tin',
        'chipotle_bowl',
        'toilet_bowl',
        'buffet_food_tray',
        'keyhole',
        'bathtub',
        'crooked_rim_capsule_container',
        'colunnade_top',
        'pacifier_vase',
        'square_rod_embellishment',
        'bongo_drum_bowl',
        'flat_bottom_sack_vase',
        'stalagcite_chunk',
        'pear_ringed_vase',
        'two_handled_vase',
        'goblet',
        'ringed_cup_oversized_base',
        't_cup',
        'teepee',
        'bullet_vase',
        'haystack_sofa',
        'box_wood_frame',
        'rect_spotted_hollow_bottom_sofa',
        'box_sofa',
        'earmuff',
        'l_automatic_faucet',
        'double_l_faucet',
        'box_crank',
        'glass_half_gallon',
        'pepsi_bottle',
        'two_layered_lampshade',
        'beehive_funnel',
        'rabbit_lamp',
        'elliptical_capsule',
        'trapezoidal_bin',
        'staple_table',
        'grill_park_bench',
        'thick_wood_chair',
        'park_grill_chair',
        'long_half_pipe_smooth_park_chair',
        'flat_boat_dish',
        'modern_canoe',
        'vintage_canoe',
        'oil_tanker',
        'x_curved_modern_bookshelf',
        'pitchfork_shelf',
        'baseball_cap',
        'tongue_chair',
        'grill_trash_can',
        'crooked_lid_trash_can',
        'aero_cylinder',
    ]

    hard_objects_to_test = []
    # objects_to_test = [
    #     'gatorade',
    #     'jar',
    #     'beer_bottle',
    #     'bunsen_burner',
    #     'square_prism_bin',
    #     'long_vase',
    #     'ball',
    #     'shed',
    #     'long_sofa',
    #     'l_sofa',
    #     'smushed_dumbbell',
    #     'mug',
    #     'conic_bin',
    #     'conic_cup',
    #     'sack_vase',
    #     'fountain_vase',
    #     'hex_deep_bowl',
    #     'circular_picnic_table',
    #     'oblong_scooper',
    #     'circular_table',
    # ]

    # hard_objects_to_test = [
    #     'conic_bowl',
    #     'wide_circular_vase',
    #     'pitcher',
    #     'narrow_tray',
    #     'square_deep_bowl',
    #     'flat_circular_basket',
    # ]

    scalings_to_try = [0.2, 0.3, 0.5]
    hard_scalings_to_try = [0.2]

    obj_scaling_to_try = (list(itertools.product(objects_to_test, scalings_to_try)) +
        list(itertools.product(hard_objects_to_test, hard_scalings_to_try)))

    print("obj_scaling_to_try", obj_scaling_to_try)

    # data saving lists
    obj_names_data = []
    scalings_data = []
    final_success_data = []

    for obj, scaling in tqdm(obj_scaling_to_try):
        print("testing object", obj, scaling)
        env = roboverse.make("Widow200GraspV6BoxPlaceV0RandObj-v0",
                             gui=False,
                             possible_train_objects=[obj],
                             scaling_local_list=[scaling],
                             reward_type='sparse',
                             observation_mode='pixels_debug',)
    
        object_ind = 0
        num_successes = 0
        for i in range(num_trials):
            obs = env.reset()
            # object_pos[2] = -0.30
    
            dist_thresh = 0.04 + np.random.normal(scale=0.01)
    
            images = [] # new video at the start of each trajectory.
    
            for _ in range(env.scripted_traj_len):
                if isinstance(obs, dict):
                    state_obs = obs[env.fc_input_key]
                    obj_obs = obs[env.object_obs_key]
    
                ee_pos = state_obs[:3]
                object_pos = obj_obs[object_ind * 7 : object_ind * 7 + 3]
                # object_pos += np.random.normal(scale=0.02, size=(3,))
    
                object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
                theta_action = 0.
                object_goal_dist = np.linalg.norm(object_pos - env._goal_position)
    
                info = env.get_info()
                # theta_action = np.random.uniform()
                # print(object_gripper_dist)
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
                    # print(object_goal_dist)
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
    
                action[:3] += np.random.normal(scale=0.1, size=(3,))
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

