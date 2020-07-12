import gym
from roboverse.envs.env_object_list import (
    POSSIBLE_TRAIN_OBJECTS, POSSIBLE_TRAIN_SCALINGS,
    POSSIBLE_TEST_OBJECTS, POSSIBLE_TEST_SCALINGS)

SEQUENTIAL_ENVIRONMENT_SPECS = (
    {
        'id': 'SawyerBase-v0',
        'entry_point': ('roboverse.envs.sawyer_base:SawyerBaseEnv'),
    },
    {
        'id': 'SawyerLift-v0',
        'entry_point': ('roboverse.envs.sawyer_lift:SawyerLiftEnv'),
    },
    {
        'id': 'SawyerGraspOne-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp:SawyerGraspOneEnv'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'SawyerGraspOneV2-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v2:SawyerGraspV2Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'object_ids': [1]
                   }
    },
    {
        'id': 'SawyerGraspV2-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v2:SawyerGraspV2Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'num_objects': 5,
                   }
    },
    {
        'id': 'SawyerGraspTenV2-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v2:SawyerGraspV2Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'num_objects': 10,
                   }
    },
    {
        'id': 'SawyerGraspOneV3-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v3:SawyerGraspV3Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'height_threshold': -0.3,
                   'object_ids': [1]
                   }
    },
    {
        'id': 'SawyerGraspOneV4-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v4:SawyerGraspV4Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   # 'height_threshold': -0.3,
                   'object_ids': [1]
                   }
    },
    {
        'id': 'SawyerGraspOneObjectSetTenV3-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp_v3:SawyerGraspV3ObjectSetEnv'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'height_threshold': -0.3,
                   }
    },
    {
        'id': 'SawyerReach-v0',
        'entry_point': ('roboverse.envs.sawyer_reach:SawyerReachEnv'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'SawyerLid-v0',
        'entry_point': ('roboverse.envs.sawyer_lid:SawyerLidEnv'),
    },
    {
        'id': 'SawyerSoup-v0',
        'entry_point': ('roboverse.envs.sawyer_soup:SawyerSoupEnv'),
    },
    {
        'id': 'WidowBase-v0',
        'entry_point': ('roboverse.envs.widow_base:WidowBaseEnv'),
    },
    {
        'id': 'Widow200Grasp-v0',
        'entry_point': ('roboverse.envs.widow200_grasp:Widow200GraspEnv'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'Widow200GraspV2-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v2:Widow200GraspV2Env'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'Widow200GraspV5-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v5:Widow200GraspV5Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.20}
    },
    {
        'id': 'Widow200GraspFiveV5-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v5:Widow200GraspV5Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.20,
                   'num_objects': 5,
                   'object_names': ('gatorade', 'jar', 'beer_bottle',
                                    'bunsen_burner', 'square_prism_bin')
                   }
    },
    {
        'id': 'Widow200GraspV5RandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v5:Widow200GraspV5RandObjEnv'),
        'kwargs': {'max_force': 100,
        'action_scale': 0.05,
        'reward_height_threshold': -.20}
    },
    {
        'id': 'Widow200GraspV6-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275}
    },
    {
        'id': 'Widow200GraspV7-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v7:Widow200GraspV7Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275}
    },
    # RANDOM OBJECT GRASP V6 ENVS
    {
        'id': 'Widow200GraspV6RandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275}
    },
    {
        'id': 'Widow200GraspV6OneRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:1],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:1],}
    },
    {
        'id': 'Widow200GraspV6FiveRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:5],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:5],}
    },
    {
        'id': 'Widow200GraspV6TenSameTrainTestRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TEST_OBJECTS[:10], # Same Train and Test objs.
                   'train_scaling_list': POSSIBLE_TEST_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6TenRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:10],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6TwentyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:20],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:20],}
    },
    {
        'id': 'Widow200GraspV6FortyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6:Widow200GraspV6RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:40],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:40],}
    },
    # END RANDOM OBJECT GRASP V6 ENVS
    {
        'id': 'Widow200GraspV5PlaceV0Env-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v5_and_place_v0:Widow200GraspV5AndPlaceV0Env'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.20}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceV0-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.20}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceOnlyV0-v0',
        'entry_point': (
            'roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.20,
                   'place_only': True}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceV0RandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.20}
    },
    {
        'id': 'Widow200GraspV6BoxV0-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275}
    },
    {
        'id': 'Widow200GraspV7BoxV0-v0',
        'entry_point': (
            'roboverse.envs.widow200_grasp_v7_box_v0:Widow200GraspV7BoxV0Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275}
    },
    {
        'id': 'Widow200GraspV6BoxV0RandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275}
    },
    # RANDOM OBJECT GRASP V6 ENVS WITH BOX
    {
        'id': 'Widow200GraspV6BoxV0OneRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:1],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:1],}
    },
    {
        'id': 'Widow200GraspV6BoxV0FiveRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:5],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:5],}
    },
    {
        'id': 'Widow200GraspV6BoxV0TenSameTrainTestRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TEST_OBJECTS[:10], # Same Train and test objs
                   'train_scaling_list': POSSIBLE_TEST_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6BoxV0TenRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:10],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6BoxV0TwentyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:20],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:20],}
    },
    {
        'id': 'Widow200GraspV6BoxV0FortyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_v0:Widow200GraspV6BoxV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:40],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:40],}
    },
    # RANDOM OBJECT PICK + BOX PLACE ENVS.
    {
        'id': 'Widow200GraspV6BoxPlaceV0OneRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:1],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:1],}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceV0FiveRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:5],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:5],}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceV0TenSameTrainTestRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TEST_OBJECTS[:10], # Same Train and Test objs
                   'train_scaling_list': POSSIBLE_TEST_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceV0TenRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:10],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceV0TwentyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:20],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:20],}
    },
    {
        'id': 'Widow200GraspV6BoxPlaceV0FortyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_box_place_v0:Widow200GraspV6BoxPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:40],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:40],}
    },
    # Drawer Place Envs
    {
        'id': 'Widow200GraspV6DrawerPlaceV0-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,}
    },
    {
        'id': 'Widow200GraspV6DrawerPlaceV0RandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,}
    },
    {
        'id': 'Widow200GraspV6DrawerPlaceV0OneRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:1],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:1],}
    },
    {
        'id': 'Widow200GraspV6DrawerPlaceV0FiveRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:5],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:5],}
    },
    {
        'id': 'Widow200GraspV6DrawerPlaceV0TenSameTrainTestRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TEST_OBJECTS[:10], # Same Train and Test objs.
                   'train_scaling_list': POSSIBLE_TEST_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6DrawerPlaceV0TenRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:10],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6DrawerPlaceV0TwentyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:20],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:20],}
    },
    {
        'id': 'Widow200GraspV6DrawerPlaceV0FortyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_place_v0:Widow200GraspV6DrawerPlaceV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:40],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:40],}
    },
    # Drawer Open Envs
    {
        'id': 'Widow200GraspV6DrawerOpenV0-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,}
    },
    {
        'id': 'Widow200GraspV6DrawerOpenV0RandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,}
    },
    {
        'id': 'Widow200GraspV6DrawerOpenV0OneRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:1],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:1],}
    },
    {
        'id': 'Widow200GraspV6DrawerOpenV0FiveRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:5],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:5],}
    },
    {
        'id': 'Widow200GraspV6DrawerOpenV0TenSameTrainTestRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TEST_OBJECTS[:10], # Same Train and Test objs.
                   'train_scaling_list': POSSIBLE_TEST_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6DrawerOpenV0TenRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:10],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:10],}
    },
    {
        'id': 'Widow200GraspV6DrawerOpenV0TwentyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:20],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:20],}
    },
    {
        'id': 'Widow200GraspV6DrawerOpenV0FortyRandObj-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0RandObjEnv'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'possible_train_objects': POSSIBLE_TRAIN_OBJECTS[:40],
                   'train_scaling_list': POSSIBLE_TRAIN_SCALINGS[:40],}
    },
    # Used for stitching exps.
    {
        'id': 'Widow200GraspV6DrawerGraspOnlyV0-v0',
        'entry_point': ('roboverse.envs.widow200_grasp_v6_drawer_open_v0:Widow200GraspV6DrawerOpenV0Env'),
        'kwargs': {'max_force': 10,
                   'action_scale': 0.05,
                   'reward_height_threshold': -.275,
                   'noisily_open_drawer': True,
                   'close_drawer_on_reset': False}
    },
    # BOX PACKING ENVS
    {
        'id': 'WidowBoxPackingOne-v0',
        'entry_point': ('roboverse.envs.widow_box_packing:WidowBoxPackingOneEnv'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },
    {
        'id': 'Widow200BoxPackingV2-v0',
        'entry_point': ('roboverse.envs.widow200_box_packing_v2:WidowBoxPackingV2Env'),
        'kwargs': {'max_force': 100, 'action_scale': 0.05}
    },

)

GRASP_V3_ENV_SPECS = []
OBJ_IDS_TEN = [0, 1, 25, 30, 50, 215, 255, 265, 300, 310]
for i, obj_id in enumerate(OBJ_IDS_TEN):
    env_params = dict(
        id='SawyerGraspOne-{}-V3-v0'.format(i),
        entry_point=('roboverse.envs.sawyer_grasp_v3:SawyerGraspV3Env'),
        kwargs={'max_force': 100,
                   'action_scale': 0.05,
                   'pos_init': [0.7, 0.2, -0.2],
                   'pos_low': [.5, -.05, -.38],
                   'pos_high': [.9, .30, -.15],
                   'object_position_low': (.65, .10, -.20),
                   'object_position_high': (.80, .25, -.20),
                   'num_objects': 1,
                   'height_threshold': -0.3,
                   'object_ids': [obj_id]
               }
    )
    GRASP_V3_ENV_SPECS.append(env_params)
GRASP_V3_ENV_SPECS = tuple(GRASP_V3_ENV_SPECS)


PROJECTION_ENVIRONMENT_SPECS = tuple(
    {
        'id': env['id'].split('-')[0] + '2d-' + env['id'].split('-')[1],
        'entry_point': ('roboverse.envs.sawyer_2d:Sawyer2dEnv'),
        'kwargs': {'env': env['id']},
    } for env in SEQUENTIAL_ENVIRONMENT_SPECS
)

PARALLEL_ENVIRONMENT_SPECS = tuple(
    {
        'id': 'Parallel' + env['id'],
        'entry_point': ('roboverse.envs.parallel_env:ParallelEnv'),
        'kwargs': {'env': env['id']},
    } for env in SEQUENTIAL_ENVIRONMENT_SPECS + PROJECTION_ENVIRONMENT_SPECS
)

BULLET_ENVIRONMENT_SPECS = SEQUENTIAL_ENVIRONMENT_SPECS + \
                           PROJECTION_ENVIRONMENT_SPECS + \
                           PARALLEL_ENVIRONMENT_SPECS + \
                           GRASP_V3_ENV_SPECS

def register_bullet_environments():
    for bullet_environment in BULLET_ENVIRONMENT_SPECS:
        gym.register(**bullet_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  BULLET_ENVIRONMENT_SPECS)

    return gym_ids

def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
