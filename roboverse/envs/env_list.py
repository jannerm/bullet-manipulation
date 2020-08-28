V2_GRASPING_ENVS = ['SawyerGraspV2-v0',
                    'SawyerGraspTenV2-v0',
                    'SawyerGraspOneV2-v0']
V4_GRASPING_ENVS = ['SawyerGraspOneV4-v0']
V5_GRASPING_ENVS = ['Widow200GraspV5-v0',
                    'Widow200GraspFiveV5-v0',
                    'Widow200GraspV5RandObj-v0',
                    'Widow200GraspThreeV5-v0']
V6_GRASPING_ENVS = ['Widow200GraspV6-v0',
                    'Widow200GraspV6BoxV0-v0',
                    'Widow200GraspV6BoxV0RandObj-v0',
                    'Widow200GraspV6BoxV0OneRandObj-v0',
                    'Widow200GraspV6BoxV0FiveRandObj-v0',
                    'Widow200GraspV6BoxV0TenSameTrainTestRandObj-v0',
                    'Widow200GraspV6BoxV0TenRandObj-v0',
                    'Widow200GraspV6BoxV0TwentyRandObj-v0',
                    'Widow200GraspV6BoxV0FortyRandObj-v0']
V6_GRASPING_V0_PLACING_ENVS = ['Widow200GraspV6BoxPlaceV0-v0',
                               'Widow200GraspV6BoxPlaceV0RandObj-v0',
                               'Widow200GraspV6BoxPlaceV0OneRandObj-v0',
                               'Widow200GraspV6BoxPlaceV0FiveRandObj-v0',
                               'Widow200GraspV6BoxPlaceV0TenSameTrainTestRandObj-v0',
                               'Widow200GraspV6BoxPlaceV0TenRandObj-v0',
                               'Widow200GraspV6BoxPlaceV0TwentyRandObj-v0',
                               'Widow200GraspV6BoxPlaceV0FortyRandObj-v0',]
V6_GRASPING_V0_PLACING_ONLY_ENVS = ['Widow200GraspV6BoxPlaceOnlyV0-v0',]
V6_GRASPING_V0_DRAWER_PLACING_ENVS = ['Widow200GraspV6DrawerPlaceV0-v0',
                                      'Widow200GraspV6DrawerPlaceV0RandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0OneRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0FiveRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0TenSameTrainTestRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0TenRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0TwentyRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0FortyRandObj-v0',]
V6_GRASPING_V0_DRAWER_OPENING_ENVS = ['Widow200GraspV6DrawerPlaceThenOpenV0OpenGraspOnly-v0',
                                      'Widow200GraspV6DrawerOpenV0-v0',
                                      'Widow200GraspV6DrawerOpenV0RandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0OneRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0FiveRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0TenSameTrainTestRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0TenRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0TwentyRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0FortyRandObj-v0']
V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS = ['Widow200GraspV6DrawerOpenOnlyV0-v0',
                                           'Widow200GraspV6DrawerOpenThenPlaceV0OpenOnly-v0',
                                           'Widow200GraspV6DoubleDrawerV0Open-v0',
                                           'Widow200GraspV6DrawerPlaceThenOpenV0OpenOnly-v0']
V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS = ['Widow200GraspV6DrawerGraspOnlyV0-v0',
                                            'Widow200GraspV6DoubleDrawerV0Grasp-v0',
                                            'Widow200GraspV6DrawerPlaceThenOpenV0GraspOnly-v0',
                                            'Widow200GraspV6DoubleDrawerPlaceThenOpenV0Grasp-v0']
V6_GRASPING_V0_DRAWER_CLOSED_PLACING_ENV = ['Widow200GraspV6DrawerPlaceThenOpenV0PlaceOnlyRandQuat-v0',
                                            'Widow200GraspV6DrawerPlaceThenOpenV0PickPlaceOnly-v0']
V6_GRASPING_V0_DRAWER_CLOSED_PLACING_40_ENV = ['Widow200GraspV6DrawerPlaceThenOpenV0PickPlace40Only-v0']
V6_GRASPING_V0_DRAWER_PLACING_OPENING_ENVS = ['Widow200GraspV6DrawerPlaceThenOpenV0-v0',
                                              'Widow200GraspV6DoubleDrawerPlaceThenOpenV0-v0']
V6_GRASPING_V0_DRAWER_OPENING_PLACING_ENVS = ['Widow200GraspV6DrawerOpenThenPlaceV0-v0']
V6_GRASPING_V0_DRAWER_OPEN_PLACE_PLACING_ENVS = ['Widow200GraspV6DrawerOpenThenPlaceV0PickPlaceOnly-v0']
V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_GRASPING_ENVS = ['Widow200GraspV6DoubleDrawerV0CloseOpenGrasp-v0']
V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_ENVS = ['Widow200GraspV6DoubleDrawerV0Close-v0']
V6_GRASPING_V0_DOUBLE_DRAWER_OPENING_ENVS = ['Widow200GraspV6DoubleDrawerV0OpenGrasp-v0']
V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_ENVS = ['Widow200GraspV6DoubleDrawerV0CloseOpen-v0'] # basically, no grasping.
V6_GRASPING_V0_DOUBLE_DRAWER_PICK_PLACE_OPEN_ENVS = ['Widow200GraspV6DoubleDrawerPlaceThenOpenV0PickPlaceOpen-v0']
V6_GRASPING_V0_DOUBLE_DRAWER_CLOSE_OPEN_GRASP_PLACE_ENVS = ['Widow200GraspV6DoubleDrawerV0CloseOpenGraspPlace-v0']
V6_GRASPING_V0_DOUBLE_DRAWER_OPEN_GRASP_PLACE_ENVS = ['Widow200GraspV6DoubleDrawerV0OpenGraspPlace-v0']
V6_GRASPING_V0_DOUBLE_DRAWER_GRASP_PLACE_ENVS = ['Widow200GraspV6DoubleDrawerV0GraspPlace-v0']
V7_GRASPING_ENVS = ['Widow200GraspV7-v0',
                    'Widow200GraspV7BoxV0-v0',
                    'Widow200GraspV7BoxV0TenSameTrainTestRandObj-v0',
                    'Widow200GraspV7BoxV0TenRandObj-v0',
                    'Widow200GraspV7BoxV0TwentyIncludeTestRandObj-v0',
                    'Widow200GraspV7BoxV0FiftyIncludeTestRandObj-v0']
VR_ENVS = []

PROXY_ENVS_MAP = {}

ENV_TO_MAX_PATH_LEN_MAP = {
    frozenset(V6_GRASPING_V0_PLACING_ONLY_ENVS): 10,
    frozenset(V2_GRASPING_ENVS + V5_GRASPING_ENVS): 20,
    frozenset(V6_GRASPING_ENVS + V7_GRASPING_ENVS +
        V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS): 25,
    frozenset(V6_GRASPING_V0_PLACING_ENVS +
        V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS +
        V6_GRASPING_V0_DRAWER_CLOSED_PLACING_ENV +
        V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_ENVS +
        V6_GRASPING_V0_DRAWER_OPEN_PLACE_PLACING_ENVS +
        V6_GRASPING_V0_DOUBLE_DRAWER_GRASP_PLACE_ENVS +
        VR_ENVS): 30,
    frozenset(V6_GRASPING_V0_DRAWER_CLOSED_PLACING_40_ENV): 40,
    frozenset(V6_GRASPING_V0_DRAWER_OPENING_ENVS +
        V6_GRASPING_V0_DRAWER_PLACING_ENVS +
        V6_GRASPING_V0_DOUBLE_DRAWER_OPENING_ENVS +
        V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_ENVS): 50,
    frozenset(V6_GRASPING_V0_DRAWER_OPENING_PLACING_ENVS +
        V6_GRASPING_V0_DOUBLE_DRAWER_OPEN_GRASP_PLACE_ENVS +
        V6_GRASPING_V0_DOUBLE_DRAWER_PICK_PLACE_OPEN_ENVS): 60,
    frozenset(V6_GRASPING_V0_DRAWER_PLACING_OPENING_ENVS +
        V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_GRASPING_ENVS): 80,
    frozenset(V6_GRASPING_V0_DOUBLE_DRAWER_CLOSE_OPEN_GRASP_PLACE_ENVS): 90,
}

def get_max_path_len(env_name):
    for env_set in ENV_TO_MAX_PATH_LEN_MAP:
        if env_name in env_set:
            return ENV_TO_MAX_PATH_LEN_MAP[env_set]
    raise NotImplementedError
