V2_GRASPING_ENVS = ['SawyerGraspV2-v0',
                    'SawyerGraspTenV2-v0',
                    'SawyerGraspOneV2-v0']
V4_GRASPING_ENVS = ['SawyerGraspOneV4-v0']
V5_GRASPING_ENVS = ['Widow200GraspV5-v0',
                    'Widow200GraspFiveV5-v0',
                    'Widow200GraspV5RandObj-v0']
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
                               'Widow200GraspV6BoxPlaceV0FortyRandObj-v0']
V6_GRASPING_V0_PLACING_ONLY_ENVS = ['Widow200GraspV6BoxPlaceOnlyV0-v0',]
V6_GRASPING_V0_DRAWER_PLACING_ENVS = ['Widow200GraspV6DrawerPlaceV0-v0',
                                      'Widow200GraspV6DrawerPlaceV0RandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0OneRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0FiveRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0TenSameTrainTestRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0TenRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0TwentyRandObj-v0',
                                      'Widow200GraspV6DrawerPlaceV0FortyRandObj-v0']
V6_GRASPING_V0_DRAWER_OPENING_ENVS = ['Widow200GraspV6DrawerOpenV0-v0',
                                      'Widow200GraspV6DrawerOpenV0RandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0OneRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0FiveRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0TenSameTrainTestRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0TenRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0TwentyRandObj-v0',
                                      'Widow200GraspV6DrawerOpenV0FortyRandObj-v0']
V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS = ['Widow200GraspV6DrawerOpenOnlyV0-v0',]
V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS = ['Widow200GraspV6DrawerGraspOnlyV0-v0',]
V7_GRASPING_ENVS = ['Widow200GraspV7-v0',
                    'Widow200GraspV7BoxV0-v0',
                    'Widow200GraspV7BoxV0TenSameTrainTestRandObj-v0',
                    'Widow200GraspV7BoxV0TenRandObj-v0']

# Proxy envs do not have reward functions since they are used for prior datasets.
PROXY_ENVS_MAP = {
    V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS[0]: V6_GRASPING_V0_DRAWER_OPENING_ENVS[0],
}