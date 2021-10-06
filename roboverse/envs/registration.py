import gym

SEQUENTIAL_ENVIRONMENT_SPECS = (
    {
        'id': 'SawyerBase-v0',
        'entry_point': ('roboverse.envs.sawyer_base:SawyerBaseEnv'),
    },
    {
        'id': 'SawyerRigMultiobj-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_multiobj_v0:SawyerRigMultiobjV0'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   }
    },
    {
        'id': 'SawyerRigMultiobjTray-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_multiobj_tray_v0:SawyerRigMultiobjTrayV0'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   }
    },
    {
        'id': 'SawyerRigMultiobjDrawer-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_multiobj_drawer_v0:SawyerRigMultiobjDrawerV0'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_low': [0.5,-0.2,-.36],
                   'pos_high': [0.85,0.2,-0.1],
                   'pos_init': [0.6, -0.15, -0.2],
                   }
    },
    {
        'id': 'SawyerRigAffordances-v0',
        'entry_point': ('roboverse.envs.sawyer_rig_affordances_v0:SawyerRigAffordancesV0'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_low': [0.5,-0.2,-.36],
                   'pos_high': [0.85,0.2,-0.1],
                   'pos_init': [0.6, -0.15, -0.2],
                   }
    },
    {
        'id': 'BridgeKitchenBase-v0',
        'entry_point': ('roboverse.envs.kitchen.bridge_kitchen_base_v0:BridgeKitchenBaseV0'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_low': [0.5,-0.2,-.36],
                   'pos_high': [0.85,0.2,-0.1],
                   'pos_init': [0.6, -0.15, -0.2],
                   }
    },
    {
        'id': 'BridgeKitchen-v0',
        'entry_point': ('roboverse.envs.kitchen.bridge_kitchen_v0:BridgeKitchenVO'),
    },

    ### SUBSET TASKS ###
    {
        'id': 'RemoveLid-v0',
        'entry_point': ('roboverse.envs.kitchen.remove_lid_v0:RemoveLidVO'),
    },
    {
        'id': 'MugDishRack-v0',
        'entry_point': ('roboverse.envs.kitchen.mug_dishrack_v0:MugDishRackVO'),
    },
    {
        'id': 'CarrotPlate-v0',
        'entry_point': ('roboverse.envs.kitchen.carrot_plate_v0:CarrotPlateVO'),
    },
    {
        'id': 'FlipPot-v0',
        'entry_point': ('roboverse.envs.kitchen.flip_pot_v0:FlipPotV0'),
    },
)

GRASP_V3_ENV_SPECS = []
OBJ_IDS_TEN = [0, 1, 25, 30, 50, 215, 255, 265, 300, 310]
for i, obj_id in enumerate(OBJ_IDS_TEN):
    env_params = dict(
        id='SawyerGraspOne-{}-V3-v0'.format(i),
        entry_point=('roboverse.envs.sawyer_grasp_v3:SawyerGraspV3Env'),
        kwargs={'max_force': 1000,
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
        for environment_spec in BULLET_ENVIRONMENT_SPECS)

    return gym_ids

def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
