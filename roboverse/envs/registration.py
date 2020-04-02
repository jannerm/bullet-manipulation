import gym

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
        'id': 'SawyerLiftMulti-v0',
        'entry_point': ('roboverse.envs.sawyer_lift_multi:SawyerLiftMultiEnv'),
    },
    {
        'id': 'SawyerGraspOne-v0',
        'entry_point': ('roboverse.envs.sawyer_grasp:SawyerGraspOneEnv'),
        'kwargs': {'max_force': 100}
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
        'id': 'SawyerMultiSoup-v0',
        'entry_point': ('roboverse.envs.sawyer_multi_soup:SawyerMultiSoupEnv'),
    },
    {
        'id': 'SawyerLiftGC-v0',
        'entry_point': ('roboverse.envs.goal_conditioned.sawyer_lift:SawyerLiftEnvGC'),
        'kwargs': {
            'img_dim': 84
        },
    },
    {
        'id': 'SawyerLiftBowlGC0.0-v0',
        'entry_point': ('roboverse.envs.goal_conditioned.sawyer_lift:SawyerLiftEnvGC'),
        'kwargs': {
            'goal_mult': 4,
            'action_scale': .03,
            'action_repeat': 15,
            'timestep': 1./240,
            'gui': False,
            'pos_init': [.75, -.3, 0],
            'pos_high':[.75, .4,.3],
            'pos_low':[.75,-.5,-.36],
            'max_force': 1000,
            'solver_iterations': 150,
            'reset_obj_in_hand_rate': 0.0,
            'img_dim': 48,
            'goal_mode': 'obj_in_bowl',
            'num_obj': 2
        }
    },
    {
        'id': 'SawyerLiftBowlGC0.5-v0',
        'entry_point': ('roboverse.envs.goal_conditioned.sawyer_lift:SawyerLiftEnvGC'),
        'kwargs': {
            'goal_mult': 4,
            'action_scale': .03,
            'action_repeat': 15,
            'timestep': 1./240,
            'gui': False,
            'pos_init': [.75, -.3, 0],
            'pos_high':[.75, .4,.3],
            'pos_low':[.75,-.5,-.36],
            'max_force': 1000,
            'solver_iterations': 150,
            'reset_obj_in_hand_rate': 0.5,
            'img_dim': 48,
            'goal_mode': 'obj_in_bowl',
            'num_obj': 2
        }
    },
    {
        'id': 'SawyerLiftGC0.0-v0',
        'entry_point': ('roboverse.envs.goal_conditioned.sawyer_lift:SawyerLiftEnvGC'),
        'kwargs': {
            'goal_mult': 0,
            'action_scale': .03,
            'action_repeat': 15,
            'timestep': 1./240,
            'gui': False,
            'pos_init': [.75, -.3, 0],
            'pos_high':[.75, .4,.3],
            'pos_low':[.75,-.5,-.36],
            'max_force': 1000,
            'solver_iterations': 150,
            'reset_obj_in_hand_rate': 0.0,
            'img_dim': 48,
            'num_obj': 2
        }
    },
    {
        'id': 'SawyerLiftGC0.5-v0',
        'entry_point': ('roboverse.envs.goal_conditioned.sawyer_lift:SawyerLiftEnvGC'),
        'kwargs': {
            'goal_mult': 0,
            'action_scale': .03,
            'action_repeat': 15,
            'timestep': 1./240,
            'gui': False,
            'pos_init': [.75, -.3, 0],
            'pos_high':[.75, .4,.3],
            'pos_low':[.75,-.5,-.36],
            'max_force': 1000,
            'solver_iterations': 150,
            'reset_obj_in_hand_rate': 0.5,
            'img_dim': 48,
            'num_obj': 2
        }
    },

)

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

BULLET_ENVIRONMENT_SPECS = SEQUENTIAL_ENVIRONMENT_SPECS + PROJECTION_ENVIRONMENT_SPECS + PARALLEL_ENVIRONMENT_SPECS

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
