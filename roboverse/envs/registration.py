import gym

BULLET_ENVIRONMENT_SPECS = (
    {
        'id': 'SawyerDrawerPnpPush-v0',
        'entry_point': ('roboverse.envs.sawyer_drawer_pnp_push:SawyerDrawerPnpPush'),
        'kwargs': {'max_force': 100,
                   'action_scale': 0.05,
                   'pos_low': [0.5,-0.2,-.36],
                   'pos_high': [0.85,0.2,-0.1],
                   'pos_init': [0.6, -0.15, -0.2],
                   }
    },
)

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
