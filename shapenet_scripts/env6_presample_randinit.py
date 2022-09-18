import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg
import os
from PIL import Image
import math
import argparse
from multiprocess import Pool
import gc

from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands  # NOQA

def collect():
    state_env = roboverse.make('SawyerRigAffordances-v6', expl=True, **kwargs)

    # FOR TESTING, TURN COLORS OFF
    imsize = state_env.obs_img_dim

    renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            width=imsize,
            height=imsize,
            flatten_image=True,)

    renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
    env = InsertImageEnv(state_env, renderer=renderer)
    imlength = env.obs_img_dim * env.obs_img_dim * 3

    act_dim = env.action_space.shape[0]
    dataset = []

    for j in tqdm(range(args.num_frames)):
        env.demo_reset()
        dataset.append(np.uint8(env.render_obs().transpose()).flatten())

    dataset = np.stack(dataset)

    env.close()

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/media/ashvin/data1/patrickhaoy/data/env6_td_pnp_push_vary_color_angle_mixed_tasks')
    parser.add_argument('--test_env_seed', type=int)
    parser.add_argument("--num_frames", type=int, default=16)

    args = parser.parse_args()
    output_path = os.path.join(args.data_path, f'td_pnp_push_init_states_seed{args.test_env_seed}.pkl')

    kwargs = {
        'demo_num_ts': 1,
        'reset_interval': 1,
        'expert_policy_std': .05,
        'downsample': True,
        'env_obs_img_dim': 196,
        'random_init_gripper_pos': True,
        'random_init_gripper_yaw': True,
    }
    if args.test_env_seed != -1:
        kwargs.update({
            'test_env_command': drawer_pnp_push_commands[args.test_env_seed],
            'use_test_env_command_sequence': False,
            'test_env': True,
        })

    dataset = collect()

    file = open(output_path, 'wb')
    pkl.dump(dataset, file)
    file.close()