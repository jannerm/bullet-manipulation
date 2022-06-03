import roboverse as rv
import numpy as np
import skvideo.io

# from experiments.kuanfang.iql.drawer_pnp_commands import drawer_pnp_commands
# from experiments.kuanfang.iql.drawer_pnp_single_obj_commands import drawer_pnp_single_obj_commands
from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands
from roboverse.envs.configs.drawer_pnp_push_env_configs import drawer_pnp_push_env_configs
from tqdm import tqdm

ts = 75
num_traj = 100

for config in tqdm(drawer_pnp_push_env_configs):
    env = rv.make(
        "SawyerRigAffordances-v6", 
        gui=True,#False, 
        expl=True, 
        reset_interval=2, 
        env_obs_img_dim=196, 
        #test_env=True, 
        #test_env_command=drawer_pnp_push_commands[31],
        demo_num_ts=ts,
        expert_policy_std=.05,
        downsample=False,
        configs=config,
    )

    save_video = False

    if save_video:
        video_save_path = '/media/ashvin/data1/patrickhaoy/test/'
        num_traj = 2
        observations = np.zeros((num_traj*ts, 196, 196, 3))

    tasks_success = dict()
    tasks_count = dict()
    for i in range(num_traj):
        env.demo_reset()
        curr_task = env.curr_task
        is_done = False
        for t in range(ts):
            if save_video:
                img = np.uint8(env.render_obs())
                observations[i*ts + t, :] = img
            action, done = env.get_demo_action(first_timestep=(t == 0), return_done=True)
            next_observation, reward, _, info = env.step(action)
    if save_video:
        writer = skvideo.io.FFmpegWriter(video_save_path + f"{config['camera_angle']['id']}_{config['object_rgbs']['id']}.mp4")
        for i in range(num_traj*ts):
            writer.writeFrame(observations[i, :, :, :])
        writer.close()
    
    env.close()
