import roboverse as rv
import numpy as np
import skvideo.io

# from experiments.kuanfang.iql.drawer_pnp_commands import drawer_pnp_commands
# from experiments.kuanfang.iql.drawer_pnp_single_obj_commands import drawer_pnp_single_obj_commands
from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands
from roboverse.envs.configs.drawer_pnp_push_env_configs import drawer_pnp_push_env_configs

ts = 100
num_traj = 75

#obs_img_dim=196, 
env = rv.make(
    "SawyerDiverseDrawerPnpPush-v0", #"SawyerRigAffordances-v6", #"SawyerResetFreeDrawerPnpPush-v0", 
    gui=False, 
    expl=True, 
    reset_interval=1, #10, 
    #reset_gripper_interval=1,
    env_obs_img_dim=196, 
    test_env=True, 
    test_env_command=drawer_pnp_push_commands[94],
    # use_test_env_command_sequence=False,
    demo_num_ts=ts,
    # fixed_drawer_yaw=171.86987153482346,
    # fixed_drawer_quadrant=1,
    expert_policy_std=.05,
    downsample=False,
    #configs=drawer_pnp_push_env_configs[1],
    # fixed_task='move_obj_slide',
    # random_init_gripper_pos=True
    # render_depth=True,
    # render_segmentation=True,
)

save_video = True

if save_video:
    video_save_path = '/media/ashvin/data1/patrickhaoy/data/test/'
    num_traj = 1 #4 #2
    observations = np.zeros((num_traj*ts, 196, 196, 3))


tasks_success = dict()
tasks_count = dict()
for i in range(num_traj):
    print(i)
    env.demo_reset()
    curr_task = env.curr_task
    is_done = False
    for t in range(ts):
        # np.uint8(env.render_obs())
        if save_video:
            img = np.uint8(env.render_obs())
            # from PIL import Image
            # im = Image.fromarray(img)
            # im.save(video_save_path + "debug.jpeg")
            # exit()
            observations[i*ts + t, :] = img
        action, done = env.get_demo_action(first_timestep=(t == 0), return_done=True)
        # sign = 1 if i % 2 == 0 else -1
        action = np.array([-1, -1, 0, 0, -1])
        # done = False
        next_observation, reward, _, info = env.step(action)
        if done and not is_done:
            is_done = True 
            
            if curr_task not in tasks_success.keys():
                tasks_success[curr_task] = 1
            else:
                tasks_success[curr_task] += 1
            print("success")
    
    if curr_task not in tasks_count.keys():
        tasks_count[curr_task] = 1
    else:
        tasks_count[curr_task] += 1

print()
total_successes = 0

for task in tasks_count.keys():
    num_successes = tasks_success.get(task, 0)
    num_tries = tasks_count.get(task, 0)
    print(f"{task} | success rate: {num_successes/num_tries}, count: {num_tries} \n")
    total_successes += num_successes

print(f"Overall success rate: {total_successes/num_traj}, count: {num_traj} \n")


if save_video:
    writer = skvideo.io.FFmpegWriter(video_save_path + "debug.mp4")
    for i in range(num_traj*ts):
        writer.writeFrame(observations[i, :, :, :])
    writer.close()
