import roboverse as rv
import numpy as np
import skvideo.io
import torch

# from experiments.kuanfang.iql.drawer_pnp_commands import drawer_pnp_commands
# from experiments.kuanfang.iql.drawer_pnp_single_obj_commands import drawer_pnp_single_obj_commands
from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands
from rlkit.experimental.kuanfang.utils import io_util
import rlkit.torch.pytorch_util as ptu

ptu.set_gpu_mode(True)

ts = 400
num_traj = 1

#obs_img_dim=196, 
env = rv.make(
    "SawyerRigAffordances-v6", 
    gui=False, 
    expl=True, 
    reset_interval=1, 
    drawer_sliding=False, 
    env_obs_img_dim=512, 
    random_color_p=0.0, 
    test_env=True, 
    test_env_command=drawer_pnp_push_commands[31], ## CHANGE
    use_single_obj_idx=1,
    #large_obj=False,
    demo_num_ts=ts,
    # version=5,
    #move_gripper_task=True,
    # use_trash=True,
    # fixed_drawer_yaw=24.18556394023222,
    # fixed_drawer_position=np.array([0.50850424, 0.11416014, -0.34]),
    expert_policy_std=.001,
    downsample=False,
)

#model = torch.load('/2tb/home/patrickhaoy/iros_2022_sim_runs/1/itr_150.pt')
#policy = model['trainer/policy'].to('cuda:0')
#model = io_util.load_model('/2tb/home/patrickhaoy/iros_2022_sim_runs/pretrained_ptp')
#vqvae = model['vqvae'].to('cuda:0')
#itr_0_plan = np.load('/2tb/home/patrickhaoy/iros_2022_sim_runs/1/itr_0_plan.npy').squeeze()
#itr_150_plan = np.load('/2tb/home/patrickhaoy/iros_2022_sim_runs/1/itr_150_plan.npy').squeeze()
actions = np.load('/2tb/home/patrickhaoy/iros_2022_sim_runs/1_gcp/itr_150_actions.npy') ## CHANGE

save_video = True

video_save_path = '/2tb/home/patrickhaoy/iros_2022_sim_runs/1_gcp/' ## CHANGE
if save_video:
    num_traj = 1
    observations = np.zeros((num_traj*ts, 512, 512, 3))
else:
    #actions_save_path = '/2tb/home/patrickhaoy/iros_2022_sim_runs/1/'
    num_traj = 1
    actions = np.zeros((num_traj*ts, 5))

tasks_success = dict()
tasks_count = dict()
for i in range(num_traj):
    print(i)
    env.demo_reset()
    curr_task = env.curr_task
    is_done = False
    for t in range(ts):
        img = np.uint8(env.render_obs())
        if save_video:
            # from PIL import Image
            # im = Image.fromarray(img)
            # im.save(video_save_path + "debug.jpeg")
            # exit()
            observations[i*ts + t, :] = img
        #action, done = env.get_demo_action(first_timestep=(t == 0), return_done=True)
        done = False
        #img = img / 255.0
        #img = img.transpose(2, 1, 0)
        #latent = vqvae.encode_np(img).squeeze().reshape(1, 720)
        #idx = min(t // 30, 7)
        #subgoal = itr_150_plan[idx].reshape(1, 720)
        #obs = np.concatenate((latent, subgoal), axis=1)
        #obs = ptu.from_numpy(obs)
       # action = ptu.get_numpy(policy(obs).mean.squeeze())
        #print(action)
        action = actions[i*ts + t]
        if not save_video:
            actions[i*ts + t, :] = action
        next_observation, reward, _, info = env.step(action)
        if done and not is_done:
            is_done = True 
            
            if curr_task not in tasks_success.keys():
                tasks_success[curr_task] = 1
            else:
                tasks_success[curr_task] += 1
    
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
    writer = skvideo.io.FFmpegWriter(video_save_path + "itr_0.mp4") ## CHANGE
    for i in range(num_traj*ts):
        writer.writeFrame(observations[i, :, :, :])
    writer.close()
else:
    np.save(video_save_path + "itr_150_actions", actions)
