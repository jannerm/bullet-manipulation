import roboverse as rv
import numpy as np
import skvideo.io
from PIL import Image

# from experiments.kuanfang.iql.drawer_pnp_commands import drawer_pnp_commands
# from experiments.kuanfang.iql.drawer_pnp_single_obj_commands import drawer_pnp_single_obj_commands
from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands
from roboverse.envs.configs.drawer_pnp_push_env_configs import drawer_pnp_push_env_configs
from rlkit.util.io import load_local_or_remote_file
import rlkit.torch.pytorch_util as ptu
ptu.set_gpu_mode(1)

ts = 200
num_traj = 2
SWITCH_SUBGOAL_EVERY = 25
subgoal_idx = -1

env = rv.make(
    "SawyerRigAffordances-v6", 
    gui=False, 
    expl=True, 
    reset_interval=1, 
    env_obs_img_dim=196, 
    test_env=True, 
    test_env_command=drawer_pnp_push_commands[31],
    demo_num_ts=ts,
    expert_policy_std=.05,
    downsample=False,
)

save_video = False

video_save_path = '/media/ashvin/data1/patrickhaoy/data/test/'
observations = np.zeros((num_traj*ts, 196, 196, 3))

pretrained_rl_path = '/media/ashvin/data1/patrickhaoy/corl2022/sim_models/task31/itr_150.pt'
rl_model_dict = load_local_or_remote_file(pretrained_rl_path)
policy = rl_model_dict['trainer/policy']
obs_encoder = rl_model_dict['trainer/obs_encoder']

goal_path = '/media/ashvin/data1/patrickhaoy/corl2022/sim_models/td_pnp_push_goals_seed31.pkl'
goal_data = np.load(goal_path, allow_pickle=True)
image_plan = goal_data['image_plan'][4]
vib_plan = obs_encoder.encode_np(image_plan)

for i in range(num_traj):
    print(i)
    o = env.reset()
    for t in range(ts):
        if t % SWITCH_SUBGOAL_EVERY == 0:
            subgoal_idx += 1
            # subgoal_idx = -1
            print(f"SUBGOAL{subgoal_idx}")
        latent_subgoal = vib_plan[subgoal_idx]

        img = np.uint8(env.render_obs())
        observations[i*ts + t, :] = img

        img_downsampled = Image.fromarray(np.uint8(img), 'RGB').resize((48, 48), resample=Image.ANTIALIAS)
        img_downsampled = np.array(img_downsampled) / 255.0
        img_downsampled = img_downsampled.transpose(2, 1, 0).flatten()
        # np.save('img.npy', img_downsampled)
        # np.save('img_plan.npy', image_plan)
        # assert False
        latent_downsampled = obs_encoder.encode_one_np(img_downsampled)
        input_to_policy = np.concatenate((latent_downsampled, latent_subgoal), axis=0)[None]
        input_to_policy = ptu.from_numpy(input_to_policy)
        action = ptu.get_numpy(policy(input_to_policy).mean).flatten()
        action = np.random.normal(action, .05)
        o, reward, _, info = env.step(action)

writer = skvideo.io.FFmpegWriter(video_save_path + "debug.mp4")
for i in range(num_traj*ts):
    writer.writeFrame(observations[i, :, :, :])
writer.close()

np.save('observations.npy', observations)
np.save('goals.npy', image_plan[-1])