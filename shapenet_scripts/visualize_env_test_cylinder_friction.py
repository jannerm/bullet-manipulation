import roboverse as rv
import numpy as np
import skvideo.io

# from experiments.kuanfang.iql.drawer_pnp_commands import drawer_pnp_commands
# from experiments.kuanfang.iql.drawer_pnp_single_obj_commands import drawer_pnp_single_obj_commands
from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands

ts = 75
num_traj = 100

scripted_actions = [[ 1.        , -1.        ,  1.        , -0.38197389,  1.        ],
 [-0.0680387 , -0.3162878 ,  0.47608402,  1.        ,  1.        ],
 [ 1.        , -0.32946565,  1.        ,  0.52306027, -1.        ],
 [ 1.        ,  1.        ,  0.45763562,  1.        , -1.        ],
 [ 0.56354435, -0.17839762, -1.        ,  0.10654536, -1.        ],
 [-1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
 [ 0.02114751,  0.9142159 ,  0.98673796, -0.92093719,  1.        ],
 [ 0.19506898, -0.56687077,  1.        ,  0.37727066,  0.14351878],
 [-0.10828301,  0.03878448,  0.65296343,  1.        , -0.59258125],
 [ 1.        , -1.        ,  1.        ,  0.82267811, -0.67268624],
 [ 1.        , -1.        ,  0.59302012, -0.9356188 , -1.        ],
 [ 0.94932255, -1.        ,  0.93823869,  1.        , -1.        ],
 [ 0.7371644 ,  0.46436993,  1.        ,  1.        , -1.        ],
 [ 1.        ,  0.44588208,  1.        , -0.47641037, -0.97629543],
 [-0.8978898 , -0.36525919, -0.95061198,  1.        ,  1.        ],
 [-0.62780453,  1.        ,  0.23725787,  1.        ,  1.        ],
 [ 0.53084827, -1.        , -1.        ,  0.15594954, -1.        ],
 [-1.        , -1.        , -1.        ,  1.        , -0.17346886],
 [-1.        , -0.32803218,  0.20959551, -0.66660338, -0.65343984],
 [-0.21160018, -1.        ,  0.15248572,  1.        , -1.        ],
 [-1.        ,  1.        , -0.72783436,  0.4920086 , -1.        ],
 [ 1.        , -0.95338302,  0.83360456, -1.        ,  0.09831024],
 [ 1.        , -1.        ,  0.92663782,  1.        , -0.53596842],
 [ 0.60456032,  0.44264058, -1.        , -0.49656827,  1.        ],
 [-0.4820325 ,  0.85805592, -1.        ,  0.13993829, -1.        ],
 [-1.        , -0.3349382 , -0.79810183,  0.03951688, -1.        ],
 [ 0.72694698, -1.        , -1.        ,  1.        , -1.        ],
 [-0.04763314,  1.        , -0.33552543,  1.        ,  1.        ],
 [-0.06451102, -0.61663528, -0.38434436,  1.        ,  0.87177212],
 [ 0.57977255, -1.        ,  1.        , -1.        , -1.        ],
 [ 0.50125556, -0.48173981,  0.79450878,  1.        ,  0.12334192],
 [ 0.3484175 , -0.58687907, -0.56158488,  0.85599172, -1.        ],
 [ 1.        ,  0.00226558, -1.        ,  0.63220337, -1.        ],
 [ 0.23845591,  0.97963238,  1.        , -1.        , -1.        ],
 [-1.        , -0.78549762,  1.        , -0.47436029, -1.        ],
 [ 1.        , -1.        ,  0.15539782,  0.48889616, -1.        ],
 [ 1.        , -1.        , -1.        , -0.76916074, -0.59690728],
 [ 0.69111508,  1.        ,  0.33791016,  1.        , -1.        ],
 [ 1.        ,  1.        , -0.16345362,  1.        , -1.        ],
 [ 1.        , -0.9983909 ,  0.20953994, -0.4469306 , -0.6683885 ],
 [-1.        ,  1.        , -1.        ,  1.        , -1.        ],
 [-0.16781372, -0.39778882, -1.        ,  0.88939469, -1.        ],
 [-0.3140202 ,  0.78656449, -1.        ,  0.62167268, -1.        ],
 [ 0.31828555, -0.39321832,  0.44136349, -1.        , -0.25972321],
 [ 1.        , -0.50445494, -1.        ,  0.84087437, -1.        ],
 [ 1.        ,  1.        , -0.59146376, -0.76114268, -1.        ],
 [-1.        ,  0.64799589, -1.        ,  1.        , -1.        ],
 [-1.        ,  1.        ,  1.        , -0.79981752, -1.        ],
 [-1.        , -1.        , -1.        ,  1.        , -0.18229342],
 [-0.73051761, -1.        ,  1.        ,  1.        , -1.        ],
 [-0.76289753,  0.48442463, -1.        , -1.        , -1.        ],
 [-1.        , -0.24266244, -0.4317757 ,  0.29034938, -1.        ],
 [-1.        , -0.47163896, -0.91941793, -0.93764971, -1.        ],
 [ 1.        , -0.34824804,  1.        ,  0.61403315,  1.        ],
 [ 0.27365055, -1.        ,  0.16284646,  1.        ,  0.14362363],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.        ,  0.        ,  1.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ],
 [ 0.23314331,  0.52015167,  0.        ,  0.        , -1.        ]]

#obs_img_dim=196, 
env = rv.make(
    "SawyerRigAffordances-v6", 
    gui=True, 
    expl=True, 
    reset_interval=4, 
    drawer_sliding=False, 
    env_obs_img_dim=196, 
    random_color_p=0.0, 
    test_env=True, 
    test_env_command=drawer_pnp_push_commands[1],
    use_single_obj_idx=1,
    #large_obj=False,
    demo_num_ts=ts,
    # version=5,
    #move_gripper_task=True,
    # use_trash=True,
    # fixed_drawer_yaw=24.18556394023222,
    # fixed_drawer_position=np.array([0.50850424, 0.11416014, -0.34]),
    expert_policy_std=1.5,
    downsample=False,
)

save_video = False

if save_video:
    video_save_path = '/2tb/home/patrickhaoy/data/test/'
    num_traj = 1
    observations = np.zeros((num_traj*ts, 196, 196, 3))

tasks_success = dict()
tasks_count = dict()
#actions = []
for i in range(num_traj):
    print(i)
    env.demo_reset()
    curr_task = env.curr_task
    is_done = False
    for t in range(ts):
        if save_video:
            img = np.uint8(env.render_obs())
            # from PIL import Image
            # im = Image.fromarray(img)
            # im.save(video_save_path + "debug.jpeg")
            # exit()
            observations[i*ts + t, :] = img
        action, done = env.get_demo_action(first_timestep=(t == 0), return_done=True)
        #actions.append(action)
        action = scripted_actions[t]
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
    #print(np.array2string(np.array(actions), separator=', '))
    #actions = []
    print()

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
