import roboverse as rv

env = rv.make(
    "SawyerRigAffordances-v0", 
    gui=True, 
    expl = False,
    random_color_p=1,
    max_episode_steps = 150,
    obs_img_dim = 64,
    claw_spawn_mode='fixed', #Maybe Uniform Instead, so distractor is more distracting....?
    #drawer_yaw_setting = (0, 360),
    demo_action_variance = 0.1,
    color_range = (0, 255//2),
    #max_distractors = 4,
    #reset_interval=1,
    test_env=True,
    env_type='top_drawer',
    swap_eval_task=True,
)
ts = 150

for i in range(100):
    env.demo_reset()
    for t in range(ts):
        action = env.get_demo_action(first_timestep=(t == 0), final_timestep=(t == ts - 1))
        next_observation, reward, done, info = env.step(action)
        if done:
            print(t)
            break
            #import pdb; pdb.set_trace()