import numpy as np
import roboverse as rv
import roboverse.bullet as bullet
import pdb
import cv2

render = True 
env = rv.make('SawyerLiftGC-v0', goal_mult=4, action_scale=.1, action_repeat=10,
# env = rv.make('SawyerLift2d-v0', goal_mult=4, action_scale=.1, action_repeat=10,
              timestep=1./120, gui=render)


obs = env.reset()
# for _ in range(100):
    # img = env.render(mode='rgb_array').transpose().flatten() / 255.0
# cv2.imwrite('test.png', img.reshape(3, env._img_dim, env._img_dim).transpose() * 255)
while True:
    total_dists = []
    for _ in range(20):
        action = env.action_space.sample()
        # action[:3] = 1
        action[3] = 1
        obs, reward, done, info = env.step(action)
        dist = obs['state_observation'][-1]
        # print(dist)
        total_dists.append(dist)
        # bullet.get_index_by_attribute(env._sawyer, 'link_name', 'right_gripper_l_finger')
    print(np.array(total_dists).mean())
    env.reset()
