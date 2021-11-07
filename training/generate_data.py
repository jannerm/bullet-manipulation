import os
import shutil
import time

import numpy as np
import roboverse as rv
from gym.wrappers import TimeLimit

from training.chunk_writer import ChunkWriter

#change to done 2 seperate
#spacemouse = rv.devices.SpaceMouse(DoF=6)
env = rv.make('RemoveLid-v0', gui=False)
#env = rv.make('RemoveLid-v0', gui=False)
# env = rv.make('MugDishRack-v0', gui=False)
# env = rv.make('FlipPot-v0', gui=True)

time_limit = 50
env = TimeLimit(env, max_episode_steps=time_limit)

data_folder = 'dataset'
if os.path.exists(data_folder):
    shutil.rmtree(data_folder)
os.makedirs(data_folder, exist_ok=True)

num_traj = 10
chunk_writter = ChunkWriter(data_folder, (time_limit + 1) * num_traj)

start = time.time()
for j in range(num_traj):
    env.reset()
    env.demo_reset()
    done = False
    returns = 0
    while not done:
        img = env.render_obs()
        # human_action = spacemouse.get_action()
        action, noisy_action = env.get_demo_action()
        # noisy_action = env.action_space.sample()

        next_observation, reward, done, info = env.step(noisy_action)
        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        chunk_writter.add_transiton(img, action, reward, mask, done)

        returns += reward

    img = env.render_obs()
    chunk_writter.add_transiton(img,
                                np.zeros_like(action),
                                0,
                                0,
                                False,
                                dummy=True)

    print(f"Episode {j} returns {returns}")

print('Simulation Time:', (time.time() - start) / num_traj)

path = './rollout.mp4'

# video = ImageSequenceClip(images, fps=24)
# video.write_videofile(path)
