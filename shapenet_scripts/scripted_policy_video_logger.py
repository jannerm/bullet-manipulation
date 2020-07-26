import skvideo.io

import roboverse
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

from roboverse.envs.env_list import *
from scripted_collect import *

env_to_policy_map = {
    frozenset(V6_GRASPING_V0_DRAWER_PLACING_OPENING_ENVS): scripted_grasping_V6_place_then_open_V0,
}

class BulletVideoLogger:
    def __init__(self, env_name, video_save_dir, noise=0.2):
        self.env_name = env_name
        self.noise = noise
        self.video_save_dir = video_save_dir
        self.image_size = 512
        # camera settings
        self.camera_target_pos = [1.05, -0.05, -0.1]
        self.camera_pitch = -30
        self.camera_distance = 0.1
        self.view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance, yaw=90,
                                pitch=self.camera_pitch, roll=0, up_axis_index=2)
        self.view_matrix = roboverse.bullet.get_view_matrix(**self.view_matrix_args)
        self.projection_matrix = roboverse.bullet.get_projection_matrix(self.image_size, self.image_size)
        # end camera settings
        self.env = self.instantiate_env()
        self.scripted_policy_func = self.get_scripted_policy_func(env_name)

    def instantiate_env(self):
        env = roboverse.make(self.env_name, reward_type="sparse",
                       gui=False, observation_mode="pixels_debug",
                       transpose_image=True)
        return env

    def get_scripted_policy_func(self, env_name):
        for env_group in env_to_policy_map.keys():
            if env_name in env_group:
                return env_to_policy_map[env_group]

    def get_single_path_pool(self):
        pool_size = self.env.scripted_traj_len + 1
        obs_keys = (self.env.cnn_input_key, self.env.fc_input_key)
        railrl_pool = ObsDictReplayBuffer(pool_size, self.env,
            observation_keys=obs_keys)
        railrl_success_pool = ObsDictReplayBuffer(pool_size, self.env,
            observation_keys=obs_keys)
        self.scripted_policy_func(
            self.env, railrl_pool, railrl_success_pool, noise=self.noise)
        return railrl_pool

    def save_video_from_path(self, single_path_pool, path_idx):
        actions = single_path_pool._actions
        assert path_idx * self.env.scripted_traj_len < actions.shape[0]
        print("single_path_pool._actions.shape", single_path_pool._actions.shape)
        images = []
        reward_list = []
        self.env.reset()
        for t in range(self.env.scripted_traj_len):
            img, depth, segmentation = roboverse.bullet.render(
                self.image_size, self.image_size,
                self.view_matrix, self.projection_matrix)
            images.append(img)
            action = actions[path_idx * self.env.scripted_traj_len + t]
            obs, rew, done, info = self.env.step(actions[t])

        save_path = "{}/eval_{}.mp4".format(self.video_save_dir, path_idx)
        if not os.path.exists(args.video_save_dir):
            os.makedirs(args.video_save_dir)
        skvideo.io.vwrite(save_path, images)



    def save_videos(self, num_videos):
        for i in range(num_videos):
            single_path_pool = self.get_single_path_pool()
            self.save_video_from_path(single_path_pool, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--video-save-dir", type=str, default="scripted_rollouts")
    parser.add_argument("--num-videos", type=int, default=1)
    args = parser.parse_args()

    vid_log = BulletVideoLogger(args.env, args.video_save_dir)
    vid_log.save_videos(args.num_videos)
