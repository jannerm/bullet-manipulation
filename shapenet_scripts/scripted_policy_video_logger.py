import skvideo.io
import roboverse
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

from roboverse.envs.env_list import *
from scripted_collect import *
from suboptimal_scripted_collect import *

env_to_policy_map = {
    frozenset(V6_GRASPING_V0_DRAWER_PLACING_OPENING_ENVS): scripted_grasping_V6_place_then_open_V0,
    frozenset(V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_GRASPING_ENVS):scripted_grasping_V6_close_open_grasp_V0,
    frozenset(V6_GRASPING_V0_DOUBLE_DRAWER_OPENING_ENVS):scripted_grasping_V6_double_drawer_open_grasp_V0,
    frozenset(V6_GRASPING_V0_DRAWER_GRASPING_ONLY_ENVS):scripted_grasping_V6,
    frozenset(V6_GRASPING_V0_PLACING_ENVS):scripted_grasping_V6_placing_V0,
    frozenset(V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS):scripted_grasping_V6_opening_only_V0,
    frozenset(V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_ENVS): scripted_grasping_V6_double_drawer_close_open_V0,
    frozenset(V6_GRASPING_V0_DOUBLE_DRAWER_PICK_PLACE_OPEN_ENVS): scripted_grasping_V6_double_drawer_pick_place_open_V0,
}

env_to_suboptimal_policy_map = {
    frozenset(V6_GRASPING_V0_DOUBLE_DRAWER_CLOSING_OPENING_ENVS): suboptimal_scripted_grasping_V6_double_drawer_close_open_V0,
    frozenset(V6_GRASPING_V0_DRAWER_OPENING_ONLY_ENVS): suboptimal_scripted_grasping_V6_opening_only_V0,
}

class BulletVideoLogger:
    def __init__(self, env_name, video_save_dir, success_only, suboptimal_policy, noise=0.2):
        self.env_name = env_name
        self.noise = noise
        self.video_save_dir = video_save_dir
        self.success_only = success_only
        self.suboptimal_policy = suboptimal_policy
        self.image_size = 512

        if not os.path.exists(self.video_save_dir):
            os.makedirs(self.video_save_dir)
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
        self.trajectories_collected = 0

    def instantiate_env(self):
        env = roboverse.make(self.env_name, reward_type="sparse",
                       gui=False, observation_mode="pixels_debug",
                       transpose_image=True)
        return env

    def get_scripted_policy_func(self, env_name):
        if self.suboptimal_policy:
            env_policy_map = env_to_suboptimal_policy_map
        else:
            env_policy_map = env_to_policy_map
        for env_group in env_policy_map.keys():
            if env_name in env_group:
                return env_to_policy_map[env_group]
        raise NotImplementedError

    def get_single_path_pool_and_success(self):
        self.trajectories_collected += 1
        print("trajectories collected", self.trajectories_collected)
        pool_size = self.env.scripted_traj_len + 1
        obs_keys = (self.env.cnn_input_key, self.env.fc_input_key)
        railrl_pool = ObsDictReplayBuffer(pool_size, self.env,
            observation_keys=obs_keys)
        railrl_success_pool = ObsDictReplayBuffer(pool_size, self.env,
            observation_keys=obs_keys)
        self.scripted_policy_func(
            self.env, args.env, railrl_pool, railrl_success_pool, noise=self.noise)
        success = railrl_pool._rewards[self.env.scripted_traj_len - 1]
        print(railrl_pool._rewards.T)
        return railrl_pool, success

    def get_single_path_pool(self):
        if self.success_only:
            success = False
            while not success:
                railrl_pool, success = self.get_single_path_pool_and_success()
            print("collected success", railrl_pool, success)
        else:
            railrl_pool, _ = self.get_single_path_pool_and_success()
        return railrl_pool

    def save_video_from_path(self, single_path_pool, path_idx):
        actions = single_path_pool._actions
        assert self.env.scripted_traj_len < actions.shape[0]
        print("single_path_pool._actions.shape", single_path_pool._actions.shape)
        images = []
        reward_list = []
        self.env.reset()
        for t in range(self.env.scripted_traj_len):
            img, depth, segmentation = roboverse.bullet.render(
                self.image_size, self.image_size,
                self.view_matrix, self.projection_matrix)
            images.append(img)
            obs, rew, done, info, imgs = self.env.step_slow(
                actions[t], self.image_size,
                self.view_matrix, self.projection_matrix)
            if len(imgs) > 0:
                images.extend(imgs)

        save_path = "{}/{}_scripted_reward_{}_{}.mp4".format(
            self.video_save_dir, self.env_name, rew, path_idx)
        inputdict = {'-r': str(12)}
        outputdict = {'-vcodec': 'libx264', '-pix_fmt': 'yuv420p'}
        writer = skvideo.io.FFmpegWriter(
            save_path, inputdict=inputdict, outputdict=outputdict)
        for i in range(len(images)):
            writer.writeFrame(images[i])
        writer.close()


    def save_videos(self, num_videos):
        for i in range(num_videos):
            single_path_pool = self.get_single_path_pool()
            self.save_video_from_path(single_path_pool, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--video-save-dir", type=str, default="scripted_rollouts")
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--success-only", action="store_true", default=False)
    parser.add_argument("--use-suboptimal-policy", action="store_true", default=False)
    # Currently, success-only collects only successful trajectories,
    # but these trajectories do not always succeed again due to randomized initial conditions
    args = parser.parse_args()

    vid_log = BulletVideoLogger(
        args.env, args.video_save_dir, args.success_only, args.use_suboptimal_policy)
    vid_log.save_videos(args.num_videos)
