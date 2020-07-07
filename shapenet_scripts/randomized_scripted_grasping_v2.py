import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time
import roboverse.bullet as bullet
import roboverse

OBJECT_NAME = 'lego'
EPSILON = 1e-8

def scripted_markovian(env, render_images, object_name):
    env.reset()

    object_info = bullet.get_body_info(env._objects[object_name],
                                       quat_to_deg=False)
    target_pos = np.asarray(object_info['pos'])

    # the object is initialized above the table, so let's compensate for it
    # target_pos[2] += -0.01
    print("target_pos: ", target_pos)
    images = []

    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    for i in range(args.num_timesteps):
        ee_pos = env.get_end_effector_pos()
        if i == 0:
            print("initial end effector pos: ", ee_pos)

        grip = 0

        action = target_pos - ee_pos
        action = action * 2
        #action = action / np.linalg.norm(action)
        action = np.append(action, [grip])
        action += np.random.normal(scale=0.025)
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        if render_images:
            img = env.render_obs()
            images.append(Image.fromarray(np.reshape(np.uint8(img), (48, 48, 3))))

        next_state, reward, done, info = env.step(action)
        # time.sleep(0.2)
        observation = next_state
        traj["observations"].append(observation)
        next_state, reward, done, info = env.step(action)
        traj["next_observations"].append(next_state)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)

        end_effector_pos = env.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(target_pos - end_effector_pos)
        object_gripper_distance_z = target_pos[2] - end_effector_pos[2]
        print("object_gripper_distance_z: ", object_gripper_distance_z)
        success = object_gripper_distance < 0.02
        if success:
            return success, images, traj
    return success, images, traj


def main(args):

    timestamp = roboverse.utils.timestamp()
    data_save_path = os.path.join(__file__, "../..", 'data',
                                  args.data_save_directory, timestamp)
    data_save_path = os.path.abspath(data_save_path)
    #video_save_path = os.path.join(data_save_path, "videos")
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    #if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
    #    os.makedirs(video_save_path)

    reward_type = 'sparse' if args.sparse else 'shaped'
    env = roboverse.make('SawyerGraspV2Reaching-v0', reward_type=reward_type,
                         gui=args.gui, randomize=True,
                         observation_mode="pixels_debug")

    all_trajs = []
    for j in tqdm(range(args.num_trajectories)):
        env.reset()
        render_images = args.video_save_frequency > 0 and \
                        j % args.video_save_frequency == 0
        obj = env.get_observation()
        
        all_obj_pos = []
        for o_id in range(env._num_objects):
            obj_pos = obj["state"][-(o_id + 1) * 7:-(o_id + 1) * 7 + 3]
            all_obj_pos.append([obj_pos[0], obj_pos[1], obj_pos[2]])
        env._trimodal_positions = all_obj_pos
        env.randomize = False

        for obj_id in range(3):
            for t in range(50):
                success, images, traj = scripted_markovian(env, render_images, obj_id)
                for i in range(5):
                    if success:
                        break
                    success, images, traj = scripted_markovian(env, render_images, obj_id)
                if success:
                    all_trajs.append(traj)
                else:
                    break
        print("len(all_trajs): ", len(all_trajs))

        env.randomize = True

    path = os.path.join(data_save_path, "reaching_random_in_graspingV2_env_{}.npy".format(timestamp))
    print(path)
    np.save(path, all_trajs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=5000)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--video_save_frequency", type=int,
                        default=1, help="Set to zero for no video saving")
    parser.add_argument("--randomize", dest="randomize",
                        action="store_true", default=False)
    parser.add_argument("--gui", dest="gui", action="store_true", default=False)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("--non-markovian", dest="non_markovian",
                        action="store_true", default=False)
    parser.add_argument("-o", "--observation-mode", type=str, default='pixels_debug',
                        choices=('state', 'pixel', 'pixels_debug'))

    args = parser.parse_args()

    main(args)