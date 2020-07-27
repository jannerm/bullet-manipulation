import argparse
import time
import subprocess


def get_data_save_directory(args):
    data_save_directory = args.data_save_directory

    data_save_directory += '_{}_{}'.format(args.env, args.observation_mode)

    if args.num_trajectories > 1000:
        data_save_directory += '_{}K'.format(int(args.num_trajectories/1000))
    else:
        data_save_directory += '_{}'.format(args.num_trajectories)

    if args.suboptimal:
        data_save_directory += '_nonrelevant'

    if args.end_at_neutral:
        data_save_directory += '_end_at_neutral'

    data_save_directory += '_noise_std_{}'.format(args.noise_std)

    if args.sparse:
        data_save_directory += '_sparse_reward'
    elif args.semisparse:
        data_save_directory += '_semisparse_reward'
    else:
        data_save_directory += '_dense_reward'

    # if args.random_actions:
    #     data_save_directory += '_random_actions'
    # else:
    #     data_save_directory += '_scripted_actions'
    #
    # if args.randomize:
    #     data_save_directory += '_randomize'
    # else:
    #     data_save_directory += '_fixed_position'

    return data_save_directory


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, required=True)
    parser.add_argument("-d", "--data-save-directory", type=str)
    parser.add_argument("-n", "--num-trajectories", type=int, default=2000)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=10)
    parser.add_argument("--sparse", dest="sparse", action="store_true",
                        default=False)
    parser.add_argument("--semisparse", dest="semisparse", action="store_true",
                        default=False)
    parser.add_argument("--randomize", dest="randomize", action="store_true",
                        default=False)
    parser.add_argument("--random_actions", dest="random_actions",
                        action="store_true", default=False)
    parser.add_argument("-o", "--observation-mode", type=str, default='pixels')
    parser.add_argument("--allow-grasp-retries", dest="allow_grasp_retries",
                        action="store_true", default=False)
    parser.add_argument("--joint-norm-thresh", dest="joint_norm_thresh",
                        type=float, default=0.05)
    parser.add_argument("--one-reset-per-traj", dest="one_reset_per_traj",
                        action="store_true", default=False)
    parser.add_argument("--end-at-neutral", dest="end_at_neutral",
                        action="store_true", default=False)
    parser.add_argument("--suboptimal", dest="suboptimal",
                        action="store_true", default=False)
    parser.add_argument("--success-only", action="store_true", default=False)
    args = parser.parse_args()

    assert args.semisparse != args.sparse

    num_trajectories_per_thread = int(
        args.num_trajectories / args.num_parallel_threads)
    if args.num_trajectories % args.num_parallel_threads != 0:
        num_trajectories_per_thread += 1
    save_directory = get_data_save_directory(args)
    if args.suboptimal:
        script_name = 'suboptimal_scripted_collect.py'
    else:
        script_name = 'scripted_collect.py'
    command = ['python',
               'shapenet_scripts/{}'.format(script_name),
               '-e{}'.format(args.env),
               '-d{}'.format(save_directory),
               '--noise-std',
               str(args.noise_std),
               '-n {}'.format(num_trajectories_per_thread),
               '-p {}'.format(args.num_parallel_threads),
               '-o{}'.format(args.observation_mode),
               '-j {}'.format(args.joint_norm_thresh),
               ]
    if args.sparse:
        command.append('--sparse')
    if args.semisparse:
        command.append('--semisparse')
    if args.randomize:
        command.append('--randomize')
    if args.random_actions:
        command.append('--random_actions')
    if args.allow_grasp_retries:
        command.append('--allow-grasp-retries')
    if args.one_reset_per_traj:
        command.append('--one-reset-per-traj')
    if args.end_at_neutral:
        command.append('--end-at-neutral')

    subprocesses = []
    for i in range(args.num_parallel_threads):
        subprocesses.append(subprocess.Popen(command))
        time.sleep(1)

    exit_codes = [p.wait() for p in subprocesses]

    # subprocess.call(['python',
    #                  'shapenet_scripts/combine_trajectories.py',
    #                  '-d{}'.format(save_directory)]
    #                 )
    merge_command = ['python',
                     'shapenet_scripts/combine_railrl_pools.py',
                     '-d{}'.format(save_directory),
                     '-o{}'.format(args.observation_mode),
                     '-e{}'.format(args.env)]
    if args.success_only:
        merge_command.append('--success-only')

    subprocess.call(merge_command)

    print(exit_codes)
