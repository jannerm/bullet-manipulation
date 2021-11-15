import os

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
from env_utils import make_env_and_dataset, sample
from evaluation import evaluate
from pixel_iql.learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_integer('max_episode_steps', 50, 'Episode length.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'configs/bc_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    wandb.init(project="bridge",
               config={
                   "seed": FLAGS.seed,
                   "use_data_aug": FLAGS.config.use_data_aug,
                   "dropout": FLAGS.config.dropout_rate,
                   "share_encoder": FLAGS.config.share_encoder
               })

    env, dataset = make_env_and_dataset(None, FLAGS.seed,
                                        FLAGS.max_episode_steps)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis], **kwargs)

    for i in tqdm.tqdm(range(FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = sample(dataset, FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            wandb.log(update_info)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            wandb.log(eval_stats)


if __name__ == '__main__':
    app.run(main)
