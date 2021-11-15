"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pixel_iql import policy
from pixel_iql.actor import update as awr_update_actor
from pixel_iql.augmentations import batched_random_crop
from pixel_iql.common import Batch, InfoDict, Model, PRNGKey
from pixel_iql.critic import update_q, update_v
from pixel_iql.drq_networks import DrQDoubleCritic, DrQPolicy, DrQValueCritic


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=('use_data_aug', 'share_encoder'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float, use_data_aug: bool,
    share_encoder: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    if use_data_aug:
        rng, key = jax.random.split(rng)
        observations = batched_random_crop(key, batch.observations)
        rng, key = jax.random.split(rng)
        next_observations = batched_random_crop(key, batch.next_observations)

        batch = batch._replace(observations=observations,
                               next_observations=next_observations)

    new_value, value_info = update_v(target_critic, value, batch, expectile)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    if share_encoder:
        # Use critic conv layers in actor:
        new_actor_params = actor.params.copy(
            add_or_replace={
                'SharedEncoder': new_critic.params['SharedEncoder']
            })
        actor = actor.replace(params=new_actor_params)

    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, batch, temperature)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_filters: Sequence[int] = (3, 3, 3, 3),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 hidden_dims: Sequence[int] = (256, 256),
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 opt_decay_schedule: str = "cosine",
                 share_encoder: bool = False,
                 use_data_aug: bool = False):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        self.use_data_aug = use_data_aug
        self.share_encoder = share_encoder

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = DrQPolicy(hidden_dims,
                              action_dim,
                              cnn_features,
                              cnn_filters,
                              cnn_strides,
                              cnn_padding,
                              latent_dim,
                              dropout_rate=dropout_rate,
                              share_encoder=share_encoder)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, int(1e6))
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_filters,
                                     cnn_strides, cnn_padding, latent_dim)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = DrQValueCritic(hidden_dims, cnn_features, cnn_filters,
                                   cnn_strides, cnn_padding, latent_dim)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.temperature,
            self.use_data_aug, self.share_encoder)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info
