from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from pixel_iql.common import default_init
from pixel_iql.policy import NormalTanhPolicy
from pixel_iql.value_net import DoubleCritic, ValueCritic


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (3, 3, 3, 2)
    strides: Sequence[int] = (2, 2, 3, 3)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        for features, filter, stride in zip(self.features, self.filters,
                                            self.strides):
            x = nn.Conv(features,
                        kernel_size=(filter, filter),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class DrQDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_filters: Sequence[int] = (3, 3, 3, 3)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_filters,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return DoubleCritic(self.hidden_dims)(x, actions)


class DrQValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_filters: Sequence[int] = (3, 3, 3, 3)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = Encoder(self.cnn_features,
                    self.cnn_filters,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return ValueCritic(self.hidden_dims)(x)


class DrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_filters: Sequence[int] = (3, 3, 3, 3)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50
    dropout_rate: float = 0.1
    share_encoder: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        x = Encoder(self.cnn_features,
                    self.cnn_filters,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        if self.share_encoder:
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        policy_def = NormalTanhPolicy(self.hidden_dims,
                                      self.action_dim,
                                      log_std_scale=1e-3,
                                      log_std_min=0.0,
                                      log_std_max=0.0,
                                      dropout_rate=self.dropout_rate,
                                      state_dependent_std=False,
                                      tanh_squash_distribution=False)

        return policy_def(x, temperature=temperature, training=training)
