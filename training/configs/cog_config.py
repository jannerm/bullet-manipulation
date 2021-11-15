import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.cnn_features = (16, 32, 64, 128)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = 'SAME'
    config.latent_dim = 50
    config.hidden_dims = (256, 256)

    config.use_data_aug = False
    config.share_encoder = False

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 10.0
    config.dropout_rate = config_dict.placeholder(
        float)  # Float argument, defauts to None

    config.tau = 0.005  # For soft target updates.

    return config
