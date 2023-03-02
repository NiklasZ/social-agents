# Describes models to use for training.
# NOTE: try not to change existing entries and instead add new ones.
# Every entry should contain enough information to create and train the model. This means things such as:
# - hyper-parameters
# - environment preprocessors

from ml_collections.config_dict import config_dict
from examples.rllib.utils import MeltingPotEnv
from king_of_the_hill.preprocessor import FrameStacker, RGBObservationWrapper, FullRGBObservationWrapper
from meltingpot.python.substrate import get_factory_from_config
from ml_collections import config_dict


def get_model(name: str):
    assert name in MODELS, f"Model {name} not found in available models: {', '.join(MODELS.keys())}"
    return MODELS[name]


def get_cnn_env_constructor(stack_frames: int):
    def constructor(env_config):
        config = config_dict.ConfigDict(env_config)
        env = get_factory_from_config(config).build([])
        env = MeltingPotEnv(env)
        env = RGBObservationWrapper(env)
        env = FrameStacker(env, stack_frames=stack_frames)
        return env

    return constructor


def get_fully_observable_cnn_env_constructor(world_height: int, world_width: int, stack_frames: int):
    def constructor(env_config):
        config = config_dict.ConfigDict(env_config)
        env = get_factory_from_config(config).build([])
        env = MeltingPotEnv(env)
        env = FullRGBObservationWrapper(env, world_height, world_width)
        env = FrameStacker(env, stack_frames=stack_frames)
        return env

    return constructor


def get_cnn_lstm_constructor():
    def constructor(env_config):
        config = config_dict.ConfigDict(env_config)
        env = get_factory_from_config(config).build([])
        env = MeltingPotEnv(env)
        env = RGBObservationWrapper(env)
        return env

    return constructor


MODELS = {
    'basic_cnn_4x': {
        'env_constructor': get_cnn_env_constructor(4),
        'model_config': {
            'conv_filters': [[16, [8, 8], 8], [128, [11, 11], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [256],
            'post_fcnet_activation': 'relu'
        },
        'notes': 'This is a 2-layer CNN, where the first layer has 16 kernels, of 8x8 size, with a stride of 8. '
                 'The stride was deliberately chosen to be the same as the size of 8x8 sprites. As a result, each'
                 'convolution with a kernel is of a sprite.'
    },
    'basic_cnn_8x': {
        'env_constructor': get_cnn_env_constructor(8),
        'model_config': {
            'conv_filters': [[16, [8, 8], 8], [128, [11, 11], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [256],
            'post_fcnet_activation': 'relu'
        },
        'notes': 'Same as basic_cnn_4x, but we stack 8 frames for a longer memory on the agent'
    },
    'fully_obs_cnn_8x': {
        'env_constructor': get_fully_observable_cnn_env_constructor(13, 13, 8),
        'model_config': {
            'conv_filters': [[16, [8, 8], 8], [128, [13, 13], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [256],
            'post_fcnet_activation': 'relu'
        },
        'notes': '8x frame-stacked CNN that uses a fully observable 13x13 map. '
    },
    'basic_cnn_lstm': {
        'env_constructor': get_cnn_lstm_constructor(),
        'model_config': {
            'conv_filters': [[16, [8, 8], 8], [128, [11, 11], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [256],
            'post_fcnet_activation': 'relu',
            'use_lstm': True,
            'lstm_use_prev_action': True,
            'lstm_use_prev_reward': False,
            'lstm_cell_size': 256
        },
        'notes': 'CNN combined with an LSTM, requiring no framestacking'
    },
    'chunky_cnn_lstm': {
        'env_constructor': get_cnn_lstm_constructor(),
        'model_config': {
            'conv_filters': [[32, [8, 8], 8], [256, [11, 11], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [512],
            'post_fcnet_activation': 'relu',
            'use_lstm': True,
            'lstm_use_prev_action': True,
            'lstm_use_prev_reward': False,
            'lstm_cell_size': 512
        },
        'notes': 'CNN combined with an LSTM, requiring no framestacking'
    },
    'basic_cnn_lstm_rewards': {
        'env_constructor': get_cnn_lstm_constructor(),
        'model_config': {
            'conv_filters': [[16, [8, 8], 8], [128, [11, 11], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [256],
            'post_fcnet_activation': 'relu',
            'use_lstm': True,
            'lstm_use_prev_action': True,
            'lstm_use_prev_reward': True,
            'lstm_cell_size': 256
        },
        'notes': 'CNN combined with an LSTM with previous reward fed into network'
    },
}
