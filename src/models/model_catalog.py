# Describes models to use for training.
# NOTE: try not to change existing entries and instead add new ones.
# Every entry should contain enough information to create and train the model. This means things such as:
# - hyper-parameters
# - environment preprocessors

from ml_collections.config_dict import config_dict
from src.deepmind.examples.rllib.utils import MeltingPotEnv
from src.deepmind.meltingpot.substrate import get_factory_from_config
from ml_collections import config_dict
from ray.rllib.models import ModelCatalog

from src.environment.wrappers import RGBObservationWrapper, FrameStacker, FullRGBObservationWrapper, \
    DeepmindObservationWrapper
from src.models.baseline_cnn_lstm import BaselineCNNLSTM
from src.models.baseline_cnn_fc_lstm import BaselineCNNFCLSTM
from src.models.baseline_test_lstm import BaselineCNNFCLSTM_test
from src.models.cnnfc_lstm import CNNFCLSTM_simplified
from src.models.cnnfc_lstm2 import CNNFCLSTM_simplified as CNNFCLSTM_simplified2
from src.models.baseline_cnn_lstm_addobs import BaselineCNNLSTM as BaselineCNNLSTM_addobs


def register_models():
    ModelCatalog.register_custom_model(
        'baseline_cnn_lstm', BaselineCNNLSTM,
    )
    ModelCatalog.register_custom_model(
        'baseline_cnn_fc_lstm', BaselineCNNFCLSTM,
    )
    ModelCatalog.register_custom_model(
        'baseline_test_lstm', BaselineCNNFCLSTM_test,
    )
    ModelCatalog.register_custom_model(
        'cnnfc_lstm', CNNFCLSTM_simplified,
    )
    ModelCatalog.register_custom_model(
        'cnnfc_lstm2', CNNFCLSTM_simplified2,
    )
    ModelCatalog.register_custom_model(
        'baseline_cnn_lstm_addobs', BaselineCNNLSTM_addobs,
    )


register_models()


def get_model(name: str):
    assert name in MODELS, f"Model {name} not found in available models: {', '.join(MODELS.keys())}"
    return MODELS[name]

def deepmind_observation_constructor():
    def constructor(env_config):
        config = config_dict.ConfigDict(env_config)
        # config['default_player_roles'] = env_config['roles']
        # print(config)
        # There was an issue when build()
        env = get_factory_from_config(config).build(env_config['roles'])
        env = MeltingPotEnv(env)
        env = DeepmindObservationWrapper(env)
        return env
    return constructor

def stacked_rgb_observation_constructor(stack_frames: int):
    def constructor(env_config):
        config = config_dict.ConfigDict(env_config)
        env = get_factory_from_config(config).build([])
        env = MeltingPotEnv(env)
        env = RGBObservationWrapper(env)
        env = FrameStacker(env, stack_frames=stack_frames)
        return env

    return constructor


def stacked_rgb_world_constructor(world_height: int, world_width: int, stack_frames: int):
    def constructor(env_config):
        config = config_dict.ConfigDict(env_config)
        env = get_factory_from_config(config).build([])
        env = MeltingPotEnv(env)
        env = FullRGBObservationWrapper(env, world_height, world_width)
        env = FrameStacker(env, stack_frames=stack_frames)
        return env

    return constructor


def single_rgb_observation_constructor():
    def constructor(env_config):
        config = config_dict.ConfigDict(env_config)
        env = get_factory_from_config(config).build(env_config['roles'])
        env = MeltingPotEnv(env)
        env = RGBObservationWrapper(env)
        return env

    return constructor


MODELS = {
    'baseline_cnn_fc_lstm': {
      'env_constructor': deepmind_observation_constructor(),
      'model_config': {
        'custom_model': 'baseline_cnn_fc_lstm',
        '_disable_preprocessor_api': True,  # prevents flattening of the observation via preprocessor
        'custom_model_config': {}
      },
    },
    'baseline_test_lstm': {
      'env_constructor': deepmind_observation_constructor(),
      'model_config': {
        'custom_model': 'baseline_test_lstm',
        '_disable_preprocessor_api': True,  # prevents flattening of the observation via preprocessor
        'custom_model_config': {}
      },
    },
    'cnnfc_lstm': {
      'env_constructor': deepmind_observation_constructor(),
      'model_config': {
        'custom_model': 'cnnfc_lstm',
        '_disable_preprocessor_api': True,  # prevents flattening of the observation via preprocessor
        'custom_model_config': {}
      },
    },
    'cnnfc_lstm2': {
        'env_constructor': deepmind_observation_constructor(),
        'model_config': {
            'custom_model': 'cnnfc_lstm2',
            '_disable_preprocessor_api': True,  # prevents flattening of the observation via preprocessor
            'custom_model_config': {}
        },
    },

    'baseline_cnn_lstm_addobs': {
        'env_constructor': deepmind_observation_constructor(),
        'model_config': {
            'custom_model': 'baseline_cnn_lstm_addobs',
            '_disable_preprocessor_api': True,  # prevents flattening of the observation via preprocessor
            'custom_model_config': {}
        },
    },

    'baseline_cnn_lstm': {
        'env_constructor': single_rgb_observation_constructor(),
        'model_config': {
            'custom_model': 'baseline_cnn_lstm',
            '_disable_preprocessor_api': True,  # prevents flattening of the observation via preprocessor
            'custom_model_config': {}
        },
        'notes': 'This is a baseline cnn-lstm with the same composition as the one described in the meltingpot paper:'
                 'https://browse.arxiv.org/pdf/2211.13746.pdf'
    },

    'basic_cnn_4x': {
        'env_constructor': stacked_rgb_observation_constructor(4),
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
        'env_constructor': stacked_rgb_observation_constructor(8),
        'model_config': {
            'conv_filters': [[16, [8, 8], 8], [128, [11, 11], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [256],
            'post_fcnet_activation': 'relu'
        },
        'notes': 'Same as basic_cnn_4x, but we stack 8 frames for a longer memory on the agent'
    },
    'fully_obs_cnn_8x': {
        'env_constructor': stacked_rgb_world_constructor(13, 13, 8),
        'model_config': {
            'conv_filters': [[16, [8, 8], 8], [128, [13, 13], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [256],
            'post_fcnet_activation': 'relu'
        },
        'notes': '8x frame-stacked CNN that uses a fully observable 13x13 map. '
    },
    'basic_cnn_lstm': {
        'env_constructor': single_rgb_observation_constructor(),
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
        'env_constructor': single_rgb_observation_constructor(),
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
        'notes': 'larger CNN combined with LSTM'
    },
    'basic_cnnx4_lstmx2': {
        'env_constructor': single_rgb_observation_constructor(),
        'model_config': {
            'conv_filters': [[64, [8, 8], 8], [512, [11, 11], 1]],
            'conv_activation': 'relu',
            'post_fcnet_hiddens': [512],
            'post_fcnet_activation': 'relu',
            'use_lstm': True,
            'lstm_use_prev_action': True,
            'lstm_use_prev_reward': False,
            'lstm_cell_size': 512
        },
        'notes': 'larger CNN combined with LSTM'
    },
    'basic_cnn_lstm_rewards': {
        'env_constructor': single_rgb_observation_constructor(),
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
