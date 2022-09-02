import copy
from typing import Dict
import ray
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from king_of_the_hill.custom_metrics import MetricsCallbacks
from king_of_the_hill.playable_maps import PLAYABLE_MAPS
from king_of_the_hill.preprocessor import FrameStacker
from meltingpot.python import substrate
from meltingpot.python.configs.substrates.king_of_the_hill import get_config
import re


# Gets trailing number from a string.
# E.g "Bananas123" yields 123
def get_trailing_number(s: str) -> int:
    m = re.search(r'\d+$', s)
    if m:
        return int(m.group())
    else:
        raise Exception(f"No number found at the end of '{s}'")


# A wrapper constructor
def wrapped_env_constructor(env_config: ConfigDict):
    """Outputs an environment for registering."""
    env = substrate.build(config_dict.ConfigDict(env_config))
    env = FrameStacker(env)
    return env


def get_alg_config(run_config: Dict) -> Dict:
    # The agent algorithm to use
    # PPO
    config = copy.deepcopy(
        ray.rllib.agents.ppo.PPOConfig().to_dict()
    )

    # Choose a map to play
    map_name = run_config['map_name']
    if map_name not in PLAYABLE_MAPS:
        raise Exception(f"Map named {map_name} not found."
                        f" Available maps are: {', '.join(list(PLAYABLE_MAPS.keys()))}")

    # Set environment config
    env_config = get_config(num_players=run_config['num_agents'],
                            play_map=PLAYABLE_MAPS[map_name]['map'],
                            map_dimensions=PLAYABLE_MAPS[map_name]['dimensions'])
    config['env_config'] = env_config

    # Register env constructor and name
    register_env('meltingpot', wrapped_env_constructor)
    config['env'] = 'meltingpot'

    # Extract space dimensions
    test_env = wrapped_env_constructor(config['env_config'])

    # Specify policy
    # Every agent gets a separate policy
    if run_config['multi_model']:
        policies = {f'policy{i}': PolicySpec(
            policy_class=None,  # use default policy
            observation_space=test_env.observation_space[f'player_{i}'],
            action_space=test_env.action_space[f'player_{i}'],
            config={},
        ) for i in range(0, run_config['num_agents'])}
        config['multiagent'] = {
            'policies': policies,
            'policy_mapping_fn': lambda agent_id, episode, worker, **kwargs: f'policy{get_trailing_number(agent_id)}',
            'count_steps_by': 'env_steps',
        }
    # Every agent shares the same policy
    else:
        config['multiagent'] = {
            'policies': {
                'av':
                    PolicySpec(
                        policy_class=None,  # use default policy
                        observation_space=test_env.observation_space['player_0'],
                        action_space=test_env.action_space['player_0'],
                        config={},
                    ),
            },
            'policy_mapping_fn': lambda agent_id, episode, worker, **kwargs: 'av',
            'count_steps_by': 'env_steps',
        }

    # Set number of GPUs to use
    config['num_gpus'] = run_config['num_gpus']
    # Set log level ERROR, INFO or DEBUG
    config['log_level'] = 'DEBUG' if run_config['debug_mode'] else 'ERROR'
    config['num_workers'] = run_config['num_workers']
    # After n steps, force end episode
    config['horizon'] = config['env_config'].lab2d_settings[
        'maxEpisodeLengthFrames']
    # each unroll happens exactly over one episode
    # ensuring that we have a full episode allows a reward to be calculated
    config['batch_mode'] = 'complete_episodes'
    # Fragment length, in this case the number of episodes
    # collected at once from each worker and for each agent!
    config['rollout_fragment_length'] = 1
    config['train_batch_size'] = config['env_config'].lab2d_settings[
        'maxEpisodeLengthFrames']
    # fix the batch size for a training pass
    # Default: 128
    config['sgd_minibatch_size'] = run_config['sgd_minibatch_size']
    # Default: False
    config['no_done_at_end'] = False
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.

    # Setup for the neural network.
    config['model'] = {}
    config['callbacks'] = MetricsCallbacks
    # The strides of the first convolutional layer were chosen to perfectly line
    # up with the sprites, which are 8x8.
    # The final layer must be chosen specifically so that its output is
    # [B, 1, 1, X]. See the explanation in
    # https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
    # because rllib is unable to flatten to a vector otherwise.
    # The a3c models used as baselines in the meltingpot paper were not run using
    # rllib, so they used a different configuration for the second convolutional
    # layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.
    config['model']['conv_filters'] = [[16, [8, 8], 8], [128, [11, 11], 1]]
    config['model']['conv_activation'] = 'relu'
    config['model']['post_fcnet_hiddens'] = [256]
    config['model']['post_fcnet_activation'] = 'relu'
    # config['model']['use_lstm'] = True
    # config['model']['lstm_use_prev_action'] = True
    # config['model']['lstm_use_prev_reward'] = False
    # config['model']['lstm_cell_size'] = 256

    # config['framework'] = 'torch'  # set pytorch framework
    config['framework'] = 'tf2'  # set tensorflow framework
    config['num_cpus_per_worker'] = run_config['cpus_per_worker']
    # Speeds up tf evaluation, but makes debugging difficult
    config['eager_tracing'] = not run_config['debug_mode']
    # Sets the number of environments to explore simultaneously. Not Sure if it is beneficial to change this.
    config['num_envs_per_worker'] = 1

    config = {k: v for k, v in config.items() if v is not None}  # filter out None values
    return config
