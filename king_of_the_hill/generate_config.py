import copy
from typing import Dict
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from king_of_the_hill.callbacks.custom_metrics import MetricsCallbacks
from king_of_the_hill.callbacks.enhanced_self_play import EnhancedSelfPlayCallback
from king_of_the_hill.models import get_model
from king_of_the_hill.maps import PLAYABLE_MAPS
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


def get_alg_config(run_config: Dict, training=True) -> Dict:

    # Support for generating evaluation configs for older agents whose configs may not have these new fields.
    if not training:
        default_values = {'episodes_per_training_iteration': 4,
                          'framework': 'tf2'}
        run_config = run_config | default_values

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

    # Get model description
    model_description = get_model(run_config['model_name'])

    # Set environment config
    env_config = get_config(num_players=run_config['num_agents'],
                            play_map=PLAYABLE_MAPS[map_name]['map'],
                            map_dimensions=PLAYABLE_MAPS[map_name]['dimensions'])
    config['env_config'] = env_config
    env_constructor = model_description['env_constructor']
    # Register env constructor and name
    register_env('meltingpot', env_constructor)
    config['env'] = 'meltingpot'

    # Extract space dimensions
    test_env = env_constructor(config['env_config'])

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
    # Fragment length, in this case the number of steps
    # collected at once from each worker and for each agent!
    # Ignored when 'batch_mode' = 'complete_episodes'
    config['rollout_fragment_length'] = 1

    episodes_per_batch = run_config['episodes_per_training_iteration'] // run_config['num_workers']
    check_val = run_config['episodes_per_training_iteration'] / run_config['num_workers']
    assert episodes_per_batch == check_val, f"Number of episodes per batch ({run_config['episodes_per_training_iteration']}) " \
                                            f"needs to be divisible by number of workers ({run_config['num_workers']})"

    # Controls total batch size of a training iteration.
    config['train_batch_size'] = config['env_config'].lab2d_settings[
                                     'maxEpisodeLengthFrames'] * episodes_per_batch
    # fix the batch size for a training pass
    # Default: 128
    config['sgd_minibatch_size'] = run_config['sgd_minibatch_size']
    # Default: False
    config['no_done_at_end'] = False
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.

    # Set model config
    config['model'] = model_description['model_config']
    # Choose whether to play against older policies during training to generalise more.
    config['callbacks'] = EnhancedSelfPlayCallback if training and run_config[
                                                          'generalising_self_play'] else MetricsCallbacks

    config['framework'] = run_config['framework']  # set tensorflow or torch framework
    config['num_cpus_per_worker'] = run_config['cpus_per_worker']

    if run_config['framework'] == 'tf2':
        # Speeds up tf evaluation, but makes debugging difficult
        config['eager_tracing'] = not run_config['debug_mode']

    # Sets the number of environments to explore simultaneously. Not Sure if it is beneficial to change this.
    config['num_envs_per_worker'] = run_config['num_parallel_envs'] // run_config['num_workers']

    config = {k: v for k, v in config.items() if v is not None}  # filter out None values
    return config
