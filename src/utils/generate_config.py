import copy
from typing import Dict
import ray
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from src.environment.configs import PLAYABLE_MAPS, ENV_TRAINING_METRICS_CALLBACKS, MELTING_POT_CONFIG_FUNCS, \
    DEFAULT_ENV_SETTINGS
from src.models.model_catalog import get_model
from src.training.callbacks.enhanced_self_play import EnhancedSelfPlayCallback
from src.training.callbacks.pooling_self_play import PoolingSelfPlayCallback
from src.utils.helpers import get_trailing_number, merge_dict
from training.callbacks.population_self_play import PopulationSelfPlayCallback, training_policy_mapping_fn
from training.policies.idle_policy import IdlePolicy


def patch_run_config(run_config: Dict) -> Dict:
    """
    Patches an old run config to include default values for missing fields. This is useful for backwards compatibility.
    @param run_config:
    @return:
    """
    default_values = {
        'model_sharing': 'independent' if 'multi_model' in run_config and run_config['multi_model']
        else 'shared',
        'solo_play': False,
        'population_self_play': {'policy_map_capacity': 100} if run_config['population_self_play'] else None,
        'env_settings': DEFAULT_ENV_SETTINGS[run_config['env']]
    }
    return merge_dict(default_values, run_config)


def get_alg_config(run_config: Dict, training=True) -> Dict:
    # The agent algorithm to use
    # PPO
    config = copy.deepcopy(
        ray.rllib.algorithms.ppo.PPOConfig().to_dict()
    )

    # The RL Module API is a bleeding edge experimental API that was introduced with Ray 2.6.0.
    # Using it restricts the available algorithms and models. For those reasons it is disabled
    # until the API is more mature. See https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html
    # Also it seems that checkpoints generated via this API are insanely large (1 GB, compared to 20 MB)
    # so until that is fixed, it's not worth using.
    config['_enable_rl_module_api'] = False
    config['_enable_learner_api'] = False

    # Choose a map to play
    map_name = run_config['map_name']
    if map_name not in PLAYABLE_MAPS[run_config['env']]:
        raise Exception(f"Map named {map_name} not found."
                        f" Available maps are: {', '.join(list(PLAYABLE_MAPS[run_config['env']].keys()))}")
    else:
        play_map = PLAYABLE_MAPS[run_config['env']][map_name]
    # Get model description
    model_description = get_model(run_config['model_name'])

    # Set environment config
    env_config = MELTING_POT_CONFIG_FUNCS[run_config['env']](
        roles=run_config['agent_roles'],
        play_map=play_map['map'],
        env_settings=run_config['env_settings'])

    # Do not preprocess observations and provide from environment as is.
    config['preprocessor_pref'] = None

    config['env_config'] = env_config
    env_constructor = model_description['env_constructor']
    # Register env constructor and name
    register_env('meltingpot', env_constructor)
    config['env'] = 'meltingpot'

    # Extract space dimensions
    test_env = env_constructor(config['env_config'])

    # Specify policy to model mapping
    match run_config['model_sharing']:
        # Every agent gets a separate policy
        case 'independent':
            policies = {f'policy_{i}': PolicySpec(
                policy_class=None,  # use default policy
                observation_space=test_env.observation_space[f'player_{i}'],
                action_space=test_env.action_space[f'player_{i}'],
                config={},
            ) for i in range(0, len(env_config['roles']))}
            config['multiagent'] = {
                'policies': policies,
                'policy_mapping_fn': lambda agent_id, episode, worker,
                                            **kwargs: f'policy_{get_trailing_number(agent_id)}',
                'count_steps_by': 'env_steps',
            }
        # Every agent shares the same policy
        case 'shared':
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
        # When using one policy per role
        case 'one-per-role':
            role_types = list(set(env_config['roles']))
            policies = {f'policy_{r}': PolicySpec(
                policy_class=None,  # use default policy
                observation_space=test_env.observation_space[f'player_0'],
                action_space=test_env.action_space[f'player_0'],
                config={},
            ) for r in role_types}

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                agent_idx = get_trailing_number(agent_id)
                role = env_config['roles'][agent_idx]
                return f'policy_{role}'

            config['multiagent'] = {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
                'count_steps_by': 'env_steps',
            }
        case _:
            raise Exception(f"Model sharing mode {run_config['model_sharing']} not recognised.")

    # Set policy config for population-self-play case:
    if run_config['population_self_play']:
        assert run_config[
                   'model_sharing'] == 'independent', 'Population self-play only supports independent model sharing'
        policies = {f'policy_{player}_p{pop}': PolicySpec(
            policy_class=None,  # use default policy
            observation_space=test_env.observation_space[f'player_{player}'],
            action_space=test_env.action_space[f'player_{player}'],
            config={},
        ) for player in range(len(env_config['roles']))
            for pop in range(run_config['population_self_play']['population_per'])}
        config['multiagent'] = {
            'policies': policies,
            'policy_mapping_fn': training_policy_mapping_fn,
            'count_steps_by': 'env_steps',
        }

    if run_config['solo_play']:
        policies = {f'policy_{i}': PolicySpec(
            policy_class=None if i == 0 else IdlePolicy,  # use default policy and make all other agents idle
            observation_space=test_env.observation_space[f'player_{i}'],
            action_space=test_env.action_space[f'player_{i}'],
            config={},
        ) for i in range(0, len(env_config['roles']))}
        config['multiagent'] = {
            'policies_to_train': ['policy_0'],
            'policies': policies,
            'policy_mapping_fn': lambda agent_id, episode, worker,
                                        **kwargs: f'policy_{get_trailing_number(agent_id)}',
            'count_steps_by': 'env_steps',
        }

    # Set number of GPUs to use
    config['num_gpus'] = run_config['num_gpus']
    # Set log level ERROR, INFO or DEBUG
    config['log_level'] = 'DEBUG' if run_config['debug_mode'] else 'ERROR'
    config['num_workers'] = run_config['num_workers']
    config['num_gpus_per_worker'] = run_config['num_gpus_per_worker']
    config['max_episode_steps'] = config['env_config'].lab2d_settings['maxEpisodeLengthFrames']
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
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.

    # Set model config
    config['model'] = model_description['model_config']

    # Choose whether to add callbacks that collect more training metrics
    chosen_callbacks = []
    if ENV_TRAINING_METRICS_CALLBACKS[run_config['env']]:
        chosen_callbacks.append(ENV_TRAINING_METRICS_CALLBACKS[run_config['env']])
    # Choose whether to add callbacks that impact opponent selection for training
    if run_config['generalising_self_play'] and training:
        chosen_callbacks.append(EnhancedSelfPlayCallback)
    elif run_config['pooled_self_play'] and training:
        chosen_callbacks.append(PoolingSelfPlayCallback)
    elif run_config['population_self_play'] and training:
        chosen_callbacks.append(PopulationSelfPlayCallback)

    config['callbacks'] = make_multi_callbacks(chosen_callbacks)

    config['framework'] = run_config['framework']  # set tensorflow or torch framework
    config['num_cpus_per_worker'] = run_config['cpus_per_worker']

    if run_config['framework'] == 'tf2':
        # Speeds up tf evaluation, but makes debugging difficult
        config['eager_tracing'] = not run_config['debug_mode']

    # Sets the number of environments to explore simultaneously. Not Sure if it is beneficial to change this.
    config['num_envs_per_worker'] = run_config['num_parallel_envs'] // run_config['num_workers']

    if run_config['pooled_self_play']:
        config['evaluation_num_workers'] = run_config['pooled_self_play']['evaluation_num_workers']
        config['evaluation_interval'] = run_config['pooled_self_play']['evaluation_interval']

    if run_config['population_self_play']:
        config['create_env_on_driver'] = True
        config['multiagent']['policy_map_capacity'] = run_config['population_self_play']['policy_map_capacity']

    config = {k: v for k, v in config.items() if v is not None}  # filter out None values
    return config
