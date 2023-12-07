import argparse
import json
from datetime import datetime
from time import time
from typing import List

from src.environment.configs import DEFAULT_ENV_SETTINGS, ENV_ALIASES, MELTING_POT_CONFIG_FUNCS
from src.environment.helpers import validate_environment_roles
from src.models.model_catalog import MODELS

default_pooled_self_play_settings = json.dumps(
    {'evaluation_num_workers': 8,
     'mastered_pool_max_prob': 0.4,  # probability of selecting opponents from mastery pool
     'reward_distance': 30,  # distance in reward for an agent to be considered having "won"
     'consecutive_training_wins_needed': 3,  # how many training episodes an agent needs to win to master an opponent
     'consecutive_eval_wins_needed': 5,  # how many evaluation episodes an agent needs to win to master an opponent
     # We never want rllib to evaluate automatically in this mode, but the framework will not run unless we specify one.
     # This is stupid and shouldn't be.
     'evaluation_interval': 99999999999999999,
     # Whether to initialise agents with the same weights or not. Note, this is not the same as seeding. Rather, we copy
     # the weights from one policy to all the others at the start.
     'init_with_shared_weights': False,
     # Who the first frozen opponent to train against should be. Can be "random policy" or "frozen main policy".
     'init_opponents': 'frozen main policy',
     # Determines how to sample an opponent from a pool. Supported options are:
     # "uniform": every opponent has the same probability
     # "inverse last minimum train reward": samples inversely with respect to the lowest reward achieved against that
     # opponent the last time they played during training. Effectively, opponents the main policy has recently gotten a
     # high reward against will be sampled less often than those with a low reward.
     # This approach is very responsive to recent changes in performance.
     'mastered_pool_sampling': 'uniform',
     'competing_pool_sampling': 'uniform'
     }
)

default_population_self_play_settings = json.dumps({
    'population_per': 4,  # number of agents per player, role or overall. This depends on the model sharing mode.
    'min_games_against_new_opponent': 3,  # how many times to pick new opponents before sampling from older ones again.
    # how many recent rewards to consider when calculating the average reward against an opponent.
    'recent_reward_window': 5,
    # rate of average reward improvement needed for a policy to considered sufficiently improved in a curriculum iteration.
    # E.g if a policy has an initial average episode reward of 10, and the improvement factor is 1.1,
    # then the policy needs reach a reward of 11.
    'policy_improvement_factor': 1.1,
    # ratio of policies that need to be sufficiently improved in order for the population to move to the next curriculum iteration.
    'sufficient_improved_population_ratio': 0.5,
    # After a sufficient number of policies have improved, control how many more timesteps the remaining
    # policies have to meet their improvement target before dropping them.
    'curriculum_end_timesteps': 50,
})

# Here the configurations have the following meaning:
MODEL_SHARING_OPTIONS = ['shared',  # all agents have the same underlying model.
                         'one-per-role',  # all agents share a model for each type of role defined by melting pot.
                         'independent']  # all agents have their own model.


def merge_input_json_with_default_json(custom_json: str, default_json: str, setting_name='') -> dict:
    custom_settings = json.loads(custom_json.replace("\'", "\""))
    default_settings = json.loads(default_json)
    for key in custom_settings.keys():
        if key not in default_settings:
            raise Exception(f"Unknown f{setting_name} key '{key}'")
    return default_settings | custom_settings


def generate_run_id(run_name):
    """
    Generates a run id based on the current time and the run name.
    @param run_name:
    @return:
    """
    # Reference time: January 1, 2022, in milliseconds - DO NOT CHANGE
    reference_date = datetime(2022, 1, 1)
    reference_time_milliseconds = int(reference_date.timestamp() * 1000)

    # Current time in milliseconds
    current_time_milliseconds = int(time() * 1000)

    # Time difference in milliseconds
    time_difference = current_time_milliseconds - reference_time_milliseconds

    # Convert to hexadecimal
    hex_time = hex(time_difference)[2:]  # [2:] to remove the '0x' prefix

    # Combine with the run name
    run_id = f'{run_name}-{hex_time}'
    return run_id


def get_train_args(arg_list: List[str] = None):
    parser = argparse.ArgumentParser(description='Melting Pot Agent Training')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    subparsers.required = True

    # NOTE: No longer maintained
    # # Arguments to resume a run
    # parser_resume = subparsers.add_parser('resume', help='when you want to resume training an existing agent.')
    # parser_resume.add_argument('--run-id', type=str, required=False, default=None,
    #                            help='supply the wandb ID of the run to resume.')
    # parser_resume.add_argument('--entity-name', type=str, required=False, default='ece-239-as',
    #                            help="name of the entity whose run should be resumed")

    # Arguments for a new run
    parser_new = subparsers.add_parser('new', help='when you want to train a new agent.')
    parser_new.add_argument('--run-name', type=str, required=False, default='run',
                            help='the name of this training run')
    parser_new.add_argument('--group-name', type=str, required=False, default=None,
                            help='name of the group of runs this one belongs to')
    # parser_new.add_argument('--num-agents', type=int, required=False, default=2,
    #                         help='number of agents that should play in the environment')
    parser_new.add_argument('--agent-roles', type=str, nargs='+', required=False, default=None,
                            help='List of strings describing the roles that the agents in the environment. '
                                 'Note that the number of roles listed also dictates how many agents '
                                 'will play in the environment overall.')
    parser_new.add_argument('--sgd-minibatch-size', type=int, required=False, default=128,
                            help='How many sampled environment steps to use per gradient descent training batch.')
    parser_new.add_argument('--training-iterations', type=int, required=False, default=10000,
                            help='For how many epochs the model should train. Usually equivalent to the number of '
                                 'episodes')
    parser_new.add_argument('--checkpoint-frequency', type=int, required=False, default=50,
                            help='How frequently (in terms of epochs) a model should be checkpointed')
    parser_new.add_argument('--num-gpus', type=float, required=False, default=0,
                            help='how many GPUS to use')
    parser_new.add_argument('--num-workers', type=int, required=False, default=1,
                            help='how many worker processes to use. Use only 1 when debugging')
    parser_new.add_argument('--cpus-per-worker', type=int, required=False, default=None,
                            help='how cpus to allocate per worker process')
    parser_new.add_argument('--num-gpus-per-worker', type=float, required=False, default=0,
                            help='how many gpus to allocate per worker sampling the environment. '
                                 'Note: this can be fractional (e.g 0.1)')
    parser_new.add_argument('--num-parallel-envs', type=int, required=False,
                            help='how many envs to sample in parallel. If not set, defaults to number of workers')
    parser_new.add_argument('--map-name', type=str, required=False, default='default',
                            help='name of the map to play KotH on. New maps can be added to the PLAYABLE_MAPS dict')
    parser_new.add_argument('--model-name', type=str, required=True,
                            help='Which model to use for training. Available models are: '
                                 f"{', '.join(MODELS.keys())}")
    parser_new.add_argument('--model-sharing', type=str, required=False, default='independent',
                            help='determines whether agents should share models or not. '
                                 f"Available options are: {', '.join(MODEL_SHARING_OPTIONS)}")
    parser_new.add_argument('--generalising-self-play', action=argparse.BooleanOptionalAction, required=False,
                            default=False,
                            help='makes agent play against itself and past agents. Cannot be used alongside')
    parser_new.add_argument('--pooled-self-play', type=str, required=False, default=None,
                            help='settings JSON for the pooled-self-play. '
                                 f"Default is: {default_pooled_self_play_settings}")
    parser_new.add_argument('--population-self-play', type=str, required=False, default=None,
                            help='Lets agents learn together as a population. '
                                 f"Default setting for this is: {default_population_self_play_settings}")
    parser_new.add_argument('--episodes-per-training-iteration', type=int, required=False, default=4,
                            help='how many episodes to collect before training. Affects the training batch size')
    parser_new.add_argument('--framework', type=str, required=False, default='torch',
                            help="Deep learning framework to use. Must be either 'torch' or 'tf2'.")
    parser_new.add_argument('--env', type=str, required=True,
                            help=f'Environment to play in. Supported environments are: {list(DEFAULT_ENV_SETTINGS.keys())}')
    parser_new.add_argument('--env-settings', type=str, required=False,
                            help='custom settings to set for the environment (should be a JSON). ')
    parser_new.add_argument('--obs-filter', type=str, required=False, default="NoFilter",
                            help="Whether to perform any normalisation on the observation before passing it to an agent."
                                 "A common example is 'MeanStdFilter'")
    parser_new.add_argument('--wandb-project-name', required=False, type=str, default='social_marl_koth',
                            help="the wandb's project name")
    parser_new.add_argument('--debug-mode', action=argparse.BooleanOptionalAction, required=False, default=False,
                            help='toggle in order to allow debugging of code by running it on a local process.')
    parser_new.add_argument('--rllib-path', required=False, type=str, default='/tmp/ray_results',
                            help='directory where to save rllib training states to')

    args = parser.parse_args(arg_list)

    if args.command == 'new':
        args.run_id = generate_run_id(args.run_name)
        args.rllib_run_name = args.run_id # Kept for backwards compatibility - should be removed later

        if args.env not in DEFAULT_ENV_SETTINGS:
            if args.env in ENV_ALIASES:
                args.env = ENV_ALIASES[args.env]
            else:
                raise Exception(
                    f'Unsupported environment {args.env}. Supported environments are {list(DEFAULT_ENV_SETTINGS.keys())}')

        # If a custom environment config is provided, check it is valid and merge it with the default config.
        if args.env_settings:
            args.env_settings = merge_input_json_with_default_json(args.env_settings,
                                                                   json.dumps(DEFAULT_ENV_SETTINGS[args.env]),
                                                                   'env_settings')
        else:
            args.env_settings = DEFAULT_ENV_SETTINGS[args.env]

        if not args.group_name:
            args.group_name = args.map_name

        args.group_name = f'{ENV_ALIASES[args.env]}-{args.group_name}'

        if args.agent_roles:
            # Check role definitions are valid
            validate_environment_roles(args.env, args.agent_roles)

        if not args.num_parallel_envs:
            args.num_parallel_envs = args.num_workers

        if args.framework != 'torch':
            raise Exception("Deep learning framework must be 'torch' as 'tf2' is not supported yet.")

        if sum(bool(e) for e in [args.pooled_self_play, args.generalising_self_play, args.pooled_self_play]) > 1:
            raise Exception('Only one self-play callback method can be enabled at a time.')

        if args.pooled_self_play or args.generalising_self_play:
            raise Exception('Advanced self-play methods are outdated and need to be re-implemented.')

        if args.pooled_self_play and not args.multi_model:
            raise Exception('Pooled self-play currently only supports multiple models.')

        # Check pooled self-play dict to make sure keys are valid.
        if args.pooled_self_play:
            args.pooled_self_play = merge_input_json_with_default_json(args.pooled_self_play,
                                                                       default_pooled_self_play_settings,
                                                                       'pooled_self_play')

        if args.population_self_play:
            args.population_self_play = merge_input_json_with_default_json(args.population_self_play,
                                                                           default_population_self_play_settings,
                                                                           'population_self_play')

        if args.model_sharing not in MODEL_SHARING_OPTIONS:
            raise Exception(f"Model sharing must be one of {MODEL_SHARING_OPTIONS}")

    return args
