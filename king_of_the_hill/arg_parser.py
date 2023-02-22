import argparse
import json
from time import time
from king_of_the_hill.models import MODELS

default_env_settings = json.dumps(
    {'rewardPerStepInControl': 1.0,  # reward per timestep controlling the hill
     'penaltyForBeingZapped': 0,  # reward for being zapped
     'rewardForZapping': 0,  # reward for zapping
     'framesTillRespawn': 80,  # how many timesteps until an agent respawns after being defeated
     }
)

default_pooled_self_play_settings = json.dumps(
    {'evaluation_num_workers': 8,
     'mastered_pool_max_prob': 0.4,  # probability of selecting opponents from mastery pool
     'reward_distance': 30,  # distance in reward for an agent to be considered having "won"
     'consecutive_training_wins_needed': 3,  # how many training episodes an agent needs to win to master an opponent
     'consecutive_eval_wins_needed': 5,  # how many evaluation episodes an agent needs to win to master an opponent
     }
)

# Maps a particular layer name to the index of the respective dimension in the activation
supported_activation_layers = {
    'lstm_hidden': 0,
    'lstm_cell': 1,
    'rnn': 0
}


def get_train_args():
    parser = argparse.ArgumentParser(description='King of the Hill Agent Training')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    subparsers.required = True

    # Arguments to resume a run
    parser_resume = subparsers.add_parser('resume', help='when you want to resume training an existing agent.')
    parser_resume.add_argument('--run-id', type=str, required=False, default=None,
                               help='supply the wandb ID of the run to resume.')
    parser_resume.add_argument('--entity-name', type=str, required=False, default='ece-239-as',
                               help="name of the entity whose run should be resumed")

    # Arguments for a new run
    parser_new = subparsers.add_parser('new', help='when you want to train a new agent.')
    parser_new.add_argument('--run-name', type=str, required=False, default='run',
                            help='the name of this training run')
    parser_new.add_argument('--group-name', type=str, required=False, default=None,
                            help='name of the group of runs this one belongs to')
    parser_new.add_argument('--num-agents', type=int, required=False, default=2,
                            help='number of agents that should play in the environment')
    parser_new.add_argument('--sgd-minibatch-size', type=int, required=False, default=128,
                            help='How many sampled environment steps to use per gradient descent training batch.')
    parser_new.add_argument('--training-iterations', type=int, required=False, default=10000,
                            help='For how many epochs the model should train. Usually equivalent to the number of '
                                 'episodes')
    parser_new.add_argument('--checkpoint-frequency', type=int, required=False, default=50,
                            help='How frequently (in terms of epochs) a model should be checkpointed')
    parser_new.add_argument('--num-gpus', type=int, required=False, default=0,
                            help='how many GPUS to use')
    parser_new.add_argument('--num-workers', type=int, required=False, default=1,
                            help='how many worker processes to use. Use only 1 when debugging')
    parser_new.add_argument('--cpus-per-worker', type=int, required=False, default=None,
                            help='how cpus to allocate per worker process')
    parser_new.add_argument('--num-parallel-envs', type=int, required=False,
                            help='how many envs to sample in parallel. If not set, defaults to number of workers')
    parser_new.add_argument('--map-name', type=str, required=False, default='default',
                            help='name of the map to play KotH on. New maps can be added to the PLAYABLE_MAPS dict')
    parser_new.add_argument('--model-name', type=str, required=True,
                            help='Which model to use for training. Available models are: '
                                 f"{', '.join(MODELS.keys())}")
    parser_new.add_argument('--multi-model', action=argparse.BooleanOptionalAction, required=False, default=False,
                            help='if set lets each agent have its own model instance for both'
                                 ' value function and policy. '
                                 'Otherwise, uses a shared model instance for every agent')
    parser_new.add_argument('--generalising-self-play', action=argparse.BooleanOptionalAction, required=False,
                            default=False,
                            help='makes agent play against itself and past agents. Cannot be used alongside --multi-model')
    parser_new.add_argument('--pooled-self-play', type=str, required=False, default=None,
                            help='settings JSON for the curriculum pooled-self-play. '
                            f"Default is: {default_pooled_self_play_settings}")
    parser_new.add_argument('--episodes-per-training-iteration', type=int, required=False, default=4,
                            help='how many episodes to collect before training. Affects the training batch size')
    parser_new.add_argument('--framework', type=str, required=False, default='torch',
                            help="Deep learning framework to use. Must be either 'torch' or 'tf2'.")
    parser_new.add_argument('--env-settings', type=str, required=False, default=default_env_settings,
                            help='custom settings to set for the environment (should be a JSON). '
                                 f"Default is: {default_env_settings}")
    parser_new.add_argument('--obs-filter', type=str, required=False, default="NoFilter",
                            help="Whether to perform any normalisation on the observation before passing it to an agent."
                                 "A common example is 'MeanStdFilter'")

    # Shared arguments
    for p in [parser_resume, parser_new]:
        p.add_argument('--wandb-project-name', required=False, type=str, default='social_marl_koth',
                       help="the wandb's project name")
        p.add_argument('--debug-mode', action=argparse.BooleanOptionalAction, required=False, default=False,
                       help='toggle in order to allow debugging of code by running it on a local process.')
        p.add_argument('--rllib-path', required=False, type=str, default='/tmp/ray_results',
                       help='directory where to save rllib training states to')

    args = parser.parse_args()

    if args.command == 'new':
        args.rllib_run_name = f'{args.run_name}-{int(time())}'  # Attach initial run timestamp for convenience later

        if not args.group_name:
            args.group_name = args.map_name

        if not args.num_parallel_envs:
            args.num_parallel_envs = args.num_workers

        if args.multi_model and args.generalising_self_play:
            raise Exception('Cannot set both --multi-model and --generalising-self-play')

        if args.framework not in ['torch', 'tf2']:
            raise Exception("Deep learning framework must be either 'torch' or 'tf2'")

        if args.pooled_self_play and not args.multi_model:
            raise Exception('Pooled self-play currently only supports multiple models.')

        # Check pooled self-play dict to make sure keys are valid.
        if args.pooled_self_play:
            custom_pool_settings = json.loads(args.pooled_self_play.replace("\'", "\""))
            default_pooling = json.loads(default_pooled_self_play_settings)
            for key in custom_pool_settings.keys():
                if key not in default_pooling:
                    raise Exception(f"Unknown pooled self-play key '{key}'")
            args.pooled_self_play = default_pooling | custom_pool_settings

        # Check custom env dict to make sure keys are valid.
        custom_env_setting = json.loads(args.env_settings.replace("\'", "\""))
        default_env = json.loads(default_env_settings)
        for key in custom_env_setting.keys():
            if key not in default_env:
                raise Exception(f"Unknown reward key '{key}'")
        args.env_settings = default_env | custom_env_setting
    return args


def get_evaluation_args():
    parser = argparse.ArgumentParser(description='King of the Hill Agent Evaluation')
    parser.add_argument('--run-id', type=str, required=True,
                        help='supply the wandb ID of the run to resume.')
    parser.add_argument('--checkpoint-number', type=int, required=False,
                        help='the number of the checkpoint to evaluate')
    parser.add_argument('--checkpoint-range', type=int, nargs=3, required=False,
                        help="range of checkpoints to evaluate. This is similar to python's range(). "
                             "Requires 3 training iteration numbers:(start, end, interval). "
                             "For example, for checkpoints 10,20,30,40,50,60 using (20,60,20) will get"
                             "checkpoints 20 and 40")
    parser.add_argument('--episodes', type=int, required=False, default=1,
                        help='the number of episodes to evaluate')
    parser.add_argument('--record-observations', action=argparse.BooleanOptionalAction, required=False,
                        help="set in order to also get video recordings of agent's observations")
    parser.add_argument('--entity-name', type=str, required=False, default='ece-239-as',
                        help="name of the entity whose run should be resumed")
    parser.add_argument('--wandb-project-name', required=False, type=str, default='social_marl_koth',
                        help="the wandb's project name")
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, required=False,
                        help='set when you would like to log more information during the evaluation runs.')
    parser.add_argument('--recording-fps', type=int, required=False, default=20,
                        help="when recording, sets the number of steps per second to encode into video.")
    parser.add_argument('--debug-mode', action=argparse.BooleanOptionalAction, required=False, default=False,
                        help='toggle in order to allow debugging of code by running it on a local process.')
    parser.add_argument('--activations', type=str, required=False, nargs='+',
                        help=f'toggle in order to save activations for the provided layer names. '
                             f'Supported layer names are {list(supported_activation_layers.keys())}')
    parser.add_argument('--rllib-path', required=False, type=str, default='/tmp/ray_results',
                        help='directory where to save rllib training states to')
    parser.add_argument('--num-gpus', type=int, required=False, default=0,
                        help='how many GPUS to use')

    args = parser.parse_args()

    if not (args.checkpoint_number or args.checkpoint_range):
        raise Exception('Either --checkpoint-number or --checkpoint-range must be set')

    if args.checkpoint_number and args.checkpoint_range:
        raise Exception('Cannot set both --checkpoint-number and --checkpoint-range')

    if args.activations:
        for a in args.activations:
            if a not in list(supported_activation_layers.keys()):
                raise Exception(f'Unsupported activation layer {a}. Available layers are {supported_activation_layers}')
    return args
