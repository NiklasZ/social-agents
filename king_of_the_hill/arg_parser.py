import argparse
from time import time


def get_run_args():
    parser = argparse.ArgumentParser(description='King of the Hill Agent')
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
    parser_new.add_argument('--map-name', type=str, required=False, default='default',
                            help='name of the map to play KotH on. New maps can be added to the PLAYABLE_MAPS dict')
    parser_new.add_argument('--multi-model', action=argparse.BooleanOptionalAction, required=False, default=False,
                            help='if set lets each agent have its own model instance for both'
                                 ' value function and policy. '
                                 'Otherwise, uses a shared model instance for every agent')
    # Shared arguments
    for p in [parser_resume, parser_new]:
        p.add_argument('--wandb-project-name', required=False, type=str, default='social_marl_koth',
                       help="the wandb's project name")
        p.add_argument('--debug-mode', action=argparse.BooleanOptionalAction, required=False, default=False,
                       help='toggle in order to allow debugging of code by running it on a local process.')

    args = parser.parse_args()

    if args.command == 'new':
        args.rllib_run_name = f'{args.run_name}-{int(time())}'  # Attach initial run timestamp for convenience later

        if not args.group_name:
            args.group_name = args.map_name

    return args
