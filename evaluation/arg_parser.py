import argparse
import json
import shlex
from typing import Optional

# Maps a particular layer name to the property name and index of the respective dimension in the activation
supported_activation_layers = {
    'lstm_hidden': ['lstm', 0],
    'lstm_cell': ['lstm', 1],
    'rnn': ['rnn', 0],
    'action_logits': ['_logits_branch', 0]
}

# # Policies can either be matched up pairwise. E.g: [['policy0','policy1'],['policy0_v0','policy1_v0']]
# # or any vs. any. E.g:
# # ['policy0','policy1','policy0_v0','policy1_v0'] <=>
# # [['policy0','policy1'],['policy0','policy1_v0'], ['policy0_v0','policy1'], ['policy0_v0','policy1_v0']]
# # Where we skip cases where policies play against versions of themselves or as a player they weren't trained to play.
# default_multi_policy = json.dumps({
#     'matchups': [['policy0', 'policy1']],
# })

example_policy_structure = json.dumps({
    'policy_id': 'policy0',
    'run_id': 'c6bd8_00000',
    'checkpoint': 1000,
})


def get_evaluation_args(arg_str: Optional[str] = None):
    parser = argparse.ArgumentParser(description='King of the Hill Agent Evaluation')
    parser.add_argument('--policies', type=str, required=False, nargs='+',
                        help='A list of JSON strings detailing which policies to play against each other'
                             f"An example structure is: {example_policy_structure}")
    parser.add_argument('--episodes', type=int, required=False, default=1,
                        help='the number of episodes to evaluate')
    parser.add_argument('--record-episodes', action=argparse.BooleanOptionalAction, required=False, default=False,
                        help='whether or not to record videos of episodes played')
    parser.add_argument('--record-observations', action=argparse.BooleanOptionalAction, required=False, default=False,
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
    parser.add_argument('--score-for-dormant-activations', action=argparse.BooleanOptionalAction, required=False, default=False,
                        help='Yields normalised scores of activations of neurons')
    parser.add_argument('--eval-folder', type=str, required=False, default='',
                        help='Add a custom sub-folder to add to the evaluation path, '
                             'changing it to "evaluation/eval_games/<env_name>/<sub-folder>/<run-id>"')

    args = parser.parse_args(shlex.split(arg_str)) if arg_str else parser.parse_args()

    if args.activations:
        for a in args.activations:
            if a not in list(supported_activation_layers.keys()):
                raise Exception(f'Unsupported activation layer {a}. Available layers are {supported_activation_layers}')

    policy_ids = set()
    policy_infos = []
    example_p = json.loads(example_policy_structure)
    for raw_p in args.policies:
        p = json.loads(raw_p.replace("\'", "\""))
        p_fields = set(p.keys())
        example_p_fields = set(example_p.keys())
        if p_fields != example_p_fields:
            raise Exception(
                f'Policy {p} is missing the following required fields: {example_p_fields.difference(p_fields)}')
        if p['policy_id'] in policy_ids:
            raise Exception(f"Duplicate policy id {p['policy_id']} found")
        policy_ids.add(p['policy_id'])

        policy_infos.append(p)

    args.policies = policy_infos

    return args
