import argparse
import json
import shlex
from typing import Optional
from utils.helpers import check_json_has_fields

# Maps a particular layer name to the property name and index of the respective dimension in the activation
supported_activation_layers = {
    'lstm_hidden': ['lstm', 0],
    'lstm_cell': ['lstm', 1],
    'rnn': ['rnn', 0],
    'action_logits': ['_logits_branch', 0]
}

example_players_structure = json.dumps({
    'policy_id': 'policy0',
    'run_id': 'c6bd8_00000',
    'checkpoint': 1000,
    'role': 'default'
})

example_activation_structure = json.dumps({
    'layer_path': 'lstm',  # Object property and index path to the layer
    'activation_path': '1.0',  # Path to the activation. Can also be empty for the whole activation.
    'alias': 'lstm_hidden',  # An alias to save the activations under
})


def get_evaluation_args(arg_str: Optional[str] = None):
    parser = argparse.ArgumentParser(description='King of the Hill Agent Evaluation')
    parser.add_argument('--players', type=str, required=True, nargs='+',
                        help='A list of JSON strings detailing which policies to play against each other in what roles.'
                             f"An example structure is: {example_players_structure}")
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
                        help=f'set as list of JSONS in order to record activations of neurons. '
                             f'An example structure is: {example_activation_structure}')
    parser.add_argument('--rllib-path', required=False, type=str, default='/tmp/ray_results',
                        help='directory where to save rllib training states to')
    parser.add_argument('--num-gpus', type=int, required=False, default=0,
                        help='how many GPUS to use')
    parser.add_argument('--score-for-dormant-activations', action=argparse.BooleanOptionalAction, required=False,
                        default=False,
                        help='Yields normalised scores of activations of neurons')
    parser.add_argument('--eval-folder', type=str, required=False, default='',
                        help='Add a custom sub-folder to add to the evaluation path, '
                             'changing it to "evaluation/eval_games/<env_name>/<sub-folder>/<run-id>"')

    args = parser.parse_args(shlex.split(arg_str)) if arg_str else parser.parse_args()


    # Validate player JSONs
    player_infos = []
    for p in args.players:
        check_json_has_fields(p.replace("\'", "\""), example_players_structure)
        player_infos.append(json.loads(p.replace("\'", "\"")))
    args.players = player_infos

    # Validate activation JSONs
    activation_infos = []
    if args.activations:
        for a in args.activations:
            check_json_has_fields(a.replace("\'", "\""), example_activation_structure)
            activation_infos.append(json.loads(a.replace("\'", "\"")))

        aliases = [a['alias'] for a in activation_infos]
        assert len(aliases) == len(set(aliases)), f"Duplicate activation alias found in {aliases}"
        args.activations = activation_infos
    return args
