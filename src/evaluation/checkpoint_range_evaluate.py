# This is a custom script designed specifically to evaluate a series of checkpoints from agents
import itertools
import json
import pickle
import shutil
from typing import List
import wandb
import os
import glob
from wandb.apis.public import Run

from evaluation.arg_parser import get_evaluation_args
from evaluation.evaluate import get_eval_path, run_eval_and_clean
from models.model_catalog import register_models

# TODO make a proper commandline argparser
# Set  WandB entity and project
wandb_entity = 'ece-239-as'
wandb_project = 'social_marl_koth'

player_configs = [{'policy_id': 'policy_0_p5', 'run_id': 'run-e0d8f801f', 'role': 'default'},
                  {'policy_id': 'policy_1_p3', 'run_id': 'run-e0d8f801f', 'role': 'default'}]
run_ids = [p['run_id'] for p in player_configs]
checkpoints = list(range(100, 800, 100))
episodes = 50
activation_config = json.dumps({'layer_path': 'lstm', 'activation_path': '1.0', 'alias': 'lstm_hidden'})
batch_folder = f'2024-01-11-coins_best'

# What to filter runs by
run_filters = {'name': {'$in': run_ids}}
eval_arguments = f"--episodes {episodes} --num-gpus 1 --activations '{activation_config}'"

# Query the WandB API for runs with exactly 1500 training iterations
api = wandb.Api()

# See https://github.com/wandb/wandb/issues/5479
query_response = api.runs(f'{wandb_entity}/{wandb_project}', filters=run_filters)
dummy_var = query_response[
    0]  # This line is needed due to lazy evaluation. The runs won't be fetched until we try to access them.
runs: List[Run] = query_response.objects
print(f'Fetched run data from wandb. Found {len(runs)} matching runs.')
run_rllib_id_map = {r.name: r.id for r in runs}
print(f'Found {len(checkpoints)} matchups to evaluate...')

# Print run details
for idx, c in enumerate(checkpoints):
    first_player = player_configs[0] | {'checkpoint': c}
    second_player = player_configs[1] | {'checkpoint': c}

    all_eval_args = (f"--players '{json.dumps(first_player)}' '{json.dumps(second_player)}' "
                     f"{eval_arguments} --eval-folder {batch_folder}")
    run_config = vars(get_evaluation_args(all_eval_args))
    matching_run = [r for r in runs if r.id == first_player['run_id']][0]
    env_name = json.loads(matching_run.json_config)['run_config']['value']['env']
    run_folder = get_eval_path(run_config, env_name)    # FIXME turn this folder name generation into a function shared with evaluate.py
    policy_names = [f"{p['run_id']}.checkpoint-{c}.{p['policy_id']}" for p in player_configs]
    print(f"Evaluating policies: {','.join(policy_names)}")
    matchup_folder = '__VS__'.join(policy_names)

    metrics_files = glob.glob(os.path.join(run_folder, "**", f"{matchup_folder}/metrics.pkl"), recursive=True)
    print(f"For matchup {idx + 1}/{len(checkpoints)}: {first_player} vs. {second_player}")
    if len(metrics_files) == 0:
        print(f"No metrics file found. Will start evaluation...")
        # This is needed because rllib seems to "forget" about the model catalog if it is re-iniailised
        # for every evaluation
        register_models()
        run_eval_and_clean(run_config)
    elif len(metrics_files) == 1:
        print(f'Found file at {metrics_files[0]}')
        with open(metrics_files[0], 'rb') as fp:
            metrics_data = pickle.load(fp)
        if len(metrics_data['general_metrics']['rewards']) == run_config['episodes']:
            print('Full metrics file found. Will skip evaluation...')
        else:
            print(
                f"Partial metrics file found with {len(metrics_data['general_metrics']['rewards'])}/{run_config['episodes']} episodes. Will redo evaluation...")
            shutil.rmtree(run_folder)
            run_eval_and_clean(run_config)
    else:
        raise Exception(
            f"Expected to find at most one metrics file for {run_config['rllib_run_name']}. Found {len(metrics_files)}")
