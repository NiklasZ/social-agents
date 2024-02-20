# This is a custom script designed specifically to evaluate a batch of agents from runs based on some criterion.
# Useful if it is too tedious to run evaluate.py on each run individually.
import itertools
import json
import pickle
import shutil
import time
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

run_ids = ['run-e0d8f7dda']
checkpoint = 700
episodes = 5
policy_ids = [list(f'policy_{p}_p{i}' for i in range(0, 8)) for p in range(0, 2)]
player_roles = ['default', 'default']
batch_folder = f'terro_test_700'

# What to filter runs by
run_filters = {'name': {'$in': run_ids}}
eval_arguments = f"--episodes {episodes} --num-gpus 1"

# Query the WandB API for runs with exactly 1500 training iterations
api = wandb.Api()

# See https://github.com/wandb/wandb/issues/5479
query_response = api.runs(f'{wandb_entity}/{wandb_project}', filters=run_filters)
dummy_var = query_response[
    0]  # This line is needed due to lazy evaluation. The runs won't be fetched until we try to access them.
runs: List[Run] = query_response.objects
print(f'Fetched run data from wandb. Found {len(runs)} matching runs.')

policy_run_ids = [[(r, p) for r in run_ids] for p in policy_ids]
matchups = list(itertools.product(*policy_ids, run_ids))  # Note this will yield max_runs^2 pairs.
print(f'Found {len(matchups)} matchups to evaluate...')

# Print run details
for idx, matchup in enumerate(matchups):
    first_policy = json.dumps(
        {'policy_id': matchup[0], 'run_id': matchup[2], 'checkpoint': checkpoint, 'role': player_roles[0]})
    second_policy = json.dumps(
        {'policy_id': matchup[1], 'run_id': matchup[2], 'checkpoint': checkpoint, 'role': player_roles[0]})
    all_eval_args = f"--players '{first_policy}' '{second_policy}' {eval_arguments} --eval-folder {batch_folder}"
    run_config = vars(get_evaluation_args(all_eval_args))
    matching_run = [r for r in runs if r.id == matchup[2]][0]
    env_name = json.loads(matching_run.json_config)['run_config']['value']['env']
    run_folder = get_eval_path(run_config, env_name)
    # FIXME turn this folder name generation into a function shared with evaluate.py
    policy_names = [f"{matchup[2]}.checkpoint-{checkpoint}.{m}" for m in matchup[:2]]
    print(f"Evaluating policies: {','.join(policy_names)}")
    matchup_folder = '__VS__'.join(policy_names)

    metrics_files = glob.glob(os.path.join(run_folder, "**", f"{matchup_folder}/metrics.pkl"), recursive=True)
    print(f"For matchup {idx + 1}/{len(matchups)}: {matchup}")
    if len(metrics_files) == 0:
        print(f"No metrics file found. Will start evaluation...")
        # This is needed because rllib seems to "forget" about the model catalog if it is re-iniailised
        # for every evaluation
        register_models()
        run_eval_and_clean(run_config)
    elif len(metrics_files) == 1:
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
