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
from king_of_the_hill.arg_parser import get_evaluation_args
from king_of_the_hill.evaluate import get_eval_path, run_eval_and_clean

# TODO make a proper commandline argparser
# Set  WandB entity and project
wandb_entity = 'ece-239-as'
wandb_project = 'social_marl_koth'

max_runs = 20
checkpoint = 1500
episodes = 20
policy_ids = ['policy0', 'policy1']
group_name = 'koth-multi-train'
batch_folder = f'{group_name}_cross_4x'

# What to filter runs by
run_filters = {'group': group_name, 'state': 'finished'}
eval_arguments = f"--episodes {episodes} --num-gpus 1 --activations lstm_hidden action_logits"

# Query the WandB API for runs with exactly 1500 training iterations
api = wandb.Api()

# See https://github.com/wandb/wandb/issues/5479
query_response = api.runs(f'{wandb_entity}/{wandb_project}', filters=run_filters)
dummy_var = query_response[
    0]  # This line is needed due to lazy evaluation. The runs won't be fetched until we try to access them.
runs: List[Run] = query_response.objects
print(f'Fetched run data from wandb. Found {len(runs)} matching runs.')

run_ids = [r.id for r in runs][:max_runs]
policy_run_ids = [[(r, p) for r in run_ids] for p in policy_ids]
rllib_ids = {r.id: r.name for r in runs[:max_runs]}
matchups = list(itertools.product(policy_run_ids[0], policy_run_ids[1]))  # Note this will yield max_runs^2 pairs.
print(f'Found {len(matchups)} matchups to evaluate...')

# Print run details
for idx, matchup in enumerate(matchups):
    first_policy = json.dumps({'policy_id': matchup[0][1], 'run_id': matchup[0][0], 'checkpoint': checkpoint})
    second_policy = json.dumps({'policy_id': matchup[1][1], 'run_id': matchup[1][0], 'checkpoint': checkpoint})
    all_eval_args = f"--policies '{first_policy}' '{second_policy}' {eval_arguments} --eval-folder {batch_folder}"
    run_config = vars(get_evaluation_args(all_eval_args))
    run_folder = get_eval_path(run_config)
    # FIXME turn this folder name generation into a function shared with evaluate.py
    policy_names = [f"{rllib_ids[m[0]]}.checkpoint-{checkpoint}.{m[1]}" for m in matchup]
    print(f"Evaluating policies: {','.join(policy_names)}")
    matchup_folder = '__VS__'.join(policy_names)

    metrics_files = glob.glob(os.path.join(run_folder, "**", f"{matchup_folder}/metrics.pkl"), recursive=True)
    print(f"For matchup {idx + 1}/{len(matchups)}: {matchup}")
    if len(metrics_files) == 0:
        print(f"No metrics file found. Will start evaluation...")
        run_eval_and_clean(run_config)
    elif len(metrics_files) == 1:
        with open(metrics_files[0], 'rb') as fp:
            metrics_data = pickle.load(fp)
        if len(metrics_data['rewards']) == run_config['episodes']:
            print('Full metrics file found. Will skip evaluation...')
        else:
            print(
                f"Partial metrics file found with {len(metrics_data['rewards'])}/{run_config['episodes']} episodes. Will redo evaluation...")
            shutil.rmtree(run_folder)
            run_eval_and_clean(run_config)
    else:
        raise Exception(
            f"Expected to find at most one metrics file for {run_config['rllib_run_name']}. Found {len(metrics_files)}")
