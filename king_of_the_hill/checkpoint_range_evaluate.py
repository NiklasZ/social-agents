# This is a custom script designed specifically to evaluate a series of checkpoints from agents
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
# TODO refactor evaluation scripts into own folder
# Set  WandB entity and project
wandb_entity = 'ece-239-as'
wandb_project = 'social_marl_koth'

checkpoints = list(range(100, 1600, 200))
episodes = 10
policy_ids = ['policy0', 'policy1']
group_name = 'koth-multi-train'
batch_folder = f'{group_name}_dormant_neurons'

# What to filter runs by
run_filters = {'group': group_name, 'state': 'finished'}
eval_arguments = f"--episodes {episodes} --num-gpus 1 --activations lstm_hidden action_logits --score-for-dormant-activations"
rllib_ids = ['run-1683629797'] #['run-1683514150', 'run-1683419885', 'run-1683401773', 'run-1683629797']

# Query the WandB API for runs with exactly 1500 training iterations
api = wandb.Api()

# See https://github.com/wandb/wandb/issues/5479
query_response = api.runs(f'{wandb_entity}/{wandb_project}', filters=run_filters)
dummy_var = query_response[
    0]  # This line is needed due to lazy evaluation. The runs won't be fetched until we try to access them.
runs: List[Run] = query_response.objects
print(f'Fetched run data from wandb. Found {len(runs)} matching runs.')
run_rllib_id_map = {r.name: r.id for r in runs}
matchups = list(itertools.product(rllib_ids, checkpoints))  # Note this will yield max_runs^2 pairs.
print(f'Found {len(matchups)} matchups to evaluate...')

# Print run details
for idx, matchup in enumerate(matchups):
    first_policy = json.dumps({'policy_id': policy_ids[0], 'run_id': run_rllib_id_map[matchup[0]], 'checkpoint': matchup[1]})
    second_policy = json.dumps({'policy_id': policy_ids[1], 'run_id': run_rllib_id_map[matchup[0]], 'checkpoint': matchup[1]})
    all_eval_args = f"--policies '{first_policy}' '{second_policy}' {eval_arguments} --eval-folder {batch_folder}"
    run_config = vars(get_evaluation_args(all_eval_args))
    run_folder = get_eval_path(run_config)
    # FIXME turn this folder name generation into a function shared with evaluate.py
    policy_names = [f"{matchup[0]}.checkpoint-{matchup[1]}.{p}" for p in policy_ids]
    print(f"Evaluating policies: {','.join(policy_names)}")
    matchup_folder = '__VS__'.join(policy_names)

    metrics_files = glob.glob(os.path.join(run_folder, "**", f"{matchup_folder}/metrics.pkl"), recursive=True)
    print(f"For matchup {idx + 1}/{len(matchups)}: {matchup}")
    if len(metrics_files) == 0:
        print(f"No metrics file found. Will start evaluation...")
        run_eval_and_clean(run_config)
    elif len(metrics_files) == 1:
        print(f'Found file at {metrics_files[0]}')
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
