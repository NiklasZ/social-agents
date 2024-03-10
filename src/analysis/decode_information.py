"""
Decodes various information from the activations of an agent
"""
# TODO code needs to be updated to use new evaluation metrics structure

import glob
import os
import pickle
from typing import List, Any
import numpy as np
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from analysis.filters import filter_by_deaths, filter_by_mutual_vision
from analysis.helpers import get_relative_player_positions
from utils.wandb_api import make_dir_if_not_exists

# Base args to configure
evaluation_folder = 'koth-multi-train'
seed = 42
test_ratio = 0.2
filter_out_deaths = False
filter_for_mutual_vision = True
drop_MSEs_gte = 100000
Model_Type = LinearRegression
model_args = {}# {'max_iter': 10000}
# Enable to clear function caching
# delete_disk_caches_for_function('decode_position_via_linear_regression')

# Path-related stuff
working_dir = os.getcwd()
base_folder = 'social-agents'
working_dir = working_dir[:working_dir.find(base_folder) + len(base_folder)]  # Ensure we use social-agents as base dir
run_batch_dir = f'{working_dir}/king_of_the_hill/evaluation/{evaluation_folder}'
cache_dir = f'{run_batch_dir}/cache'
run_paths = sorted([f'{run_batch_dir}/{name}' for name in os.listdir(run_batch_dir) if 'run' in name])


@cache_to_disk(31)  # Cache for 31 days
def decode_position_via_linear_regression(run_paths: List[str],
                                          seed: int,
                                          test_ratio: 0.1,
                                          Model_Type: Any,
                                          model_args: dict):
    np.random.seed(seed)
    example_run_path = run_paths[0]
    metrics_file_path = glob.glob(os.path.join(example_run_path, "**", "metrics.pkl"), recursive=True)
    with open(metrics_file_path[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    episode_count = len(example_pkl_file['activations'])
    indices = np.arange(episode_count)
    np.random.shuffle(indices)
    test_size = int(episode_count * test_ratio)
    train_indices, test_indices = indices[:episode_count - test_size], indices[-test_size:]
    player_ids = ['player_0', 'player_1']
    test_errors = []
    mean_returns = []

    if len(player_ids) != 2:
        raise Exception(f'Script only designed for 2 players, not {len(player_ids)}. Please implement')

    for run_idx, run_path in enumerate(tqdm(run_paths)):
        metrics_file_path = glob.glob(os.path.join(run_path, "**", "metrics.pkl"), recursive=True)
        with open(metrics_file_path[0], 'rb') as fp:
            base_metrics = pickle.load(fp)
            metrics = base_metrics

        # Check for filtering out timesteps with dead agents
        if filter_out_deaths:
            metrics = filter_by_deaths(base_metrics, player_ids)

        # Check for filtering only for timesteps where agents see each other
        if filter_for_mutual_vision:
            metrics = filter_by_mutual_vision(base_metrics, player_ids)

        for p_idx, p_id in enumerate(player_ids):
            other_p_id = [p for p in player_ids if p != p_id][0]
            relative_pos_func = lambda indices: np.vstack(
                [get_relative_player_positions(metrics, ep_id, p_id, other_p_id) for ep_id in indices])
            activations_func = lambda indices: np.vstack(
                [np.array(metrics['activations'][ep_id][p_id]['lstm_hidden']) for ep_id in indices])
            train_relative_positions = relative_pos_func(train_indices)
            train_activations = activations_func(train_indices)
            # activations = np.vstack(
            #     [np.array(metrics['activations'][ep_id][p_id]['lstm_hidden']) for ep_id in train_indices])
            # Create a model
            base_model = Model_Type(**model_args)
            # Create the multi-output regressor
            model = MultiOutputRegressor(base_model)
            model.fit(train_activations, train_relative_positions)
            test_relative_positions = relative_pos_func(test_indices)
            test_activations = activations_func(test_indices)
            pred_positions = model.predict(test_activations)
            mse = mean_squared_error(test_relative_positions, pred_positions)
            test_errors.append(mse)
            player_mean_return = sum([t[p_id] for i in train_indices for t in metrics['rewards'][i]]) / (
                    episode_count * test_ratio)
            mean_returns.append(player_mean_return)

    return np.array(mean_returns), np.array(test_errors)


mean_returns, test_errors = decode_position_via_linear_regression(run_paths, seed, test_ratio, Model_Type, model_args)

matching_indices = test_errors < drop_MSEs_gte
mean_returns = mean_returns[matching_indices]
test_errors = test_errors[matching_indices]
print(mean_returns)
print(test_errors)

fig = plt.figure(figsize=(10, 6))
plt.scatter(test_errors, mean_returns, c='orange')

model_name = Model_Type.__name__
xlabel = 'Prediction MSE'
title = f'Decoding of Position from Activations via {model_name}'
fig_name = f'decode_position_{model_name}'

if filter_out_deaths:
    title += ' + Only Alive Filter'
    fig_name += '-death_filtered'

if filter_for_mutual_vision:
    title += ' + Mutual Vision Filter'
    fig_name += '-mutual_vision_filtered'

plot_dir = f'{run_batch_dir}/plots'
make_dir_if_not_exists(plot_dir)
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel('Mean Return')
z = np.polyfit(test_errors, mean_returns, 1)
p = np.poly1d(z)
plt.plot(test_errors, p(test_errors), "r--")
plt.savefig(f'{plot_dir}/{fig_name}.png')
plt.show()
