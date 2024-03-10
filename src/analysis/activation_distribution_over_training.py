"""
This module calculates and displays the distribution of neuron activations over checkpoints for a given set of
pair-wise policy evaluation games.
"""
import glob
import os
import pickle
from collections import defaultdict
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from analysis.helpers import save_as_example_output, EXAMPLE_OUTPUTS_FOLDER
from utils.wandb_api import make_dir_if_not_exists

colours = ['red', 'blue']


def get_activations(all_metrics_paths: List[str],
                    checkpoint_iters: List[int],
                    activation_layer: str) -> Tuple[dict, str]:
    """
    Gets the activations for a given layer for a given set of checkpoints for a given set of pair-wise policy
    @param all_metrics_paths: paths to metrics files for each checkpoint
    @param checkpoint_iters: checkpoints to get activations for
    @param activation_layer: name of the layer to get activations for
    @return: dict of {policy_id: {checkpoint: activations (as numpy array)}}
    """
    with open(all_metrics_paths[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    policy_ids = [p['policy_id'] for p in example_pkl_file['general_metrics']['other']['agent_info']]
    player_policy_map = {a['player']: a['policy_id'] for a in
                         example_pkl_file['general_metrics']['other']['agent_info']}
    run_id = example_pkl_file['general_metrics']['other']['agent_info'][0]['run_id']

    activations = {p: {c: [] for c in checkpoint_iters} for p in policy_ids}

    assert len(policy_ids) == 2, 'This method is only designed for 2 agents'

    for c in checkpoint_iters:
        path_pattern = f'{run_id}.checkpoint-{c}.{policy_ids[0]}__VS__{run_id}.checkpoint-{c}.{policy_ids[1]}'
        matching_paths = [p for p in all_metrics_paths if path_pattern in p]
        assert len(matching_paths) == 1, f'Found {len(matching_paths)} matching paths for pattern {path_pattern}'

        with open(matching_paths[0], 'rb') as fp:
            metrics_file = pickle.load(fp)

        for ep_activations in metrics_file['general_metrics']['activations']:
            for player in ep_activations:
                activations[player_policy_map[player]][c].append(ep_activations[player][activation_layer])

        for p in policy_ids:
            activations[p][c] = np.vstack(activations[p][c]).flatten()

    return activations, run_id


def plot_activations(activations: dict, run_dir: str, run_id: str) -> str:
    plot_dir = f'{run_dir}/plots/activation_distribution'

    for idx, (policy_id, checkpoint_activations) in enumerate(activations.items()):
        min_act = min([min(c) for c in checkpoint_activations.values()])
        max_act = max([max(c) for c in checkpoint_activations.values()])
        checkpoint_nums = list(checkpoint_activations)

        # we use the percentile here to remove extreme outliers for easier viewing
        bin_size = 0.05
        bins = np.arange(min_act, max_act + bin_size, bin_size)

        # Determine global y-limit (maximum frequency among all histograms)
        max_freq = max([np.histogram(pop, bins=bins)[0].max() for pop in checkpoint_activations.values()])

        fig, axs = plt.subplots(len(checkpoint_nums), 1, sharex=True, figsize=(12, 2 * len(checkpoint_nums)))

        for ax, x, population in zip(axs, checkpoint_nums, checkpoint_activations.values()):
            ax.hist(population, bins=bins, alpha=0.6, align='mid', color=colours[idx])
            ax.set_ylim(0, max_freq)
            ax.set_xlim(min_act, max_act)

            # Place the title to the right of the plot
            ax.text(1.02, 0.5, f'Neuron Activations for Checkpoint={x}', transform=ax.transAxes,
                    verticalalignment='center')
            ax.legend()
        # Set common labels
        fig.text(0.5, 0.01, 'Neuron Activations', ha='center', va='center')
        fig.text(0.01, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
        plt.suptitle(f'Neuron Activations over Training - {run_id} - {policy_id}')

        plt.tight_layout()
        make_dir_if_not_exists(plot_dir)
        plt.savefig(f'{plot_dir}/{run_id}_{policy_id}.png')
        plt.show()

    return plot_dir


if __name__ == "__main__":
    run_paths = 'eval_games/coins/2024-01-11-coins_best'
    checkpoint_frequency = 200
    checkpoint_start = 100
    checkpoint_end = 700
    activation_layer = 'lstm_hidden'

    all_metrics_paths = glob.glob(os.path.join(run_paths, "**", f"metrics.pkl"), recursive=True)
    checkpoint_iters = list(range(checkpoint_start, checkpoint_end + checkpoint_frequency, checkpoint_frequency))
    activations, run_id = get_activations(all_metrics_paths=all_metrics_paths,
                                          checkpoint_iters=checkpoint_iters,
                                          activation_layer=activation_layer)
    plot_path = plot_activations(activations, run_dir=run_paths, run_id=run_id)
    save_as_example_output(plot_path, EXAMPLE_OUTPUTS_FOLDER)
