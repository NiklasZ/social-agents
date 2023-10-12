import glob
import os
import pickle
from typing import List, Dict
import re
import numpy as np
from matplotlib import pyplot as plt

from king_of_the_hill.wandb_api import make_dir_if_not_exists

"""
This is file plots neurons scores over different training checkpoints to identify dormant neurons. It generates
a histogram of neuron score distribution for each checkpoint. For more, see:
'The Dormant Neuron Phenomenon in Deep Reinforcement Learning' by Sokar et al. (2023)
paper: https://arxiv.org/pdf/2302.12902.pdf, code: https://github.com/google/dopamine/tree/master/dopamine/labs/redo
"""

player_colours = {'player_0': 'red', 'player_1': 'blue'}


def extract_checkpoint_integer(s):
    match = re.search(r'checkpoint-(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        raise Exception(f'Could not extract checkpoint integer from {s}')


def get_dormant_neurons(all_metrics_paths: List[str]) -> Dict[str, Dict[str, Dict[int, dict]]]:
    with open(all_metrics_paths[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    player_ids = sorted([agent_info['player'] for agent_info in example_pkl_file['other']['agent_info']])
    dormant_neurons = {}  # dict of {run_id: {player_id: {checkpoint_num: neuron_scores}}}
    for m_path in all_metrics_paths:
        with open(m_path, 'rb') as fp:
            metrics_file = pickle.load(fp)

        run_names = {agent_info['rllib_run_name'] for agent_info in metrics_file['other']['agent_info']}

        if len(run_names) != 1:
            raise Exception(f'Expected to find exactly one run name in {m_path}. Instead found {run_names}')

        run_id = run_names.pop()
        if run_id not in dormant_neurons:
            dormant_neurons[run_id] = {p: {} for p in player_ids}

        checkpoint_num = extract_checkpoint_integer(metrics_file['other']['agent_info'][0]['checkpoint_name'])
        for p in player_ids:
            dormant_neurons[run_id][p][checkpoint_num] = metrics_file['relative_activation_scores'][p]

    return dormant_neurons


def plot_cumulative_dormant_neurons(dormant_neurons: Dict[str, Dict[str, Dict[int, dict]]], run_dir: str,
                                    dormant_threshold=0.1):
    for run_id, player_data in dormant_neurons.items():
        for player_id, checkpoint_data in player_data.items():
            aggregated_scores = {}  # all mean neuron scores for each checkpoint (mean taken across samples)
            for checkpoint_num, layer_scores in checkpoint_data.items():
                aggregated_neurons = []
                for layer in layer_scores.values():
                    for output in layer.values():
                        aggregated_neurons.append(np.stack(output).mean(axis=0))

                aggregated_scores[checkpoint_num] = np.concatenate(aggregated_neurons)

            sorted_scores = sorted(aggregated_scores.items())
            checkpoint_nums = [t[0] for t in sorted_scores]
            neuron_scores = [t[1] for t in sorted_scores]
            all_scores = np.concatenate(neuron_scores)
            global_min, global_99th_pct = all_scores.min(), np.percentile(all_scores,
                                                                          99)  # we use the percentile here to remove extreme outliers for easier viewing
            bin_size = 0.05
            bins = np.arange(global_min, global_99th_pct + bin_size, bin_size)

            # Determine global y-limit (maximum frequency among all histograms)
            max_freq = max([np.histogram(pop, bins=bins)[0].max() for pop in neuron_scores])

            fig, axs = plt.subplots(len(checkpoint_nums), 1, sharex=True, figsize=(12, 2 * len(checkpoint_nums)))

            for ax, x, population in zip(axs, checkpoint_nums, neuron_scores):
                ax.hist(population, bins=bins, alpha=0.6, align='mid', color=player_colours[player_id])
                ax.set_ylim(0, max_freq)
                ax.set_xlim(global_min, global_99th_pct)

                # Add vertical line for the cutoff
                ax.axvline(dormant_threshold, color='black', linestyle='--',
                           label=f'dormancy threshold = {dormant_threshold}')

                # Place the title to the right of the plot
                ax.text(1.02, 0.5, f'Neuron Score for Checkpoint={x}', transform=ax.transAxes,
                        verticalalignment='center')
                ax.legend()
            # Set common labels
            fig.text(0.5, 0.01, 'Neuron Scores (Normalised by layer)', ha='center', va='center')
            fig.text(0.01, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
            plt.suptitle(f'Neuron Scores over Training - {run_id} - {player_id}')

            plt.tight_layout()
            plot_dir = f'{run_dir}/plots/dormant_neurons'
            make_dir_if_not_exists(plot_dir)
            plt.savefig(f'{plot_dir}/{run_id}_{player_id}.png')
            plt.show()


if __name__ == "__main__":
    run_paths = 'king_of_the_hill/evaluation/koth-multi-train_dormant_neurons'
    all_metrics_paths = glob.glob(os.path.join(run_paths, "**", f"metrics.pkl"), recursive=True)
    dormant_neurons = get_dormant_neurons(all_metrics_paths)
    plot_cumulative_dormant_neurons(dormant_neurons, run_dir=run_paths)
