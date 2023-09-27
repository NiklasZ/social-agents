import glob
import os
import pickle
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from king_of_the_hill.analysis.plotting.cross_play_returns import get_cross_play_returns
from king_of_the_hill.wandb_api import make_dir_if_not_exists

"""
This plots the distribution of actions when an agent of some run X plays against an agent from some run Y. We do this
for the top k agents of each playable colour and plot the action distribution for each matchup.  
"""


player_colours = {'player_0': 'red', 'player_1': 'blue'}

def get_cross_play_action_distribution(metrics_paths: List[str],
                                       player_returns: dict,
                                       top_k: int,
                                       chosen_player: str):
    with open(metrics_paths[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    player_ids = sorted([agent_info['player'] for agent_info in example_pkl_file['other']['agent_info']])

    if len(player_ids) != 2:
        raise NotImplementedError(
            f'Script only designed for 2 players per episode, not {len(player_ids)}. Please implement')

    other_player = [p for p in player_ids if p != chosen_player][0]

    top_k_by_player = {}
    for player in player_ids:
        mean_returns = [(policy, np.mean(list(opponents.values()))) for policy, opponents in
                        player_returns[player].items()]
        top_k_by_player[player] = list(reversed(sorted(mean_returns, key=lambda x: x[1])))[:top_k]

    # {(first_policy, second_policy):[action]}
    first_policy_actions_taken = {}

    for first_policy, _ in tqdm(top_k_by_player[chosen_player]):
        for second_policy, _ in top_k_by_player[other_player]:
            matching_files = [path for path in all_metrics_paths if first_policy in path and second_policy in path]
            if len(matching_files) != 1:
                raise Exception(
                    f'Expected to encounter exactly one metrics file matching {first_policy},{second_policy}. '
                    f'Instead got: {matching_files}')

            with open(matching_files[0], 'rb') as fp:
                metrics_file = pickle.load(fp)

            first_policy_actions = [step[chosen_player] for ep in metrics_file['actions'] for step in ep]
            first_policy_actions_taken[(first_policy, second_policy)] = first_policy_actions

    return first_policy_actions_taken


def plot_cross_play_action_distributions(run_path: str, player_id: str, top_k: int, player_actions: dict):
    # Suppose you have your one-hot encoded data as a numpy array
    # data = np.random.randint(0, 9, (25, 100))  # Generate random data, replace this with your actual data
    run_labels = list(player_actions.keys())
    actions = list(player_actions.values())
    fig, axs = plt.subplots(top_k, top_k, figsize=(15, 15))  # Create a grid of 5x5 subplots

    for i in range(top_k):
        for j in range(top_k):
            ax = axs[i, j]
            data = actions[top_k * i + j]
            weights = np.ones_like(data) / float(len(data))  # Create weights for each histogram
            ax.hist(data, bins=range(10), alpha=0.7, color=player_colours[player_id],
                    edgecolor='black', weights=weights)  # Plot histogram with 9 bins (0-9)
            # ax.set_title(f'Histogram {5 * i + j + 1}')  # Set the title for each subplot
            ax.set_ylim([0, 1])  # Set the y-axis limits

            # Set x and y labels for each row and column
            if i == top_k - 1:
                run_id = run_labels[top_k * i + j][1].split('.')[0]
                ax.set_xlabel(run_id)
            if j == 0:
                run_id = run_labels[top_k * i + j][0].split('.')[0]
                ax.set_ylabel(run_id)

    fig.suptitle(f'Action Distribution of {player_id} for top {top_k} agents', fontsize=16)

    # Add space between plots
    plt.tight_layout()

    chart_dir = f'{run_path}/plots/cross_play_action_distribution'
    make_dir_if_not_exists(chart_dir)
    chart_path = f"{chart_dir}/{player_id}.png"
    plt.savefig(chart_path)
    plt.show()

    plt.show()


if __name__ == "__main__":
    cross_play_path = 'king_of_the_hill/evaluation/koth-multi-train_cross_4x'
    top_k = 5
    all_metrics_paths = glob.glob(os.path.join(cross_play_path, "**", f"metrics.pkl"), recursive=True)
    player_returns_mean, player_returns_std, player_policies = get_cross_play_returns(all_metrics_paths)
    for p in ['player_0', 'player_1']:
        player_actions = get_cross_play_action_distribution(
            metrics_paths=all_metrics_paths,
            player_returns=player_returns_mean,
            top_k=top_k,
            chosen_player=p)
        plot_cross_play_action_distributions(run_path=cross_play_path,
                                             player_id=p,
                                             player_actions=player_actions,
                                             top_k=top_k)

