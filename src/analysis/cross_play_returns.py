"""
Yields plot of a matrix of reward for each matchup between pairs of policies.
"""


import glob
import os
import pickle
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors

from analysis.helpers import save_as_example_output, EXAMPLE_OUTPUTS_FOLDER
from utils.wandb_api import make_dir_if_not_exists

colours = ['blue', 'red']


def get_cross_play_returns(all_metrics_paths: List[str]) -> [dict, dict, dict]:
    with open(all_metrics_paths[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    player_ids = sorted(
        [agent_info['player'] for agent_info in example_pkl_file['general_metrics']['other']['agent_info']])

    if len(player_ids) != 2:
        raise NotImplementedError(
            f'Script only designed for 2 players per episode, not {len(player_ids)}. Please implement')

    player_returns_mean = {p: {} for p in player_ids}  # {player_id: {policy_id: {opponent_policy_id: mean return}}}
    player_returns_std = {p: {} for p in player_ids}  # {player_id: {policy_id: {opponent_policy_id: mean return}}}

    player_policies = {p: set() for p in player_ids}  # {player_id: set(player_id policies)}

    for metrics_path in tqdm(all_metrics_paths):
        with open(metrics_path, 'rb') as fp:
            all_metrics = pickle.load(fp)
            metrics = all_metrics['general_metrics']

        agent_infos = metrics['other']['agent_info']

        # Generates mapping from player to the unique policy (defined by run, checkpoint and policy id used by rllib).
        policy_mapping = {a_i['player']: f"{a_i['rllib_run_name']}.{a_i['checkpoint_name']}.{a_i['policy_id']}" for a_i
                          in agent_infos}

        for a_i in agent_infos:
            player = a_i['player']
            player_policies[player].add(policy_mapping[player])
            mean_return = np.mean([np.sum([r[player] for r in ep]) for ep in metrics['rewards']])
            std_return = np.std([np.sum([r[player] for r in ep]) for ep in metrics['rewards']])
            other_player = [p for p in player_ids if p != player][0]
            policy_id, other_policy_id = policy_mapping[player], policy_mapping[other_player]

            if policy_id not in player_returns_mean[player]:
                player_returns_mean[player][policy_id] = {}
                player_returns_std[player][policy_id] = {}

            player_returns_mean[player][policy_id][other_policy_id] = mean_return
            player_returns_std[player][policy_id][other_policy_id] = std_return

    return player_returns_mean, player_returns_std, player_policies


def plot_matrices(matrices: Dict[str, np.ndarray],
                  player_ids: List[str],
                  matrix_idx_map: dict,
                  cross_play_path: str,
                  matrix_type: str) -> str:
    plot_dir = f'{cross_play_path}/plots/cross_play_returns'

    for player in player_ids:
        other_player = [p for p in player_ids if p != player][0]

        # Define the labels for the rows and columns
        # This is a bit more complex as we put common parts of the labels in the title
        # And different parts in the rows and columns
        raw_labels = list(matrix_idx_map[player].keys()) + list(matrix_idx_map[other_player].keys())
        split_labels = [l.split('.') for l in raw_labels]
        has_common_run = len(set(l[0] for l in split_labels)) == 1
        has_common_checkpoint = len(set(l[1] for l in split_labels)) == 1
        common_policies = set(l[2] for l in split_labels)
        has_common_policies = len(common_policies) == 2

        title_details = ((f'{split_labels[0][0]}.' if has_common_run else '') +
                         (f'{split_labels[0][1]}.' if has_common_checkpoint else '') +
                         (','.join(common_policies) if has_common_policies else '')).strip('.')

        title = f'{player} vs. {other_player} for {title_details} - episode rewards {matrix_type}'
        pick_props = lambda props: ('.'.join(['' if h else prop for h, prop in
                                              zip([has_common_run, has_common_checkpoint, has_common_policies], props)])
                                    .strip('.'))
        row_labels = [pick_props(s.split('.')) for s in matrix_idx_map[player].keys()]
        col_labels = [pick_props(s.split('.')) for s in matrix_idx_map[other_player].keys()]

        fig, ax = plt.subplots(figsize=(13, 12))

        # Create the heatmap image
        # First colour represents low values, Second high values
        # We match the colours so low values match the other player's colour as they are performing well
        # And high values match the current player's colour as they are performing well
        if matrix_type == 'mean':
            text_colour = 'white'
            cmap = mcolors.LinearSegmentedColormap.from_list("", colours)

            # Choose colour gradient based on whether there are negative rewards or only positive ones
            if np.min(matrices[player]) < 0:
                # Set min-max of colour gradient to match matrix max (centers the middle of the gradient on 0).
                max_magnitude = np.max(np.abs(matrices[player]))
                cax = ax.imshow(matrices[player], cmap=cmap, vmin=-max_magnitude, vmax=max_magnitude)
            else:
                cax = ax.imshow(matrices[player], cmap=cmap)
        elif matrix_type == 'std':
            colors = ['white', 'orange']
            text_colour = 'black'
            cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
            # Set min-max of colour gradient to match matrix max (centers the middle of the gradient on 0).
            cax = ax.imshow(matrices[player], cmap=cmap)
        else:
            raise Exception(f'Unknown mode: {matrix_type}')

        # Add colorbar
        cbar = fig.colorbar(cax)

        # Add the actual numbers on the heatmap
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(j, i, int(matrices[player][i, j]), ha="center", va="center", color=text_colour)

        # Set the labels for the rows and columns
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        plt.xticks(rotation=90)

        # Title
        plt.title(title)
        plt.ylabel(f'{player} policies')
        plt.xlabel(f'{other_player} policies')

        # Save the plot
        make_dir_if_not_exists(plot_dir)
        plt.savefig(f'{plot_dir}/{title}.png')
        # Show the plot
        plt.show()

    return plot_dir


def plot_cross_play_returns(cross_play_path: str,
                            player_returns_mean: dict,
                            player_returns_std: dict,
                            player_policies: dict):
    player_ids = list(player_policies.keys())

    if len(player_ids) != 2:
        raise NotImplementedError(
            f'Script only designed for 2 players per episode, not {len(player_ids)}. Please implement')

    # Determines which rows and columns to map policy ids to.
    matrix_player_0_idx_map = {policy_id: idx for idx, policy_id in
                               enumerate(sorted(list(player_policies[player_ids[0]])))}
    matrix_player_1_idx_map = {policy_id: idx for idx, policy_id in
                               enumerate(sorted(list(player_policies[player_ids[1]])))}
    matrix_idx_map = {p: m for p, m in zip(player_ids, [matrix_player_0_idx_map, matrix_player_1_idx_map])}

    # Assign each return to a matrix entry
    player_0_matrix_dims = [len(matrix_idx_map[p]) for p in matrix_idx_map]
    player_1_matrix_dims = list(reversed(player_0_matrix_dims))
    mean_matrices = {player_ids[0]: np.zeros(player_0_matrix_dims),
                     player_ids[1]: np.zeros(player_1_matrix_dims)}
    std_matrices = {player_ids[0]: np.zeros(player_0_matrix_dims),
                    player_ids[1]: np.zeros(player_1_matrix_dims)}

    for player in player_ids:
        other_player = [p for p in player_ids if p != player][0]
        for player_policy in player_returns_mean[player]:
            row_idx = matrix_idx_map[player][player_policy]
            for opponent_policy in player_returns_mean[player][player_policy]:
                col_idx = matrix_idx_map[other_player][opponent_policy]
                mean_matrices[player][row_idx, col_idx] = player_returns_mean[player][player_policy][opponent_policy]
                std_matrices[player][row_idx, col_idx] = player_returns_std[player][player_policy][opponent_policy]

    plot_path = plot_matrices(matrices=mean_matrices,
                              cross_play_path=cross_play_path,
                              player_ids=player_ids,
                              matrix_idx_map=matrix_idx_map,
                              matrix_type='mean')

    plot_path = plot_matrices(matrices=std_matrices,
                              cross_play_path=cross_play_path,
                              player_ids=player_ids,
                              matrix_idx_map=matrix_idx_map,
                              matrix_type='std')
    return plot_path


# Print statistics on which policies perform best at minimum and an on average
def print_overall_performances(player_returns: dict, player_policies: dict, top_k: int):
    player_ids = list(player_policies.keys())
    for player in player_ids:
        mean_returns = [(policy, np.mean(list(opponents.values()))) for policy, opponents in
                        player_returns[player].items()]
        min_returns = [(policy, np.min(list(opponents.values()))) for policy, opponents in
                       player_returns[player].items()]
        top_k_mean = list(reversed(sorted(mean_returns, key=lambda x: x[1])))[:top_k]
        top_k_min = list(reversed(sorted(min_returns, key=lambda x: x[1])))[:top_k]

        print(f'\nFor {player} top {top_k} by mean return:')
        for e in top_k_mean:
            print(e)

        print(f'\nFor {player} top {top_k} by min return:')
        for e in top_k_min:
            print(e)


if __name__ == "__main__":
    cross_play_path = 'eval_games/territory_open/terro_test_700'
    # delete_disk_caches_for_function('get_cross_play_returns')
    all_metrics_paths = glob.glob(os.path.join(cross_play_path, "**", f"metrics.pkl"), recursive=True)
    player_returns_mean, player_returns_std, player_policies = get_cross_play_returns(all_metrics_paths)
    print_overall_performances(player_returns_mean, player_policies, 10)
    plot_path = plot_cross_play_returns(cross_play_path, player_returns_mean, player_returns_std, player_policies)
    save_as_example_output(plot_path, EXAMPLE_OUTPUTS_FOLDER)
