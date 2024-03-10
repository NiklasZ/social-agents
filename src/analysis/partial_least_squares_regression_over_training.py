"""
his uses PLSR to predict the activations of a player (or the PLSC of both) using features such as position.
"""
# TODO code needs to be updated to use new evaluation metrics structure

import itertools
import pickle
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from tqdm import tqdm
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function

from analysis.filters import filter_by_deaths, filter_by_mutual_vision
from analysis.helpers import get_action_softmax, action_to_one_hot, find_metrics_paths, get_relative_hill_positions, \
    get_relative_player_positions, score_shared_activity_via_plsc, get_var_explained_via_plsr
from utils.wandb_api import make_dir_if_not_exists

player_colours = {'player_0': 'red', 'player_1': 'blue'}


def get_actions(use_actions: str, metrics: dict, ep_idx: int, player: str) -> np.ndarray:
    if use_actions == 'action probabilities':
        return get_action_softmax(metrics['activations'][ep_idx][player]['action_logits'], False)
    elif use_actions == 'player actions':
        return action_to_one_hot(metrics['actions'][ep_idx], [player])[player]
    elif use_actions == 'player actions and probabilities':
        return np.hstack([get_action_softmax(metrics['activations'][ep_idx][player]['action_logits'], False),
                          action_to_one_hot(metrics['actions'][ep_idx], [player])[player]])
    else:
        raise Exception(f'Unknown actions option: {use_actions}')


@cache_to_disk(31)  # Cache for 31 days
def get_training_plsr(run_path: str,
                      run_id: str,
                      checkpoint_range: List[int],
                      policy_ids: List[str],
                      filter_out_deaths: bool,
                      filter_for_mutual_vision: bool,
                      permute_distance: int,
                      use_actions: str,
                      use_shared_activity_via_plsc: bool) -> dict:
    if len(policy_ids) != 2:
        raise NotImplementedError(
            f'Script only designed for 2 players per episode, not {len(policy_ids)}. Please implement')

    metrics_paths = find_metrics_paths(run_id=run_id,
                                       run_path=run_path,
                                       checkpoint_range=checkpoint_range,
                                       policy_ids=policy_ids)

    with open(metrics_paths[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    num_episodes = len(example_pkl_file['rewards'])
    player_ids = sorted([agent_info['player'] for agent_info in example_pkl_file['other']['agent_info']])

    explained_variances = {p: {} for p in player_ids}

    for metrics_path, c in tqdm(list(zip(metrics_paths, checkpoint_range))):
        with open(metrics_path, 'rb') as fp:
            metrics = pickle.load(fp)

        # Check for filtering out timesteps with dead agents
        if filter_out_deaths:
            metrics = filter_by_deaths(metrics, player_ids)

        # Check for filtering only for timesteps where agents see each other
        if filter_for_mutual_vision:
            metrics = filter_by_mutual_vision(metrics, player_ids)

        for player, policy in zip(player_ids, policy_ids):
            other_player = [p for p in player_ids if p != player][0]
            explained_variances[player][c] = {'all_features': [], 'permuted_self': [], 'permuted_other': []}

            for ep_idx in range(num_episodes):

                # Own properties
                relative_hill_pos = get_relative_hill_positions(metrics, ep_idx, player)
                player_action_dist = get_actions(use_actions=use_actions, player=player, ep_idx=ep_idx, metrics=metrics)

                # Other player properties
                relative_player_pos = get_relative_player_positions(metrics, ep_idx, player, other_player)
                other_player_action_dist = get_actions(use_actions=use_actions, player=other_player, ep_idx=ep_idx,
                                                       metrics=metrics)

                # PLSR using all features
                X = np.concatenate(
                    [relative_hill_pos, player_action_dist, relative_player_pos, other_player_action_dist], axis=1)

                X = np.nan_to_num(zscore(X, ddof=1))

                # Let y (target) either be a PLSC of both player activations
                if use_shared_activity_via_plsc:
                    activations = [metrics['activations'][ep_idx][pl]['lstm_hidden'] for pl in [player, other_player]]

                    y = score_shared_activity_via_plsc(activations)[0]
                # Or just the current player's activation
                else:
                    y = metrics['activations'][ep_idx][player]['lstm_hidden']

                explained_variances[player][c]['all_features'].append(get_var_explained_via_plsr(X=X, y=y))

                # PLSR with permuted player features
                X = np.concatenate(
                    [np.roll(relative_hill_pos, shift=permute_distance),
                     np.roll(player_action_dist, shift=permute_distance),
                     relative_player_pos,
                     other_player_action_dist], axis=1)
                X = np.nan_to_num(zscore(X, ddof=1))
                explained_variances[player][c]['permuted_self'].append(get_var_explained_via_plsr(X=X, y=y))

                # PLSR with permuted other player features
                X = np.concatenate(
                    [relative_hill_pos,
                     player_action_dist,
                     np.roll(relative_player_pos, shift=permute_distance),
                     np.roll(other_player_action_dist, shift=permute_distance)], axis=1)
                X = np.nan_to_num(zscore(X, ddof=1))
                explained_variances[player][c]['permuted_other'].append(get_var_explained_via_plsr(X=X, y=y))

    return explained_variances


def plot_training_plsr(run_id: str,
                       run_path: str,
                       main_player: str,
                       explained_variances: dict,
                       filter_out_deaths: bool,
                       filter_for_mutual_vision: bool,
                       vs_permuted: bool,
                       use_actions: str,
                       use_shared_activity_via_plsc: bool):
    fig = plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 16})

    # Creating plot
    box_count = len(explained_variances[main_player])
    box_width = 0.5
    box_spacing = 3

    all_features_var_exp = np.vstack([explained_variances[main_player][c]['all_features'] for c in
                                      explained_variances[main_player].keys()])
    permuted_other_features_var_exp = np.vstack([explained_variances[main_player][c]['permuted_other'] for c in
                                                 explained_variances[main_player].keys()])
    permuted_self_features_var_exp = np.vstack([explained_variances[main_player][c]['permuted_self'] for c in
                                                explained_variances[main_player].keys()])

    if vs_permuted:
        all_minus_self_perm = all_features_var_exp - permuted_self_features_var_exp
        all_minus_other_perm = all_features_var_exp - permuted_other_features_var_exp
        bplot2 = plt.boxplot(all_minus_other_perm.tolist(), patch_artist=True,
                             flierprops={'markerfacecolor': 'silver'},
                             boxprops=dict(facecolor="silver"),
                             positions=np.arange(box_count) * box_spacing + 0.8, widths=box_width)

        bplot1 = plt.boxplot(all_minus_self_perm.tolist(), patch_artist=True,
                             flierprops={'markerfacecolor': player_colours[main_player]},
                             boxprops=dict(facecolor=player_colours[main_player]),
                             positions=np.arange(box_count) * box_spacing + 0.2, widths=box_width)
        plt.ylim([-0.2, 0.3])
        plt.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['Full - Self Perm', 'Full - Other Perm'],
                   loc='upper right')
        bplots = [bplot1, bplot2]

    else:
        bplot3 = plt.boxplot(all_features_var_exp.tolist(), patch_artist=True, flierprops={'markerfacecolor': 'grey'},
                             boxprops=dict(facecolor="grey"),
                             positions=np.arange(box_count) * box_spacing + 1.6, widths=box_width)

        bplot2 = plt.boxplot(permuted_other_features_var_exp.tolist(), patch_artist=True,
                             flierprops={'markerfacecolor': 'silver'},
                             boxprops=dict(facecolor="silver"),
                             positions=np.arange(box_count) * box_spacing + 0.8, widths=box_width)

        bplot1 = plt.boxplot(permuted_self_features_var_exp.tolist(), patch_artist=True,
                             flierprops={'markerfacecolor': player_colours[main_player]},
                             boxprops=dict(facecolor=player_colours[main_player]),
                             positions=np.arange(box_count) * box_spacing + 0.2, widths=box_width)
        plt.ylim([0, 1])
        plt.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]], ['Perm Self', 'Perm Other', 'Full'],
                   loc='upper right')
        bplots = [bplot1, bplot2, bplot3]

    plt.axhline(0, color='black', linestyle='--')  # Adding dashed horizontal line at y=0

    # if discretise_actions:
    #     plt.ylim([-1000, 1000])

    # Plot labels
    title = f'{run_id},{main_player} - Variance Explained by Features via PLSR'
    fig_name = f'{run_id}_{main_player}_plsr_over_training'
    ylabel = 'Variance Explained'

    if filter_out_deaths:
        title += '+ Only Alive Filter'
        fig_name += '-death_filtered'

    if filter_for_mutual_vision:
        title += ' + Mutual Vision Filter'
        fig_name += '-mutual_vision_filtered'


    title += f' + {use_actions}'
    fig_name += f"-{use_actions.replace(' ','_')}"

    if use_shared_activity_via_plsc:
        title += ' + Shared activity via PLSC'
        fig_name += '-shared_activity_w_plsc'

    if vs_permuted:
        ylabel += ' (Full - Permuted)'
        fig_name += '-vs_permuted'

    plt.ylabel(ylabel)
    plt.xlabel('Training Iterations')
    ticks = np.arange(0, box_count * box_spacing, box_spacing)
    plt.xticks(ticks, list(explained_variances[main_player].keys()))
    plt.xticks(rotation=45)
    plt.title(title)

    # Colour medians
    for plot in bplots:
        for median in plot['medians']:
            median.set_color('black')

    # show plot
    chart_dir = f'{run_path}/plots/training_plsr'
    make_dir_if_not_exists(chart_dir)
    chart_path = f"{chart_dir}/{fig_name}.png"
    plt.savefig(chart_path)
    plt.show()


if __name__ == "__main__":
    # TODO control via argparser
    # Base args to configure
    permute_distance = 100
    # # Enable to clear function caching
    # delete_disk_caches_for_function('get_training_plsr')

    policy_ids = ['policy0', 'policy1']
    red_runs = ['run-1683514150', 'run-1683419885', 'run-1683705215']
    blue_runs = ['run-1683401773', 'run-1683629797', 'run-1683705215']
    players = ['player_0', 'player_1']

    # Path-related stuff
    run_path = 'king_of_the_hill/evaluation/koth-multi-train_4x_best'
    checkpoint_range = list(range(100, 1600, 200))

    options_dict = {
        'run_path': [run_path],
        'run_id': red_runs + blue_runs,
        'checkpoint_range': [checkpoint_range],
        'policy_ids': [policy_ids],
        'filter_out_deaths': [False],  # [False, True],
        'filter_for_mutual_vision': [False, True],
        'permute_distance': [permute_distance],
        'use_actions': ['player actions'], #  'action probabilities', 'player actions and probabilities'
        'use_shared_activity_via_plsc': [False]
    }

    # Try all possible combinations
    plsr_results = []
    combinations = list(itertools.product(*options_dict.values()))
    keys = list(options_dict.keys())
    for idx, combination in enumerate(combinations):
        print(f'Computing combination: {idx + 1}/{len(combinations)}')
        # Create kwargs dictionary
        kwargs = {keys[i]: val for i, val in enumerate(combination)}
        # Call the function with the combination of parameter values
        explained_variances = get_training_plsr(**kwargs)
        plsr_results.append(explained_variances)

        for p in players:
            for vs_permuted in [False, True]:
                plot_training_plsr(run_id=kwargs['run_id'],
                                   run_path=run_path,
                                   main_player=p,
                                   explained_variances=explained_variances,
                                   filter_out_deaths=kwargs['filter_out_deaths'],
                                   filter_for_mutual_vision=kwargs['filter_for_mutual_vision'],
                                   vs_permuted=vs_permuted,
                                   use_actions=kwargs['use_actions'],
                                   use_shared_activity_via_plsc=kwargs['use_shared_activity_via_plsc'])
