import glob
import itertools
import os
import pickle
from typing import List, NamedTuple
import numpy as np
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from scipy.stats import zscore
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import explained_variance_score, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import softmax
from king_of_the_hill.analysis.plotting.cross_play_returns import get_cross_play_returns
from king_of_the_hill.analysis.plotting.filters import filter_by_deaths, filter_by_mutual_vision
from king_of_the_hill.analysis.plotting.helpers import get_relative_player_positions, get_relative_hill_positions, \
    action_to_one_hot, get_var_explained_via_plsr
from king_of_the_hill.wandb_api import make_dir_if_not_exists


def get_feature(feature_name: str, metrics: dict, ep_idx: int, player: str, other_player: str) -> np.ndarray:
    if feature_name == 'Relative Player Position':
        return get_relative_player_positions(metrics, ep_idx, player, other_player)
    elif feature_name == 'Relative Hill Position':
        return get_relative_hill_positions(metrics, ep_idx, player)
    elif feature_name == 'Player Action':
        return action_to_one_hot(metrics['actions'][ep_idx], [player, other_player])[player]
    elif feature_name == 'Opponent Action':
        return action_to_one_hot(metrics['actions'][ep_idx], [player, other_player])[other_player]
    elif feature_name == 'Opponent Action Distribution':
        return softmax(metrics['activations'][ep_idx][other_player]['action_logits'], axis=1)
    else:
        raise Exception(f'Unknown feature: {feature_name}')


class CrossPLSRResults(NamedTuple):
    mean_returns: np.ndarray
    mean_exp_vars: np.ndarray
    mean_self_perm_exp_vars: np.ndarray
    mean_other_perm_exp_vars: np.ndarray


@cache_to_disk(31)  # Cache for 31 days
def get_cross_plsr(all_metrics_paths: List[str],
                   player_returns: dict,
                   player_policies: dict,
                   filter_out_deaths: bool,
                   filter_for_mutual_vision: bool,
                   permute_distance: int,
                   self_feature_names: List[str],
                   other_feature_names: List[str]) -> CrossPLSRResults:
    with open(all_metrics_paths[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    num_episodes = len(example_pkl_file['rewards'])
    player_ids = list(player_policies.keys())

    if len(player_ids) != 2:
        raise Exception(f'Script only designed for 2 players, not {len(player_ids)}. Please implement')

    mean_explained_variances = {}
    mean_self_perm_explained_variances = {}
    mean_other_perm_explained_variances = {}

    for player in player_ids:
        other_player = [p for p in player_ids if p != player][0]
        policies = player_policies[player]
        for policy in tqdm(policies):
            relevant_metrics_paths = [m for m in all_metrics_paths if policy in m]
            explained_variances = []
            self_perm_explained_variances = []
            other_perm_explained_variances = []

            for metrics_path in relevant_metrics_paths:
                with open(metrics_path, 'rb') as fp:
                    metrics = pickle.load(fp)

                # Check for filtering out timesteps with dead agents
                if filter_out_deaths:
                    metrics = filter_by_deaths(metrics, player_ids)

                # Check for filtering only for timesteps where agents see each other
                if filter_for_mutual_vision:
                    metrics = filter_by_mutual_vision(metrics, player_ids)

                for ep_idx in range(num_episodes):
                    # make player activations the variable to predict
                    y = metrics['activations'][ep_idx][player]['lstm_hidden']

                    # Let features be composed of own and opponent features
                    self_features = [
                        get_feature(name, metrics=metrics, ep_idx=ep_idx, player=player, other_player=other_player) for
                        name in
                        self_feature_names]
                    opponent_features = [
                        get_feature(name, metrics=metrics, ep_idx=ep_idx, player=player, other_player=other_player) for
                        name in
                        other_feature_names]
                    X = np.nan_to_num(zscore(np.hstack(self_features + opponent_features), ddof=1))
                    explained_variances.append(get_var_explained_via_plsr(X=X, y=y))

                    # Permute self features
                    self_permuted = np.roll(np.hstack(self_features), shift=permute_distance)
                    X_self_permuted = np.nan_to_num(
                        zscore(np.hstack([self_permuted, np.hstack(opponent_features)]), ddof=1))
                    self_perm_explained_variances.append(get_var_explained_via_plsr(X=X_self_permuted, y=y))

                    # Permute opponent features
                    opponent_permuted = np.roll(np.hstack(opponent_features), shift=permute_distance)
                    X_opponent_permuted = np.nan_to_num(
                        zscore(np.hstack([np.hstack(self_features), opponent_permuted]), ddof=1))
                    other_perm_explained_variances.append(get_var_explained_via_plsr(X=X_opponent_permuted, y=y))

            mean_explained_variances[policy] = np.mean(explained_variances)
            mean_self_perm_explained_variances[policy] = np.mean(self_perm_explained_variances)
            mean_other_perm_explained_variances[policy] = np.mean(other_perm_explained_variances)

    # Flatten nested dict into single dict
    mean_returns = {policy: mean_return for v in player_returns.values() for policy, mean_return in v.items()}
    mean_returns_arr = np.array([np.mean(list(return_dict.values())) for policy, return_dict in mean_returns.items()])
    mean_exp_vars_arr = np.array([mean_explained_variances[p] for p in mean_returns.keys()])
    mean_self_perm_exp_vars_arr = np.array([mean_self_perm_explained_variances[p] for p in mean_returns.keys()])
    mean_other_perm_exp_vars_arr = np.array([mean_other_perm_explained_variances[p] for p in mean_returns.keys()])

    return CrossPLSRResults(mean_returns_arr, mean_exp_vars_arr, mean_self_perm_exp_vars_arr,
                            mean_other_perm_exp_vars_arr)


def plot_plsr(plot_path: str,
              mean_returns: np.ndarray,
              mean_explained_variances: np.ndarray,
              filter_out_deaths: bool,
              filter_for_mutual_vision: bool,
              vs_permuted: bool,
              predictor_property: str):
    # Make Scatter Plot
    fig = plt.figure(figsize=(10, 6))
    flat_variances = mean_explained_variances.flatten()
    flat_rewards = mean_returns.flatten()
    plt.scatter(flat_variances, flat_rewards)

    xlabel = 'Variance Explained'
    title = f'PLSR Agent Episode Reward vs. {predictor_property} Explained'
    bar_title = f'PLSR Agent {predictor_property} Variance Explained'
    fig_name = f"plsr_{predictor_property.lower().replace(' ', '_')}"

    if filter_out_deaths:
        title += '+ Only Alive Filter'
        bar_title += '+ Only Alive Filter'
        fig_name += '-death_filtered'

    if filter_for_mutual_vision:
        title += '+ Mutual Vision Filter'
        bar_title += '+ Mutual Vision Filter'
        fig_name += '-mutual_vision_filtered'

    if vs_permuted:
        xlabel += ' (Base - Permuted)'
        fig_name += '-vs_permuted'

    plot_dir = f'{plot_path}/plots'
    make_dir_if_not_exists(plot_dir)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Mean Episode Reward')
    z = np.polyfit(flat_variances, flat_rewards, 1)
    p = np.poly1d(z)
    plt.plot(flat_variances, p(flat_variances), "r--")
    plt.savefig(f'{plot_dir}/{fig_name}.png')
    plt.show()

    # Bar chart
    fig = plt.figure(figsize=(6, 6))
    losing_var_explained = flat_variances[flat_rewards <= 0]
    winning_var_explained = flat_variances[flat_rewards > 0]

    bar_labels = ['losing', 'winning']
    var_explained_by_type = [losing_var_explained.mean(), winning_var_explained.mean()]
    plt.bar(bar_labels, var_explained_by_type, color='purple', width=0.4)

    plt.xlabel('Policies')
    plt.ylabel(xlabel)
    plt.title(bar_title)
    plt.savefig(f'{plot_dir}/{fig_name}_bar.png')
    plt.show()


if __name__ == "__main__":
    # TODO control via argparser
    # Base args to configure
    permute_distance = 100
    # Enable to clear function caching
    # delete_disk_caches_for_function('get_cross_plsr')

    # Path-related stuff
    cross_play_path = 'king_of_the_hill/evaluation/koth-multi-train_cross_4x'
    all_metrics_paths = glob.glob(os.path.join(cross_play_path, "**", f"metrics.pkl"), recursive=True)
    player_returns, _, player_policies = get_cross_play_returns(all_metrics_paths)

    options_dict = {
        'all_metrics_paths': [all_metrics_paths],
        'player_returns': [player_returns],
        'player_policies': [player_policies],
        'filter_out_deaths': [False],  # [False, True],
        'filter_for_mutual_vision': [False],  # [False, True],
        'permute_distance': [permute_distance],
        'self_feature_names': [['Relative Hill Position', 'Player Action']],
        'other_feature_names': [['Relative Player Position', 'Opponent Action']]
    }

    keys = list(options_dict.keys())

    # Try all possible combinations
    combinations = list(itertools.product(*list(options_dict.values())))
    for idx, combination in enumerate(combinations):
        print(f'Computing combination: {idx + 1}/{len(combinations)}')
        # Call the function with the combination of parameter values
        kwargs = {keys[i]: val for i, val in enumerate(combination)}
        # Call the function with the combination of parameter values
        results = get_cross_plsr(**kwargs)

        # plot_plsr(plot_path=cross_play_path,
        #           mean_returns=mean_returns,
        #           mean_explained_variances=mean_explained_variances,
        #           filter_out_deaths=combination[2],
        #           filter_for_mutual_vision=combination[3],
        #           vs_permuted=combination[4],
        #           predictor_property=combination[6])
