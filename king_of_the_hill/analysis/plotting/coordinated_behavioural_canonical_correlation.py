# Tries to find the number of coordinated dimensions in behaviours via canonical correlation analysis
import itertools
import pickle
from typing import List
import textwrap
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
from sklearn.cross_decomposition import PLSRegression
from numpy.random import randint
from tqdm import tqdm

from king_of_the_hill.analysis.plotting.filters import filter_by_mutual_vision
from king_of_the_hill.analysis.plotting.helpers import find_metrics_paths, get_relative_hill_positions, \
    get_action_softmax, get_relative_player_positions, action_to_one_hot, get_kwargs_matching, \
    get_var_explained_via_plsr, random_shift
from king_of_the_hill.wandb_api import make_dir_if_not_exists


@cache_to_disk(31)  # Cache for 31 days
def get_coordinated_dimensions_via_cca(run_path: str,
                                       run_id: str,
                                       checkpoint_range: List[int],
                                       policy_ids: List[str],
                                       filter_for_mutual_vision: bool,
                                       null_distribution_shuffle_count=100,
                                       null_distribution_cutoff_pct=97.5,
                                       max_cca_iter=2000,
                                       num_components=None) -> (dict, dict, dict, int):
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
    player_1, player_2 = player_ids

    # dict of {checkpoint:{ep index: {player_id: 2D np array of canonical variables}}}
    coordinated_dimensions = {c: {} for c in checkpoint_range}
    # dict of {checkpoint:{ep index: np.ndarray of correlations}}
    all_pearson_correlations = {c: {} for c in checkpoint_range}
    # dict of {checkpoint:{ep index: cutoff correlation}}
    cutoff_correlations = {c: {} for c in checkpoint_range}

    for metrics_path, c in tqdm(list(zip(metrics_paths, checkpoint_range))):
        with open(metrics_path, 'rb') as fp:
            metrics = pickle.load(fp)

        # Check for filtering only for timesteps where agents see each other
        if filter_for_mutual_vision:
            metrics = filter_by_mutual_vision(metrics, player_ids)

        for ep_idx in range(num_episodes):

            player_actions_one_hot = action_to_one_hot(metrics['actions'][ep_idx], player_ids)

            # Player 1 behaviours
            p1_relative_hill_pos = get_relative_hill_positions(metrics, ep_idx, player_1)
            p1_actions = player_actions_one_hot[player_1]
            p1_behaviours = np.hstack([p1_relative_hill_pos, p1_actions])
            p1_behaviours_z = np.nan_to_num(zscore(p1_behaviours, ddof=1))

            # Player 2 behaviours
            p2_relative_hill_pos = get_relative_hill_positions(metrics, ep_idx, player_2)
            p2_actions = player_actions_one_hot[player_2]
            p2_behaviours = np.hstack([p2_relative_hill_pos, p2_actions])
            p2_behaviours_z = np.nan_to_num(zscore(p2_behaviours, ddof=1))

            if num_components is None:
                num_components = np.min(
                    [np.linalg.matrix_rank(p1_behaviours_z), np.linalg.matrix_rank(p2_behaviours_z)])

            cca = CCA(n_components=num_components, max_iter=max_cca_iter)
            # Get canonical values
            bv1_values, bv2_values = cca.fit_transform(X=p1_behaviours_z, y=p2_behaviours_z)
            # Yields a correlation for every possible bv1 and bv2 component pair. Sorted by descending
            # correlation.
            pearson_correlations = [pearsonr(bv1_values[:, c], bv2_values[:, c])[0] for c in range(num_components)]

            # Permute behaviours and run it again
            permuted_correlations = []
            for i in range(null_distribution_shuffle_count):
                bv1_p_values, bv2_p_values = cca.fit_transform(X=random_shift(p1_behaviours_z),
                                                               y=random_shift(p2_behaviours_z))
                permuted_correlations += [pearsonr(bv1_p_values[:, c], bv2_p_values[:, c])[0] for c in
                                          range(num_components)]

            cut_off_val = np.percentile(permuted_correlations, null_distribution_cutoff_pct)
            significant_dimensions = pearson_correlations > cut_off_val

            all_pearson_correlations[c][ep_idx] = pearson_correlations
            cutoff_correlations[c][ep_idx] = cut_off_val

            if np.any(significant_dimensions):
                coordinated_dimensions[c][ep_idx] = {player_1: bv1_values[:, significant_dimensions],
                                                     player_2: bv2_values[:, significant_dimensions]}
    return coordinated_dimensions, all_pearson_correlations, cutoff_correlations, num_episodes


@cache_to_disk(31)  # Cache for 31 days
def variance_explained_by_coordinated_dims(run_path: str,
                                           coordinated_dimensions: dict,
                                           run_id: str,
                                           policy_ids: List[str],
                                           checkpoint_range: List[int],
                                           filter_for_mutual_vision: bool,
                                           null_distribution_shuffle_count=100,
                                           ) -> (np.ndarray, np.ndarray):
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

    vars_explained = np.zeros((len(checkpoint_range), num_episodes, len(player_ids)))
    mean_perm_vars_explained = np.zeros((len(checkpoint_range), num_episodes, len(player_ids)))

    for c_idx, (metrics_path, c) in tqdm(list(enumerate(zip(metrics_paths, checkpoint_range)))):

        with open(metrics_path, 'rb') as fp:
            metrics = pickle.load(fp)

        # Check for filtering only for timesteps where agents see each other
        if filter_for_mutual_vision:
            metrics = filter_by_mutual_vision(metrics, player_ids)

        for ep_idx in coordinated_dimensions[c]:
            activations = metrics['activations'][ep_idx]
            for p_idx, player_id in enumerate(coordinated_dimensions[c][ep_idx]):
                X = coordinated_dimensions[c][ep_idx][player_id]
                y = activations[player_id]['lstm_hidden']
                n_components = X.shape[1]  # Number of coordinating dimensions
                vars_explained[c_idx][ep_idx][p_idx] = get_var_explained_via_plsr(X=X, y=y, n_components=n_components)

                permuted_vars_explained = []
                for i in range(null_distribution_shuffle_count):
                    permuted_vars_explained.append(
                        get_var_explained_via_plsr(X=X, y=random_shift(y), n_components=n_components))
                mean_perm_vars_explained[c_idx][ep_idx][p_idx] = np.mean(permuted_vars_explained)

    return vars_explained, mean_perm_vars_explained


def plot_num_coordinated_dimensions(run_path: str,
                                    coordinated_dimensions: dict,
                                    pearson_correlations: dict,
                                    players: List[str],
                                    run_id: str,
                                    filter_for_mutual_vision: bool,
                                    num_eps: int):
    first_player = players[0]  # Can be any player, really.
    some_train_step = list(pearson_correlations.keys())[0]
    num_components = len(pearson_correlations[some_train_step][0])
    num_dimensions = np.zeros((len(coordinated_dimensions), num_eps))
    for train_idx, training_iteration in enumerate(coordinated_dimensions):
        for ep_idx in coordinated_dimensions[training_iteration]:
            num_dimensions[train_idx][ep_idx] = coordinated_dimensions[training_iteration][ep_idx][first_player].shape[
                1]

    fig = plt.figure(figsize=(14, 7))
    bplot = plt.boxplot(num_dimensions.T, patch_artist=True,
                        flierprops={'markerfacecolor': 'silver'},
                        boxprops=dict(facecolor="silver"),
                        showmeans=True)

    for median in bplot['medians']:
        median.set_color('black')
    for mean in bplot['means']:
        mean.set_color('blue')

    title = f'{run_id} - Number of Coordinated Behavioural Dimensions'
    fig_name = f'{run_id}_num_dims-{num_components}_components'
    if filter_for_mutual_vision:
        title += ' + Mutual Vision Filter'
        fig_name += '-mutual_vision_filtered'

    title = textwrap.fill(title, width=60)
    plt.title(title)
    # plt.gcf().subplots_adjust(left=0.05)  # Pad left margin
    plt.ylabel('# dims')
    plt.xlabel('training iteration')

    plt.xticks(range(1, len(coordinated_dimensions) + 1), coordinated_dimensions.keys())

    chart_dir = f'{run_path}/plots/coordinated_behavioural_canonical_correlation'
    make_dir_if_not_exists(chart_dir)
    chart_path = f"{chart_dir}/{fig_name}.png"
    plt.savefig(chart_path)
    plt.show()


def plot_var_explained_of_coordinated_dimensions(run_path: str,
                                                 vars_explained: np.ndarray,
                                                 mean_perm_vars_explained: np.ndarray,
                                                 run_id: str,
                                                 filter_for_mutual_vision: bool):
    flat_vars_explained = vars_explained.reshape(vars_explained.shape[0], -1)  # collapse last dimension
    flat_mean_perm_vars_explained = mean_perm_vars_explained.reshape(mean_perm_vars_explained.shape[0],
                                                                     -1)  # collapse last dimension
    delta_var_explained = flat_vars_explained - flat_mean_perm_vars_explained

    fig = plt.figure(figsize=(14, 7))
    bplot = plt.boxplot(delta_var_explained.T, patch_artist=True,
                        flierprops={'markerfacecolor': 'silver'},
                        boxprops=dict(facecolor="silver"),
                        showmeans=True)

    for median in bplot['medians']:
        median.set_color('black')

    plt.axhline(0, color='black', linestyle='--')  # Adding dashed horizontal line at y=0

    title = f'{run_id} - Variance Explained by Coordinated Behaviour'
    fig_name = f'{run_id}_var_explained'
    if filter_for_mutual_vision:
        title += ' + Mutual Vision Filter'
        fig_name += '-mutual_vision_filtered'

    title = textwrap.fill(title, width=60)
    plt.title(title)
    # plt.gcf().subplots_adjust(left=0.15)  # Pad left margin
    plt.ylabel('variance explained (base - permuted)')
    plt.xlabel('training iteration')

    plt.xticks(range(1, len(coordinated_dimensions) + 1), coordinated_dimensions.keys())

    chart_dir = f'{run_path}/plots/coordinated_behavioural_canonical_correlation'
    make_dir_if_not_exists(chart_dir)
    chart_path = f"{chart_dir}/{fig_name}.png"
    plt.savefig(chart_path)
    plt.show()


if __name__ == "__main__":
    # TODO control via argparser
    # Base args to configure
    permute_distance = 100
    # # Enable to clear function caching
    # delete_disk_caches_for_function('get_coordinated_dimensions_via_cca')

    policy_ids = ['policy0', 'policy1']
    red_runs = ['run-1683514150', 'run-1683419885']
    blue_runs = ['run-1683401773', 'run-1683629797']
    players = ['player_0', 'player_1']

    # Path-related stuff
    run_path = 'king_of_the_hill/evaluation/koth-multi-train_4x_best'
    checkpoint_range = list(range(100, 1600, 200))

    options_dict = {
        'run_path': [run_path],
        'run_id': red_runs + blue_runs,
        'checkpoint_range': [checkpoint_range],
        'policy_ids': [policy_ids],
        'filter_for_mutual_vision': [False, True],
        'num_components': [None, 5]
    }

    # Try all possible combinations
    cca_results = []
    plsr_results = []
    combinations = list(itertools.product(*options_dict.values()))
    keys = list(options_dict.keys())
    for idx, combination in enumerate(combinations):
        print(f'Computing combination: {idx + 1}/{len(combinations)}')
        # Create kwargs dictionary
        kwargs = {keys[i]: val for i, val in enumerate(combination)}
        # Call the function with the combination of parameter values
        coordinated_dimensions, pearson_correlations, cutoff_correlations, num_episodes = get_coordinated_dimensions_via_cca(
            **kwargs)
        cca_results.append(coordinated_dimensions)
        vars_explained, mean_perm_vars_explained = variance_explained_by_coordinated_dims(run_path=kwargs['run_path'],
                                                                                          coordinated_dimensions=coordinated_dimensions,
                                                                                          run_id=kwargs['run_id'],
                                                                                          policy_ids=kwargs[
                                                                                              'policy_ids'],
                                                                                          checkpoint_range=kwargs[
                                                                                              'checkpoint_range'],
                                                                                          filter_for_mutual_vision=
                                                                                          kwargs[
                                                                                              'filter_for_mutual_vision'])
        plsr_results.append((vars_explained, mean_perm_vars_explained))
        plot_num_coordinated_dimensions(run_path=kwargs['run_path'],
                                        coordinated_dimensions=coordinated_dimensions,
                                        pearson_correlations=pearson_correlations,
                                        num_eps=num_episodes,
                                        players=players,
                                        run_id=kwargs['run_id'],
                                        filter_for_mutual_vision=kwargs['filter_for_mutual_vision'])
        plot_var_explained_of_coordinated_dimensions(run_id=kwargs['run_id'],
                                                     filter_for_mutual_vision=kwargs['filter_for_mutual_vision'],
                                                     run_path=kwargs['run_path'],
                                                     vars_explained=vars_explained,
                                                     mean_perm_vars_explained=mean_perm_vars_explained)
