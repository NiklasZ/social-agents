import glob
import os
import pickle
from typing import List
import numpy as np
from cache_to_disk import cache_to_disk
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import explained_variance_score, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from analysis.filters import filter_by_deaths, filter_by_mutual_vision
from analysis.helpers import get_relative_player_positions
from utils.wandb_api import make_dir_if_not_exists


@cache_to_disk(31)  # Cache for 31 days
def get_plsr(run_paths: List[str],
             filter_out_deaths: bool,
             filter_for_mutual_vision: bool,
             vs_permuted: bool,
             permute_distance: int):
    example_run_path = run_paths[0]
    metrics_file_path = glob.glob(os.path.join(example_run_path, "**", "metrics.pkl"), recursive=True)
    with open(metrics_file_path[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    episode_count = len(example_pkl_file['activations'])
    player_ids = ['player_0', 'player_1']

    if len(player_ids) != 2:
        raise Exception(f'Script only designed for 2 players, not {len(player_ids)}. Please implement')

    mean_explained_variances = np.zeros((len(player_ids), len(run_paths)))
    mean_r2_scores = np.zeros((len(player_ids), len(run_paths)))
    ep_returns = np.zeros((len(player_ids), len(run_paths), episode_count))

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

        explained_variances = np.zeros((len(player_ids), episode_count))
        # sanity-check - should be more or less the same as explained variances
        r2_scores = np.zeros((len(player_ids), episode_count))

        for p_idx, p_id in enumerate(player_ids):
            other_p_id = [p for p in player_ids if p != p_id][0]
            for ep_id in range(episode_count):
                # Take difference between positions
                relative_positions = get_relative_player_positions(metrics, ep_id, p_id, other_p_id)
                player_hidden_activations = np.array(metrics['activations'][ep_id][p_id]['lstm_hidden'])

                # Calculate PLSR
                n_components = min(np.linalg.matrix_rank(relative_positions),
                                   np.linalg.matrix_rank(player_hidden_activations))
                pls = PLSRegression(n_components=n_components)
                pls.fit(relative_positions, player_hidden_activations)
                pred_activations = pls.predict(relative_positions)
                explained_variances[p_idx, ep_id] = explained_variance_score(player_hidden_activations,
                                                                             pred_activations)
                r2_scores[p_idx, ep_id] = r2_score(player_hidden_activations, pred_activations)

                # If we want to subtract the explained variance given by permuting the positions some k timesteps
                if vs_permuted:
                    pls = PLSRegression(n_components=n_components)
                    pls.fit(np.roll(relative_positions, shift=permute_distance), player_hidden_activations)
                    pred_activations = pls.predict(relative_positions)
                    explained_variances[p_idx, ep_id] -= explained_variance_score(player_hidden_activations,
                                                                                  pred_activations)
                    r2_scores[p_idx, ep_id] -= r2_score(player_hidden_activations, pred_activations)

                # Note we use base_metrics here to calculate the return over the entire episode, regardless of filters
                ep_returns[p_idx, run_idx, ep_id] = sum([r[p_id] for r in base_metrics['rewards'][ep_id]])

        mean_explained_variances[:, run_idx] = np.mean(explained_variances, axis=1)
        mean_r2_scores[:, run_idx] = np.mean(r2_scores, axis=1)
    mean_rewards = np.mean(ep_returns, axis=2)
    return mean_rewards, mean_explained_variances, mean_r2_scores


if __name__ == "__main__":
    # TODO control via argparser
    # Base args to configure
    evaluation_folder = 'koth-multi-train'
    filter_out_deaths = False
    filter_for_mutual_vision = True
    vs_permuted = False
    permute_distance = 50
    # TODO option to plot R2 score
    # Enable to clear function caching
    # delete_disk_caches_for_function('get_plsr')

    # Path-related stuff
    working_dir = os.getcwd()
    base_folder = 'social-agents'
    working_dir = working_dir[
                  :working_dir.find(base_folder) + len(base_folder)]  # Ensure we use social-agents as base dir
    run_batch_dir = f'{working_dir}/king_of_the_hill/evaluation/{evaluation_folder}'
    run_paths = sorted([f'{run_batch_dir}/{name}' for name in os.listdir(run_batch_dir) if 'run' in name])

    mean_rewards, mean_explained_variances, mean_r2_scores = get_plsr(run_paths,
                                                                      filter_out_deaths,
                                                                      filter_for_mutual_vision,
                                                                      vs_permuted,
                                                                      permute_distance)
    print(mean_explained_variances)
    print(mean_r2_scores)
    print(mean_rewards)
    fig = plt.figure(figsize=(10, 6))
    flat_variances = mean_explained_variances.flatten()
    flat_rewards = mean_rewards.flatten()
    plt.scatter(flat_variances, flat_rewards)

    xlabel = 'Variance Explained'
    title = 'PLSR Agent Return vs. Position Explained'
    fig_name = 'plsr_position'

    if filter_out_deaths:
        title += '+ Only Alive Filter'
        fig_name += '-death_filtered'

    if filter_for_mutual_vision:
        title += '+ Mutual Vision Filter'
        fig_name += '-mutual_vision_filtered'

    if vs_permuted:
        xlabel += ' (Base - Permuted)'
        fig_name += '-vs_permuted'

    plot_dir = f'{run_batch_dir}/plots'
    make_dir_if_not_exists(plot_dir)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Mean Return')
    z = np.polyfit(flat_variances, flat_rewards, 1)
    p = np.poly1d(z)
    plt.plot(flat_variances, p(flat_variances), "r--")
    plt.savefig(f'{plot_dir}/{fig_name}.png')
    plt.show()
