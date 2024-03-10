"""
Calculates the PLSC and delta-PLXC over training for a given set of checkpoints and activations.
"""


import glob
import os
import pickle
import time
from collections import defaultdict
from typing import List, Tuple
import numpy as np
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from numpy.linalg import LinAlgError
from scipy.linalg import svd
from scipy.stats import zscore
from tqdm import tqdm
from matplotlib import pyplot as plt

from analysis.helpers import save_as_example_output, EXAMPLE_OUTPUTS_FOLDER, embed_data_source
from utils.wandb_api import make_dir_if_not_exists


@cache_to_disk(31)
def get_plsc(all_metrics_paths: List[str],
             base_eval_path: str,
             checkpoint_iters: List[int],
             num_temporal_shuffles: int,
             plsc_levels: int,
             activation_layer: str,
             remove_zero_activations: bool,
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    example_file_path = all_metrics_paths[0]
    with open(example_file_path, 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    episode_count = len(example_pkl_file['general_metrics']['activations'])
    run_id = example_pkl_file['general_metrics']['other']['agent_info'][0]['run_id']
    policy_ids = [p['policy_id'] for p in example_pkl_file['general_metrics']['other']['agent_info']]
    shifted_indices = np.roll(np.arange(0, episode_count), 10)
    train_iterations = len(checkpoint_iters)

    assert len(policy_ids) == 2, 'This method is only designed for 2 agents'

    example_activations = next(iter(example_pkl_file['general_metrics']['activations'][0].values()))
    # activation_layer_names = list(example_activations.keys())
    _, activation_count = next(iter(example_activations.values())).shape
    # filter_names = ['death', 'mutual vision', 'exclusive vision']
    activation_masks = defaultdict(lambda: [])  # np.ones((train_iterations, episode_count, episode_length), dtype=bool)

    # TODO choose different names?
    # TODO index by checkpoint iteration and episode id
    # Structured as: {checkpoint_iteration: [episode_id: [activation_layer: np.ndarray]]}
    # Note that the shape of the activations can vary as not all environments have a fixed episode length.
    X1 = defaultdict(lambda: [])  # np.zeros((episode_count, episode_length, activation_count))
    X2 = defaultdict(lambda: [])  # np.zeros((train_iterations, episode_count, episode_length, activation_count))
    plscs = np.zeros((train_iterations, episode_count, plsc_levels))
    plsc_cross_pairs = np.zeros((train_iterations, episode_count, plsc_levels))
    plsc_temporal_shuffled = np.zeros((train_iterations, episode_count, plsc_levels))
    delta_plscs = np.zeros((train_iterations, episode_count, plsc_levels))
    delta_cross_pair_plscs = np.zeros((train_iterations, episode_count, plsc_levels))

    # Load in checkpoint data
    print('Loading checkpoints...')
    for i, c in tqdm(enumerate(checkpoint_iters)):
        path_pattern = f'{run_id}.checkpoint-{c}.{policy_ids[0]}__VS__{run_id}.checkpoint-{c}.{policy_ids[1]}'
        matching_paths = [p for p in all_metrics_paths if path_pattern in p]
        assert len(matching_paths) == 1, f'Found {len(matching_paths)} matching paths for pattern {path_pattern}'

        with open(matching_paths[0], 'rb') as fp:
            pkl_file = pickle.load(fp)
            # Get min episode length across all episodes in batch. We need

            for j in range(0, episode_count):
                X1[c].append(pkl_file['general_metrics']['activations'][j]['player_0'][activation_layer])
                X2[c].append(pkl_file['general_metrics']['activations'][j]['player_1'][activation_layer])
                activation_masks[c].append(np.ones(X1[c][-1].shape[0], dtype=bool))

                # TODO apply filters if working
                # Mask every timestep where at least one player is dead.
                # if death_filter:
                #     death_mask = [not (k['player_0'] or k['player_1']) for k in pkl_file['deaths'][j]]
                #     activation_masks[i, j] = np.logical_and(activation_masks[i, j], death_mask)
                #
                # # Masks every timestep where players can't see each other.
                # if mutual_vision_filter:
                #     vision_mask = [len(k['player_0']) and len(k['player_1']) for k in pkl_file['visibilities'][j]]
                #     activation_masks[i, j] = np.logical_and(activation_masks[i, j], vision_mask)
                #
                # # Masks every timestep where all players can see each other.
                # # Will still keep cases where one player can see the other, but not vice-versa.
                # if exclusive_vision_filter:
                #     vision_mask = [not (len(k['player_0']) and len(k['player_1'])) for k in pkl_file['visibilities'][j]]
                #     activation_masks[i, j] = np.logical_and(activation_masks[i, j], vision_mask)

    print('Calculation PLSC for each iteration...')
    for i, c in enumerate(checkpoint_iters):
        p, p_cp, t, d, d_cp = process_train_iteration(iteration=i,
                                                      it_X1=X1[c],
                                                      it_X2=X2[c],
                                                      it_activation_mask=activation_masks[c],
                                                      shifted_indices=shifted_indices,
                                                      n_temporal_shuffles=num_temporal_shuffles,
                                                      checkpoint_range_dir=base_eval_path,
                                                      plsc_levels=plsc_levels,
                                                      remove_zero_activations=remove_zero_activations)
        plscs[i] = p
        plsc_cross_pairs[i] = p_cp
        plsc_temporal_shuffled[i] = t
        delta_plscs[i] = d
        delta_cross_pair_plscs[i] = d_cp

    return plscs, plsc_cross_pairs, plsc_temporal_shuffled, delta_plscs, delta_cross_pair_plscs


# TODO doc method and args


def z_scored_covariance(A: np.ndarray, B: np.ndarray, filter_nan=True):
    """
    Yields a correlation matrix (i.e a covariance matrix of z-scored inputs). This is equivalent to Matlab's corr()
    See https://stackoverflow.com/questions/61624985/python-use-of-corrcoeff-to-achieve-matlabs-corr-function
    @param A: shape (n_samples, n_features)
    @param B: shape (n_samples, n_features)
    @param filter_nan: whether to remove nan values from the input matrices
    @return: correlation matrix (n_features, n_features)
    """
    A_z = np.nan_to_num(zscore(A)) if filter_nan else zscore(A)
    B_z = np.nan_to_num(zscore(B)) if filter_nan else zscore(B)
    cov = (np.dot(B_z.T, A_z) / B_z.shape[0])
    return cov


# How well a PLSC dim explains the variance (captures the patterns) of the activation.
# def explained_variances(latent_factors: np.ndarray, activations: np.ndarray):
#     return np.nanmean(np.square(z_scored_covariance(latent_factors[:, :plsc_levels], activations, filter_nan=False)),
#                       axis=0)


def score_shared_space(A: np.ndarray, B: np.ndarray):
    """
    Constructs a correlation matrix on z-scored inputs, then performs SVD to obtain latent factors.
    We then multiply the latent factors by the original data to get the scores for each data point.
    @param A: Matrix of shape (n_samples, n_features)
    @param B: Matrix of shape (n_samples, n_features)
    @return:
        U of shape (n_features, n_features)
        V of shape (n_features, n_features)
        S of shape (n_features)
        dU of shape (n_samples, n_features)
        dV of shape (n_samples, n_features)
    """
    # Get correlation matrix
    cov_mat = np.nan_to_num(np.dot(A.T, B) / A.shape[0])
    # SVD on correlation matrix
    U, S, V_T = svd(cov_mat)
    V = V_T.T

    # Score correlation with latent factors:
    dU = np.dot(A, U)
    dV = np.dot(B, V)
    return U, V, S, dU, dV


def process_train_iteration(iteration: int,
                            it_X1: List[np.ndarray],
                            it_X2: List[np.ndarray],
                            it_activation_mask: List[np.ndarray],
                            shifted_indices: np.ndarray,
                            n_temporal_shuffles: int,
                            checkpoint_range_dir: str,
                            plsc_levels: int,
                            remove_zero_activations: bool) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode_count = len(it_X1)
    plsc_corr = np.zeros((episode_count, plsc_levels))
    plsc_corr_cp = np.zeros((episode_count, plsc_levels))
    plsc_corr_temporal_shuffled = np.zeros((episode_count, plsc_levels))
    delta_corr = np.zeros((episode_count, plsc_levels))
    delta_corr_cp = np.zeros((episode_count, plsc_levels))
    start_time = time.time()
    shuffle_distances = None
    zero_activations_removed = 0

    for ep_idx in range(0, episode_count):
        try:
            sh_ep_idx = shifted_indices[ep_idx]

            x1 = it_X1[ep_idx][it_activation_mask[ep_idx]]
            x2 = it_X2[ep_idx][it_activation_mask[ep_idx]]
            x2_sh = it_X2[sh_ep_idx][it_activation_mask[sh_ep_idx]]

            if remove_zero_activations:
                # Remove zero activations
                x1 = x1[:, ~np.all(x1 == 0, axis=0)]
                x2 = x2[:, ~np.all(x2 == 0, axis=0)]
                x2_sh = x2_sh[:, ~np.all(x2_sh == 0, axis=0)]
                zero_activations_removed += it_X1[ep_idx].shape[0] - x1.shape[0] + it_X2[ep_idx].shape[0] - x2.shape[0]
                if zero_activations_removed > 0:
                    print(f'Removed {zero_activations_removed} zero activations')

            # Z-score activations
            x1_z = np.nan_to_num(zscore(x1, ddof=1))
            x2_z = np.nan_to_num(zscore(x2, ddof=1))
            x2_cross_pair_z = np.nan_to_num(zscore(x2_sh, ddof=1))

            min_size = np.min([x1_z.shape[0], x2_cross_pair_z.shape[0]])

            # Get scores for normal PLSC
            _, _, S, dU, dV = score_shared_space(x1_z, x2_z)
            _, _, _, dU_cp, dV_cp = score_shared_space(x1_z[:min_size], x2_cross_pair_z[:min_size])

            corrs = np.diag(z_scored_covariance(dU[:, :plsc_levels], dV[:, :plsc_levels]))
            corrs_cp = np.diag(z_scored_covariance(dU_cp[:, :plsc_levels], dV_cp[:, :plsc_levels]))

            # Use temporal shift as null distribution
            temporal_corr = np.zeros((n_temporal_shuffles, plsc_levels))
            temporal_corr_cp = np.zeros((n_temporal_shuffles, plsc_levels))
            shuffle_distances = np.random.randint(10, min_size - 10, (n_temporal_shuffles, 3))

            for sh_idx, (s1, s2, s3) in enumerate(shuffle_distances):
                # Shift all
                x1_shifted = np.roll(x1_z, s1, axis=0)
                x2_shifted = np.roll(x2_z, s2, axis=0)
                x2_cross_pair_shifted = np.roll(x2_cross_pair_z, s3, axis=0)

                # Score on shifted
                _, _, S_sh, dU_sh, dV_sh = score_shared_space(x1_shifted, x2_shifted)
                _, _, _, dU_sh_cp, dV_sh_cp = score_shared_space(x1_shifted[:min_size],
                                                                 x2_cross_pair_shifted[:min_size])

                # Get temporal correlations
                temporal_corr[sh_idx] = np.diag(z_scored_covariance(dU_sh[:, :plsc_levels], dV_sh[:, :plsc_levels]))
                temporal_corr_cp[sh_idx] = np.diag(
                    z_scored_covariance(dU_sh_cp[:, :plsc_levels], dV_sh_cp[:, :plsc_levels]))

            # Regular PLSC
            plsc_corr[ep_idx] = corrs
            plsc_corr_cp[ep_idx] = corrs_cp
            plsc_corr_temporal_shuffled[ep_idx] = np.median(temporal_corr, axis=0)

            # Delta PLSC (subtract null distribution)
            delta_corr[ep_idx] = corrs - np.median(temporal_corr, axis=0)
            delta_corr_cp[ep_idx] = corrs_cp - np.median(temporal_corr_cp, axis=0)

        # Sometimes (~1% of episodes) an SVD fails with an "SVD did not converge" error
        # It's not really clear why this happens, so we must save the function arguments
        # and recreate it later.
        except LinAlgError as error:
            print(error)
            exceptions_dir = f'{checkpoint_range_dir}/exceptions'
            make_dir_if_not_exists(exceptions_dir)
            d = {'iteration': iteration,
                 'ep_idx': ep_idx,
                 'sh_ep_idx': sh_ep_idx,
                 'x1': it_X1[ep_idx],
                 'x2': it_X2[ep_idx],
                 'x2_cp': it_X2[sh_ep_idx],
                 'shuffle_distances': shuffle_distances}
            with open(f"{exceptions_dir}/{iteration}_{ep_idx}.pkl", 'wb') as f:
                pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

            delta_corr[ep_idx] = np.zeros(plsc_levels)
            delta_corr_cp[ep_idx] = np.zeros(plsc_levels)

        current_time = time.time()
        avg_time = (current_time - start_time) / (ep_idx + 1)
        print(
            f'it:{iteration}, ep:{ep_idx + 1}/{episode_count} - ETA {avg_time * (episode_count - ep_idx + 1) / 60:.1f}min')

    if remove_zero_activations:
        print(f'Zero activations removed: {zero_activations_removed}')

    return plsc_corr, plsc_corr_cp, plsc_corr_temporal_shuffled, delta_corr, delta_corr_cp


def plot_delta_plsc(delta_corr: np.ndarray,
                    delta_corr_cp: np.ndarray,
                    checkpoint_iters: List[int],
                    checkpoint_range_dir: str,
                    death_filter: bool,
                    mutual_vision_filter: bool,
                    exclusive_vision_filter: bool) -> str:
    """
    Plots the delta PLSC for each iteration.
    @param delta_corr: delta PLSC for each iteration of shape (train_iterations, episode_count, plsc_levels)
    @param delta_corr_cp: delta cross-pair PLSC for each iteration of shape (train_iterations, episode_count, plsc_levels)
    @return: dir under which plots were generated
    """
    filter_names = ['death', 'mutual vision', 'exclusive vision']
    filters_used = [n for n, f in zip(filter_names, [death_filter, mutual_vision_filter, exclusive_vision_filter]) if f]
    train_iterations = delta_corr.shape[0]
    plot_dir = f'{checkpoint_range_dir}/plots/plsc_over_training'

    for i in range(plsc_levels):
        fig = plt.figure(figsize=(20, 10))
        plt.rcParams.update({'font.size': 20, 'text.usetex': True})

        # Creating plot
        box_count = delta_corr.shape[0]
        box_width = 0.5
        box_spacing = 3
        bplot2 = plt.boxplot(delta_corr_cp[:, :, i].T, patch_artist=True, flierprops={'markerfacecolor': 'silver'},
                             boxprops=dict(facecolor="silver"),
                             positions=np.arange(box_count) * box_spacing + 0.8, widths=box_width)
        bplot1 = plt.boxplot(delta_corr[:, :, i].T, patch_artist=True, flierprops={'markerfacecolor': 'blue'},
                             boxprops=dict(facecolor="blue"),
                             positions=np.arange(box_count) * box_spacing + 0.2, widths=box_width)

        plt.axhline(y=0, color='gray', linestyle='dashed')
        plt.ylim([-0.2, 1])
        plt.ylabel(r'$\Delta$PLSC' + '\n (Base - Temporal)')
        plt.xlabel('Iterations')
        ticks = np.arange(0, train_iterations * box_spacing, box_spacing)
        plt.xticks(ticks, checkpoint_iters)
        plt.xticks(rotation=45)

        if len(filters_used):
            plt.title(r'$\Delta$PLSC' + f"{i + 1} - {','.join(filters_used)} filters")
        else:
            plt.title(r'$\Delta$PLSC' + f'{i + 1}')

        plt.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['Obs', 'Cross-Pair'], loc='upper right')

        # Colour medians
        for plot in [bplot1, bplot2]:
            for median in plot['medians']:
                median.set_color('black')

        # show plot
        make_dir_if_not_exists(plot_dir)
        file_name = f"delta_plsc_{i + 1}_{','.join(filters_used)}"
        chart_path = f"{plot_dir}/{file_name}.png"
        # As the Delta PLSC is expensive to compute,
        # we also save the calculated results, in case we need the data later.
        with open(f"{plot_dir}/{file_name}.pkl", 'wb') as f:
            d = {'delta_corr': delta_corr, 'delta_corr_cp': delta_corr_cp}
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        plt.savefig(chart_path)
        plt.show()

    return plot_dir


def plot_plsc(plsc_corr: np.ndarray,
              plsc_corr_cp: np.ndarray,
              plsc_corr_temporal_shuffled: np.ndarray,
              checkpoint_iters: List[int],
              checkpoint_range_dir: str,
              death_filter: bool,
              mutual_vision_filter: bool,
              exclusive_vision_filter: bool
              ):
    """
    Plots the delta PLSC for each iteration.
    @param delta_corr: delta PLSC for each iteration of shape (train_iterations, episode_count, plsc_levels)
    @param delta_corr_cp: delta cross-pair PLSC for each iteration of shape (train_iterations, episode_count, plsc_levels)
    @return: dir under which plots were generated
    """
    filter_names = ['death', 'mutual vision', 'exclusive vision']
    filters_used = [n for n, f in zip(filter_names, [death_filter, mutual_vision_filter, exclusive_vision_filter]) if f]
    train_iterations = plsc_corr.shape[0]
    plot_dir = f'{checkpoint_range_dir}/plots/plsc_over_training'

    for i in range(plsc_levels):
        fig = plt.figure(figsize=(20, 10))
        plt.rcParams.update({'font.size': 20, 'text.usetex': True})

        # Creating plot
        box_count = plsc_corr.shape[0]
        box_width = 0.5
        box_spacing = 3
        bplot3 = plt.boxplot(plsc_corr_temporal_shuffled[:, :, i].T, patch_artist=True,
                             flierprops={'markerfacecolor': 'red'}, boxprops=dict(facecolor="red"),
                             positions=np.arange(box_count) * box_spacing + 1.4, widths=box_width)
        bplot2 = plt.boxplot(plsc_corr_cp[:, :, i].T, patch_artist=True, flierprops={'markerfacecolor': 'silver'},
                             boxprops=dict(facecolor="silver"),
                             positions=np.arange(box_count) * box_spacing + 0.8, widths=box_width)
        bplot1 = plt.boxplot(plsc_corr[:, :, i].T, patch_artist=True, flierprops={'markerfacecolor': 'blue'},
                             boxprops=dict(facecolor="blue"),
                             positions=np.arange(box_count) * box_spacing + 0.2, widths=box_width)

        plt.axhline(y=0, color='gray', linestyle='dashed')
        plt.ylim([0, 1])
        plt.ylabel('Correlation')
        plt.xlabel('Iterations')
        ticks = np.arange(0, train_iterations * box_spacing, box_spacing)
        plt.xticks(ticks, checkpoint_iters)
        plt.xticks(rotation=45)

        if len(filters_used):
            plt.title(r'PLSC' + f"{i + 1} - {','.join(filters_used)} filters")
        else:
            plt.title(r'PLSC' + f'{i + 1}')

        plt.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]],
                   ['Obs', 'Cross-Pair', 'Temporal Shuffled'], loc='upper right')

        # Colour medians
        for plot in [bplot1, bplot2, bplot3]:
            for median in plot['medians']:
                median.set_color('black')

        # show plot
        make_dir_if_not_exists(plot_dir)
        file_name = f"plsc_{i + 1}_{','.join(filters_used)}"
        chart_path = f"{plot_dir}/{file_name}.png"
        # As the PLSC is expensive to compute,
        # we also save the calculated results, in case we need the data later.
        with open(f"{plot_dir}/{file_name}.pkl", 'wb') as f:
            d = {'plsc_corr': plsc_corr,
                 'plsc_corr_cp': plsc_corr_cp,
                 'plsc_corr_temporal_shuffled': plsc_corr_temporal_shuffled}
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        plt.savefig(chart_path)
        plt.show()

    return plot_dir


def plot_plscs_1_2_3(plsc_corr: np.ndarray,
                     plsc_corr_cp: np.ndarray,
                     plsc_corr_temporal_shuffled: np.ndarray,
                     checkpoint_iters: List[int],
                     checkpoint_range_dir: str, ) -> str:

    plot_dir = f'{checkpoint_range_dir}/plots/plsc_over_training'
    if plsc_corr.shape[2] < 3:
        print('Cannot plot PLSC 1, 2, 3 as there are not enough PLSC dimensions')
        return plot_dir

    # Create a new figure
    fig = plt.figure(figsize=(10, 10))

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    corrs = [plsc_corr, plsc_corr_cp, plsc_corr_temporal_shuffled]
    corr_labels = ['PLSC', 'Cross-Pair PLSC', 'Temporal Shuffled PLSC']
    corr_colours = ['blue', 'black', 'red']

    # Plot the 3D line
    for corr, label, col in zip(corrs, corr_labels, corr_colours):
        x, y, z = np.median(corr, axis=1).T  # Get median plsc over episodes for each dimension
        ax.plot(x, y, z, label=label, color=col)

    # Set labels
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('PLSC 1', labelpad=15)
    ax.set_ylabel('PLSC 2', labelpad=15)
    ax.set_zlabel('PLSC 3', labelpad=15)

    ax.legend()

    embed_data_source(fig, checkpoint_range_dir)
    ax.set_title(f'PLSC Medians 1, 2, 3 for iterations {checkpoint_iters[0]}-{checkpoint_iters[-1]}')

    # Save & Display the plot
    plot_dir = f'{checkpoint_range_dir}/plots/plsc_over_training'
    make_dir_if_not_exists(plot_dir)
    file_name = f"plsc_3d"
    chart_path = f"{plot_dir}/{file_name}.png"
    plt.savefig(chart_path)
    plt.show()

    return plot_dir


if __name__ == "__main__":
    checkpoint_evaluation_path = 'eval_games/territory_open/2023-12-14_terro_best'
    all_metrics_paths = glob.glob(os.path.join(checkpoint_evaluation_path, "**", f"metrics.pkl"), recursive=True)
    checkpoint_frequency = 200
    checkpoint_start = 100
    checkpoint_end = 900
    num_temporal_shuffles = 100
    plsc_levels = 3
    death_filter = False
    mutual_vision_filter = False
    exclusive_vision_filter = False
    remove_zero_activations = False
    activation_layer = 'lstm_hidden'

    checkpoint_iters = list(range(checkpoint_start, checkpoint_end + checkpoint_frequency, checkpoint_frequency))
    # delete_disk_caches_for_function('get_plsc')

    # TODO verify death and mutual vision filters work for other envs, then add them
    plscs, plsc_cross_pairs, plsc_temporal_shuffled, delta_plscs, delta_cross_pair_plscs = get_plsc(
        all_metrics_paths=all_metrics_paths,
        base_eval_path=checkpoint_evaluation_path,
        checkpoint_iters=checkpoint_iters,
        num_temporal_shuffles=num_temporal_shuffles,
        plsc_levels=plsc_levels,
        activation_layer=activation_layer,
        remove_zero_activations=remove_zero_activations)

    plot_path = plot_plsc(plsc_corr=plscs,
                          plsc_corr_cp=plsc_cross_pairs,
                          plsc_corr_temporal_shuffled=plsc_temporal_shuffled,
                          checkpoint_iters=checkpoint_iters,
                          checkpoint_range_dir=checkpoint_evaluation_path,
                          death_filter=death_filter,
                          mutual_vision_filter=mutual_vision_filter,
                          exclusive_vision_filter=exclusive_vision_filter)

    plot_path = plot_delta_plsc(delta_corr=delta_plscs,
                                delta_corr_cp=delta_cross_pair_plscs,
                                checkpoint_iters=checkpoint_iters,
                                checkpoint_range_dir=checkpoint_evaluation_path,
                                death_filter=death_filter,
                                mutual_vision_filter=mutual_vision_filter,
                                exclusive_vision_filter=exclusive_vision_filter)

    plot_path = plot_plscs_1_2_3(plsc_corr=plscs,
                                 plsc_corr_cp=plsc_cross_pairs,
                                 plsc_corr_temporal_shuffled=plsc_temporal_shuffled,
                                 checkpoint_iters=checkpoint_iters,
                                 checkpoint_range_dir=checkpoint_evaluation_path)

    save_as_example_output(plot_path, EXAMPLE_OUTPUTS_FOLDER)
