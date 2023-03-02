from datetime import datetime
from typing import Dict
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from king_of_the_hill.wandb_api import make_if_not_exists

# Plots Activity PCA Principal Component 1 correlation

checkpoint_range_dir = 'king_of_the_hill/evaluation/run-1677652089/checkpoint-range-100-2900-2023-03-02-00-53-13'
policy_folder = 'policy0__VS__policy1'
now = datetime.now()
# timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

checkpoint_frequency = 200
checkpoint_start = 100
checkpoint_end = 2900
checkpoint_iters = list(range(checkpoint_start, checkpoint_end + checkpoint_frequency, checkpoint_frequency))
train_count = len(checkpoint_iters)

example_file_path = f'{checkpoint_range_dir}/checkpoint-{checkpoint_start}/{policy_folder}/metrics.pkl'
with open(example_file_path, 'rb') as fp:
    example_pkl_file = pickle.load(fp)

episode_count = len(example_pkl_file['activations'])

example_activations = next(iter(example_pkl_file['activations'][0].values()))
activation_layer_names = list(example_activations.keys())
timestep_count, activation_count = next(iter(example_activations.values())).shape

# Tensors of following dimensions: training iteration (Tr) x episode (E) x timestep (Ti) x activations (A)
agent_1_activations = {n: np.zeros((train_count, episode_count, timestep_count, activation_count)) for n in
                       activation_layer_names}
agent_2_activations = {n: np.zeros((train_count, episode_count, timestep_count, activation_count)) for n in
                       activation_layer_names}

for i, c in tqdm(enumerate(checkpoint_iters)):
    file_path = f'{checkpoint_range_dir}/checkpoint-{c}/{policy_folder}/metrics.pkl'
    with open(file_path, 'rb') as fp:
        pkl_file = pickle.load(fp)

        for j in range(0, episode_count):
            for a in activation_layer_names:
                agent_1_activations[a][i, j] = pkl_file['activations'][j]['player_0'][a]
                agent_2_activations[a][i, j] = pkl_file['activations'][j]['player_1'][a]

for a in activation_layer_names:
    # Calculate mean activity
    correlatedActivity = np.zeros((train_count, episode_count))
    temporalControl = np.zeros(correlatedActivity.shape)
    crosspairControl = np.zeros(correlatedActivity.shape)

    a1_z_scored = scipy.stats.zscore(agent_1_activations[a], ddof=1, axis=2)  # z-score across the timestep dimension
    a2_z_scored = scipy.stats.zscore(agent_2_activations[a], ddof=1, axis=2)  # z-score across the timestep dimension

    # It seems there is some dirty data in the activations (0 activations for a column). When z-scoring on this column
    # The 0-division makes it NaN. We will replace NaN with 0
    a1_z_scored = np.nan_to_num(a1_z_scored)
    a2_z_scored = np.nan_to_num(a2_z_scored)

    a1_cov = np.zeros((train_count, episode_count, timestep_count, timestep_count))
    a2_cov = np.zeros((train_count, episode_count, timestep_count, timestep_count))

    # Covariance
    for i in tqdm(range(train_count)):
        for j in range(episode_count):
            a1_cov[i, j] = np.cov(a1_z_scored[i, j])
            a2_cov[i, j] = np.cov(a2_z_scored[i, j])

    # Eigendecomposition
    a1_pc1_components = np.zeros((train_count, episode_count, timestep_count), dtype=complex)
    a2_pc1_components = np.zeros((train_count, episode_count, timestep_count), dtype=complex)

    for i in tqdm(range(train_count)):
        for j in range(episode_count):
            a1_eig_values, a1_eig_vectors = np.linalg.eig(a1_cov[i, j])
            a2_eig_values, a2_eig_vectors = np.linalg.eig(a2_cov[i, j])
            a1_e_indices = np.argsort(a1_eig_values)[::-1]
            a2_e_indices = np.argsort(a2_eig_values)[::-1]

            a1_pc1_components[i, j] = a1_eig_vectors[a1_e_indices[0]]
            a2_pc1_components[i, j] = a2_eig_vectors[a2_e_indices[0]]

    # Drop sign
    # a1_pc1_components = np.abs(a1_pc1_components)
    # a2_pc1_components = np.abs(a2_pc1_components)

    # Correlation in means over time
    for i in tqdm(range(train_count)):
        for j in range(episode_count):
            correlatedActivity[i, j] = scipy.stats.pearsonr(a1_pc1_components[i, j], a2_pc1_components[i, j])[0]

    # Correlation - Cross-pair control
    shifted_indices = np.roll(np.arange(0, episode_count), 10)
    for i in tqdm(range(train_count)):
        for j in range(episode_count):
            crosspairControl[i, j] = \
            scipy.stats.pearsonr(a1_pc1_components[i, j], a2_pc1_components[i, shifted_indices[j]])[0]

    shuffles = 1000
    for i in tqdm(range(train_count)):
        for j in range(episode_count):
            # Shuffle the data 1000 times and get the correlation each time
            shuffle_distances = np.random.randint(10, timestep_count - 10, (shuffles, 2))
            shuffled_correlations = np.zeros(shuffles)
            for idx, (s1, s2) in enumerate(shuffle_distances):
                shuffled_correlations[idx] = \
                scipy.stats.pearsonr(np.roll(a1_pc1_components[i, j], s1), np.roll(a2_pc1_components[i, j], s2))[0]
            temporalControl[i, j] = np.median(shuffled_correlations)

    fig = plt.figure(figsize=(20, 14))
    plt.rcParams.update({'font.size': 16})

    # Creating plot
    box_count = crosspairControl.shape[0]
    box_width = 0.5
    box_spacing = 3
    bplot3 = plt.boxplot(temporalControl.T, patch_artist=True, flierprops={'markerfacecolor': 'grey'},
                         boxprops=dict(facecolor="grey"),
                         positions=np.arange(box_count) * box_spacing + 1.6, widths=box_width)
    bplot2 = plt.boxplot(crosspairControl.T, patch_artist=True, flierprops={'markerfacecolor': 'silver'},
                         boxprops=dict(facecolor="silver"),
                         positions=np.arange(box_count) * box_spacing + 0.8, widths=box_width)
    bplot1 = plt.boxplot(correlatedActivity.T, patch_artist=True, flierprops={'markerfacecolor': 'purple'},
                         boxprops=dict(facecolor="red"),
                         positions=np.arange(box_count) * box_spacing + 0.2, widths=box_width)
    plt.ylim([-1, 1])
    plt.ylabel('PC1 Correlation')
    plt.xlabel('Iterations')
    ticks = np.arange(0, train_count * box_spacing, box_spacing)
    plt.xticks(ticks, checkpoint_iters)
    plt.xticks(rotation=45)
    plt.title(f"PC1 Correlation of Activity - {a} layer")

    plt.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]], ['Obs', 'Cross-Pair', 'Temporal'],
               loc='upper right')

    # Colour medians
    for plot in [bplot1, bplot2, bplot3]:
        for median in plot['medians']:
            median.set_color('black')

    # show plot
    chart_dir = f'{checkpoint_range_dir}/charts'
    make_if_not_exists(chart_dir)
    # chart_path = f"{chart_dir}/{timestamp}_{'_'.join(filters_used)}_pcc_{a}.png"
    chart_path = f"{chart_dir}/pc1_{a}.png"

    plt.savefig(chart_path)
    plt.show()
