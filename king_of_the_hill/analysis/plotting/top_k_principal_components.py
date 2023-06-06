from datetime import datetime
from typing import Dict
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from king_of_the_hill.wandb_api import make_dir_if_not_exists

# Plots Activity PCA top k Principal Components' correlation

checkpoint_range_dir = 'king_of_the_hill/evaluation/run-1675134255/checkpoint-range-50-1950-2023-02-01-23-14-11'
now = datetime.now()

num_components = 20 # (k)
checkpoint_frequency = 100
checkpoint_start = 50
checkpoint_end = 1950
checkpoint_iters = list(range(checkpoint_start, checkpoint_end + checkpoint_frequency, checkpoint_frequency))
train_count = len(checkpoint_iters)

example_file_path = f'{checkpoint_range_dir}/checkpoint-{checkpoint_start}/metrics.pkl'
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
    file_path = f'{checkpoint_range_dir}/checkpoint-{c}/metrics.pkl'
    with open(file_path, 'rb') as fp:
        pkl_file = pickle.load(fp)

        for j in range(0, episode_count):
            for a in activation_layer_names:
                agent_1_activations[a][i, j] = pkl_file['activations'][j]['player_0'][a]
                agent_2_activations[a][i, j] = pkl_file['activations'][j]['player_1'][a]

for a in activation_layer_names:
    # Calculate mean activity
    correlatedActivity = np.zeros((train_count, episode_count, num_components, num_components))

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
    a1_pck_components = np.zeros((train_count, episode_count, num_components, timestep_count), dtype=complex)
    a2_pck_components = np.zeros((train_count, episode_count, num_components, timestep_count), dtype=complex)

    for i in tqdm(range(train_count)):
        for j in range(episode_count):
            a1_eig_values, a1_eig_vectors = np.linalg.eig(a1_cov[i, j])
            a2_eig_values, a2_eig_vectors = np.linalg.eig(a2_cov[i, j])
            a1_e_indices = np.argsort(a1_eig_values)[::-1]
            a2_e_indices = np.argsort(a2_eig_values)[::-1]

            a1_pck_components[i, j] = a1_eig_vectors[a1_e_indices[:num_components]]
            a2_pck_components[i, j] = a2_eig_vectors[a2_e_indices[:num_components]]

    # Drop sign
    # a1_pck_components = np.abs(a1_pck_components)
    # a2_pck_components = np.abs(a2_pck_components)

    # Correlation in means over time
    for i in tqdm(range(train_count)):
        for j in range(episode_count):
            for k1 in range(num_components):
                for k2 in range(num_components):
                    correlatedActivity[i, j, k1, k2] = scipy.stats.pearsonr(a1_pck_components[i, j, k1], a2_pck_components[i, j, k2])[0]

    fig = plt.figure(figsize=(20, 14))
    plt.rcParams.update({'font.size': 16})

    # Creating plot
    plt.hist(correlatedActivity.flatten(), bins=1000)
    plt.ylabel('Correlation Counts')
    plt.xlabel('PCC')
    plt.title(f"Correlation Frequencies for top k={num_components} Principal Components (layer: {a})")

    # show plot
    chart_dir = f'{checkpoint_range_dir}/charts'
    make_dir_if_not_exists(chart_dir)
    chart_path = f"{chart_dir}/pck_{num_components}_{a}.png"

    plt.savefig(chart_path)
    plt.show()
