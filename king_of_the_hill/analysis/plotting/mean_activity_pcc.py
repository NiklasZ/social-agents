import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from king_of_the_hill.wandb_api import make_if_not_exists

# Plots Mean Activity Pearson correlation

checkpoint_range_dir = 'king_of_the_hill/evaluation/run-1677652089/checkpoint-range-100-2900-2023-03-02-00-53-13'
policy_folder = 'policy0__VS__policy1'

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
agent_1_activations = {n: np.zeros((train_count, episode_count, timestep_count, activation_count)) for n in activation_layer_names}
agent_2_activations = {n: np.zeros((train_count, episode_count, timestep_count, activation_count)) for n in activation_layer_names}

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

    # Mean Activations
    a1_mean = np.mean(a1_z_scored, axis=3)  # Mean along the activation axis
    a2_mean = np.mean(a2_z_scored, axis=3)  # The resulting tensor has dim: Tr x E x Ti

    # Correlation in means over time
    for i in tqdm(range(0, a1_mean.shape[0])):
        for j in range(0, a1_mean.shape[1]):
            correlatedActivity[i, j] = scipy.stats.pearsonr(a1_mean[i, j], a2_mean[i, j])[0]

    # Correlation - Cross-pair control
    shifted_indices = np.roll(np.arange(0, episode_count), 10)
    for i in tqdm(range(0, a1_mean.shape[0])):
        for j in range(0, a1_mean.shape[1]):
            crosspairControl[i, j] = scipy.stats.pearsonr(a1_mean[i, j], a2_mean[i, shifted_indices[j]])[0]

    # Correlation - Temporal control
    shuffles = 1000
    for i in tqdm(range(0, a1_mean.shape[0])):
        for j in range(0, a1_mean.shape[1]):
            # Shuffle the data 1000 times and get the correlation each time
            shuffle_distances = np.random.randint(10, timestep_count - 10, (shuffles, 2))
            shuffled_correlations = np.zeros(shuffles)
            for idx, (s1, s2) in enumerate(shuffle_distances):
                shuffled_correlations[idx] = scipy.stats.pearsonr(np.roll(a1_mean[i, j], s1), np.roll(a2_mean[i, j], s2))[0]
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
    bplot1 = plt.boxplot(correlatedActivity.T, patch_artist=True, flierprops={'markerfacecolor': 'red'},
                         boxprops=dict(facecolor="red"),
                         positions=np.arange(box_count) * box_spacing + 0.2, widths=box_width)
    plt.ylim([-1, 1])
    plt.ylabel('PCC of Mean Activity')
    plt.xlabel('Iterations')
    ticks = np.arange(0, train_count * box_spacing, box_spacing)
    plt.xticks(ticks, checkpoint_iters)
    plt.xticks(rotation=45)
    plt.title(f'PCC of Mean Activity - {a} layer')

    plt.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]], ['Obs', 'Cross-Pair', 'Temporal'],
               loc='upper right')

    # Colour medians
    for plot in [bplot1, bplot2, bplot3]:
        for median in plot['medians']:
            median.set_color('black')

    # show plot
    chart_dir = f'{checkpoint_range_dir}/charts'
    make_if_not_exists(chart_dir)
    chart_path = f'{chart_dir}/pcc_{a}.png'
    plt.savefig(chart_path)
    plt.show()