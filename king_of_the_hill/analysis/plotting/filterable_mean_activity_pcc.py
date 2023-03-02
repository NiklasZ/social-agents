from datetime import datetime
from typing import Dict
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from king_of_the_hill.wandb_api import make_if_not_exists

# Plots Mean Activity Pearson correlation allowing to filter out timesteps based on some criteria

checkpoint_range_dir = 'king_of_the_hill/evaluation/run-1677652089/checkpoint-range-100-2900-2023-03-02-00-53-13'
policy_folder = 'policy0__VS__policy1'

now = datetime.now()
# timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
death_filter = True
mutual_vision_filter = False
filter_names = ['death', 'vision']

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
activation_masks = np.ones((train_count, episode_count, timestep_count), dtype=bool)

for i, c in tqdm(enumerate(checkpoint_iters)):
    file_path = f'{checkpoint_range_dir}/checkpoint-{c}/{policy_folder}/metrics.pkl'
    with open(file_path, 'rb') as fp:
        pkl_file = pickle.load(fp)

        for j in range(0, episode_count):
            for a in activation_layer_names:
                agent_1_activations[a][i, j] = pkl_file['activations'][j]['player_0'][a]
                agent_2_activations[a][i, j] = pkl_file['activations'][j]['player_1'][a]

                # Mask every timestep where at least one player is dead.
                if death_filter:
                    death_mask = [not (k['player_0'] or k['player_1']) for k in pkl_file['deaths'][j]]
                    activation_masks[i, j] = np.logical_and(activation_masks[i, j], death_mask)

                # Masks every timestep where players can't see each other.
                if mutual_vision_filter:
                    vision_mask = [len(k['player_0']) and len(k['player_1']) for k in pkl_file['visibilities'][j]]
                    activation_masks[i, j] = np.logical_and(activation_masks[i, j], vision_mask)


def get_means(i: int,
              j: int,
              agent_1_activations: Dict[str, np.ndarray],
              agent_2_activations: Dict[str, np.ndarray],
              activation_masks: np.ndarray
              ):
    masked_agent_1_activations = agent_1_activations[a][i, j, activation_masks[i, j]]
    masked_agent_2_activations = agent_2_activations[a][i, j, activation_masks[i, j]]

    # Z-score along timestep axis
    a1_z_scored = scipy.stats.zscore(masked_agent_1_activations, ddof=1, axis=0)
    a2_z_scored = scipy.stats.zscore(masked_agent_2_activations, ddof=1, axis=0)
    a1_z_scored = np.nan_to_num(a1_z_scored)
    a2_z_scored = np.nan_to_num(a2_z_scored)

    # Mean along the activation axis
    a1_mean = np.mean(a1_z_scored, axis=1)
    a2_mean = np.mean(a2_z_scored, axis=1)

    return a1_mean, a2_mean


for a in activation_layer_names:
    # Calculate mean activity
    correlatedActivity = np.zeros((train_count, episode_count))
    temporalControl = np.zeros(correlatedActivity.shape)
    crosspairControl = np.zeros(correlatedActivity.shape)

    for i in tqdm(range(0, train_count)):
        for j in range(0, episode_count):

            a1_mean, a2_mean = get_means(i, j, agent_1_activations, agent_2_activations, activation_masks)

            # Get pearson correlation coefficient
            correlatedActivity[i, j] = scipy.stats.pearsonr(a1_mean, a2_mean)[0]

            # Cross-pair correlation control
            shifted_indices = np.roll(np.arange(0, episode_count), 10)
            _, a2_shifted_mean = get_means(i, shifted_indices[j], agent_1_activations,
                                           agent_2_activations, activation_masks)
            # As we may mask a different number of timesteps for different episodes, the resulting sizes are different.
            # To calculate a correlation, we need to trim the longer episode to the length of the smaller.
            smallest_size = np.min([a1_mean.size, a2_shifted_mean.size])
            crosspairControl[i, j] = scipy.stats.pearsonr(a1_mean[:smallest_size], a2_shifted_mean[:smallest_size])[0]

            # Temporal correlation control
            shuffles = a1_mean.size
            shuffle_distances = np.random.randint(10, timestep_count - 10, (shuffles, 2))
            shuffled_correlations = np.zeros(shuffles)
            for idx, (s1, s2) in enumerate(shuffle_distances):
                shuffled_correlations[idx] = \
                    scipy.stats.pearsonr(np.roll(a1_mean, s1), np.roll(a2_mean, s2))[0]
            temporalControl[i, j] = np.median(shuffled_correlations)

    fig = plt.figure(figsize=(20, 14))
    plt.rcParams.update({'font.size': 16})
    filters_used = [n for n, f in zip(filter_names, [death_filter, mutual_vision_filter]) if f]

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
    plt.title(f"PCC of Mean Activity - {a} layer{' - filters: ' if filters_used else ''}{','.join(filters_used)}")

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
    chart_path = f"{chart_dir}/{'_'.join(filters_used)}_pcc_{a}.png"

    plt.savefig(chart_path)
    plt.show()
