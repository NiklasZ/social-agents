"""
Plots how much time agents spend being dead in an episode.
"""
# TODO code needs to be updated to use new evaluation metrics structure

import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


checkpoint_range_dir = 'king_of_the_hill/evaluation/run-1675134255/checkpoint-range-50-1950-2023-02-01-23-14-11'

checkpoint_frequency = 50
checkpoint_start = 50
checkpoint_end = 1950
checkpoint_iters = list(range(checkpoint_start, checkpoint_end + checkpoint_frequency, checkpoint_frequency))
train_count = len(checkpoint_iters)

example_file_path = f'{checkpoint_range_dir}/checkpoint-{checkpoint_start}/metrics.pkl'
with open(example_file_path, 'rb') as fp:
    example_pkl_file = pickle.load(fp)

episode_count = len(example_pkl_file['activations'])

example_activations = next(iter(example_pkl_file['activations'][0].values()))
timestep_count, activation_count = next(iter(example_activations.values())).shape

# Tensors of following dimensions: training iteration (Tr) x episode (E) x timestep (Ti) x activations (A)
# agent_1_activations = np.zeros((train_count, episode_count, timestep_count, activation_count))
# agent_2_activations = np.zeros_like(agent_1_activations)
player_0_death_rates = []
player_1_death_rates = []

for i, c in tqdm(enumerate(checkpoint_iters)):
    file_path = f'{checkpoint_range_dir}/checkpoint-{c}/metrics.pkl'
    with open(file_path, 'rb') as fp:
        pkl_file = pickle.load(fp)
        ep_length = len(pkl_file['deaths'][0])
        p0_rates = []
        p1_rates = []
        for j in range(0, episode_count):
            player_0_dead_rate = len([i for i, p in enumerate(pkl_file['deaths'][j]) if p['player_0']])/ep_length
            player_1_dead_rate = len([i for i, p in enumerate(pkl_file['deaths'][j]) if p['player_1']])/ep_length

            p0_rates.append(player_0_dead_rate)
            p1_rates.append(player_1_dead_rate)

        player_0_death_rates.append(np.mean(p0_rates))
        player_1_death_rates.append(np.mean(p1_rates))

fig = plt.figure(1)

plt.plot(checkpoint_iters, player_0_death_rates, color='red')
plt.plot(checkpoint_iters, player_1_death_rates, color='blue')
plt.ylim([0,1])
plt.ylabel('% of time spent dead')
plt.xlabel('Training Iteration')
plt.title('Agent time spent dead')

plt.legend(['Red Player', 'Blue Player'], loc='upper right')
out_path = f'{checkpoint_range_dir}/charts/death_rates.png'
fig.savefig(out_path, bbox_inches='tight')
plt.show()
