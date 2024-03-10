"""
This module displays all activations over the course of an episode for a pair of agents.
If provided, the module also displays events for that particular environment.
"""

import glob
import os
import pickle
from typing import List, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt

from analysis.helpers import EXAMPLE_OUTPUTS_FOLDER, save_as_example_output, embed_data_source
from utils.wandb_api import make_dir_if_not_exists


def get_activations(metrics_path: str, episode_idx: int, activation_layer: str) -> Tuple[dict, dict]:
    with open(metrics_path, 'rb') as fp:
        metrics_file = pickle.load(fp)

    all_acts = metrics_file['general_metrics']['activations']
    ep_activations = {p: all_acts[episode_idx][p][activation_layer] for idx, p in
                      enumerate(all_acts[episode_idx].keys())}
    agent_info = {info['player']: info for info in metrics_file['general_metrics']['other']['agent_info']}
    return ep_activations, agent_info


def get_events(metrics_path: str, ep_idx: int) -> Tuple[List[List[int]], List[str]]:
    with open(metrics_path, 'rb') as fp:
        m = pickle.load(fp)

    if m['general_metrics']['other']['environment'] != 'king_of_the_hill':
        raise NotImplementedError('This function is only implemented for the king of the hill environment')

    player_0_zaps_1 = [i for i, p in enumerate(m['env_metrics']['zaps'][ep_idx]) if len(p['player_0'])]
    player_1_zaps_0 = [i for i, p in enumerate(m['env_metrics']['zaps'][ep_idx]) if len(p['player_1'])]
    player_0_paints = [i for i, p in enumerate(m['general_metrics']['actions'][ep_idx]) if p['player_0'] in [7, 8]]
    player_1_paints = [i for i, p in enumerate(m['general_metrics']['actions'][ep_idx]) if p['player_1'] in [7, 8]]
    player_0_sees_1 = [i for i, p in enumerate(m['env_metrics']['visibilities'][ep_idx]) if len(p['player_0'])]
    player_1_sees_0 = [i for i, p in enumerate(m['env_metrics']['visibilities'][ep_idx]) if len(p['player_1'])]
    player_0_dead = [i for i, p in enumerate(m['general_metrics']['deaths'][ep_idx]) if p['player_0']]
    player_1_dead = [i for i, p in enumerate(m['general_metrics']['deaths'][ep_idx]) if p['player_1']]
    player_0_captures = [i for i, p in enumerate(m['env_metrics']['captures'][ep_idx]) if p['player_0']]
    player_1_captures = [i for i, p in enumerate(m['env_metrics']['captures'][ep_idx]) if p['player_1']]

    # TODO assign specific colors to each event
    data = [player_1_zaps_0, player_0_zaps_1, player_1_paints, player_0_paints, player_1_sees_0, player_0_sees_1,
            player_1_captures, player_0_captures, player_1_dead, player_0_dead]
    labels = ['blue player zaps red', 'red player zaps blue', 'blue paints', 'red paints', 'blue sees red',
              'red sees blue', 'blue captures hill', 'red captures hill', 'blue is dead', 'red is dead']

    return data, labels


def plot_events(data: List[List[int]], labels: List[str], ax: plt.Axes):
    # set different colors for each set of positions
    colors1 = ['C{}'.format(i) for i in range(len(data))]
    # set different line properties for each set of positions
    # note that some overlap
    lineoffsets1 = np.array([0, -1, -2, -3, -4, -5, -6, -7, -8, -9])

    ax.eventplot(data, colors=colors1, lineoffsets=lineoffsets1)
    ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('timesteps')
    ax.axes.get_yaxis().set_visible(False)


def plot_activations(activations: Dict[str, np.ndarray],
                     agent_info: dict,
                     player: str,
                     ax: plt.Axes,
                     fig: plt.Figure):

    data = activations[player].T
    policy_id = agent_info[player]["policy_id"]
    cax = ax.imshow(data, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=ax, pad=-0.064)
    ax.set_title(f'Activations for {policy_id}')
    ax.set_ylabel('Neuron Index')


def plot_data(activations: Dict[str, np.ndarray],
              agent_info: dict,
              data: List[List[int]],
              labels: List[str],
              run_path: str,
              episode_idx: int
              ) -> str:
    plot_dir = f'{run_path}/plots/activations_over_episode/episode_{episode_idx}'

    # Iterate over each player's view
    for player in activations:
        run_id = agent_info[player]['run_id']
        policy_id = agent_info[player]['policy_id']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True, constrained_layout=True)
        plot_activations(activations=activations, agent_info=agent_info, ax=ax1, fig=fig, player=player)
        plot_events(data=data, labels=labels, ax=ax2)

        # Display and save the plot
        embed_data_source(fig, run_path)
        make_dir_if_not_exists(plot_dir)
        plt.savefig(f'{plot_dir}/{run_id}_{policy_id}.png')
        plt.show()

    return plot_dir


if __name__ == "__main__":
    run_path = 'eval_games/king_of_the_hill/2024-02-29-16-09-28'
    activation_layer = 'lstm_hidden'
    episode_idx = 2
    all_metrics_paths = glob.glob(os.path.join(run_path, "**", f"metrics.pkl"), recursive=True)
    assert len(
        all_metrics_paths) == 1, f'This method is only designed for 1 metrics file. Found {len(all_metrics_paths)}'
    activations, agent_info = get_activations(metrics_path=all_metrics_paths[0],
                                              episode_idx=episode_idx,
                                              activation_layer=activation_layer)
    event_data, event_labels = get_events(metrics_path=all_metrics_paths[0], ep_idx=episode_idx)

    plot_path = plot_data(activations=activations,
                          agent_info=agent_info,
                          data=event_data,
                          labels=event_labels,
                          run_path=run_path,
                          episode_idx=episode_idx)

    save_as_example_output(plot_path, EXAMPLE_OUTPUTS_FOLDER)
