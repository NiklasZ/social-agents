"""
This creates a plot the compares mean max action probability vs mean returns for each policy in a cross-play evaluation.
In practice, it shows whether there is any relationship between how deterministic a policy is vs. how well it performs.
"""
# TODO code needs to be updated to use new evaluation metrics structure

import glob
import os
import pickle
from typing import List
import numpy as np
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from matplotlib import pyplot as plt
from tqdm import tqdm

from analysis.cross_play_returns import get_cross_play_returns
from analysis.helpers import get_action_softmax
from utils.wandb_api import make_dir_if_not_exists

player_colours = {'player_0': 'red', 'player_1': 'blue'}


@cache_to_disk(31)  # Cache function output for input for 31 days
def get_max_action_probabilities(all_metrics_paths: List[str]) -> dict:
    with open(all_metrics_paths[0], 'rb') as fp:
        example_pkl_file = pickle.load(fp)

    num_episodes = len(example_pkl_file['rewards'])

    # {policy id: np.array(max of action probs per timestep)}
    max_action_probabilities = {}

    for metrics_path in tqdm(all_metrics_paths):
        with open(metrics_path, 'rb') as fp:
            metrics = pickle.load(fp)

        agent_infos = metrics['other']['agent_info']
        # Get maximum action probability for each timestep
        for a_i in agent_infos:
            all_actions = np.concatenate(
                [metrics['activations'][ep_idx][a_i['player']]['action_logits'] for ep_idx in range(num_episodes)])
            max_action_probs = np.max(get_action_softmax(all_actions, False), axis=1)
            dict_key = (a_i['rllib_run_name'], a_i['checkpoint_name'], a_i['policy_id'], a_i['player'])
            if dict_key in max_action_probabilities:
                max_action_probabilities[dict_key] = np.concatenate(
                    [max_action_probabilities[dict_key], max_action_probs])
            else:
                max_action_probabilities[dict_key] = max_action_probs

    return max_action_probabilities


def plot_action_distribution(run_path: str, max_action_probabilities: dict, bins=100):
    for policy_key, actions in max_action_probabilities.items():
        fig = plt.figure(figsize=(10, 6))
        plt.hist(actions, bins=bins, color=player_colours[policy_key[3]], edgecolor='black')
        plt.ylabel('Counts')
        plt.xlabel('Maximum Probability of Actions')
        plt.title(f"Probabilities of most likely actions\n{'.'.join(policy_key)}")
        plt.xlim([0, 1])

        chart_dir = f'{run_path}/plots/max_action_probability_distribution'
        make_dir_if_not_exists(chart_dir)
        chart_path = f"{chart_dir}/{'_'.join(policy_key)}.png"
        plt.savefig(chart_path)

        plt.show()


def plot_mean_max_action_probability_vs_reward(run_path: str, max_action_probabilities: dict,
                                               player_returns_mean: dict):
    mean_action_probs = []
    mean_returns = []

    for dict_key in max_action_probabilities:
        mean_action_probs.append(np.mean(max_action_probabilities[dict_key]))

        # FIXME refactor the mean_return to use the same key structure
        player = dict_key[3]
        mean_returns_key = '.'.join(dict_key[:3])
        mean_returns.append(np.mean(list(player_returns_mean[player][mean_returns_key].values())))

    fig = plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 16})
    plt.scatter(mean_action_probs, mean_returns)

    plt.xlabel('mean max action probability')
    plt.title(f'Mean of Maximally Probable Actions vs. Mean Returns')
    plt.ylabel('Mean Episode Reward')
    z = np.polyfit(mean_action_probs, mean_returns, 1)
    p = np.poly1d(z)
    plt.plot(mean_action_probs, p(mean_action_probs), "r--")

    chart_dir = f'{run_path}/plots/max_action_probability_vs_mean_reward'
    make_dir_if_not_exists(chart_dir)
    chart_path = f"{chart_dir}/{'plot'}.png"
    plt.savefig(chart_path)
    plt.show()


if __name__ == "__main__":
    cross_play_path = 'king_of_the_hill/evaluation/koth-multi-train_cross_4x'
    # Enable to clear function caching
    # delete_disk_caches_for_function('get_max_action_probabilities')
    all_metrics_paths = glob.glob(os.path.join(cross_play_path, "**", f"metrics.pkl"), recursive=True)
    max_action_probabilities = get_max_action_probabilities(all_metrics_paths)
    player_returns_mean, player_returns_std, player_policies = get_cross_play_returns(all_metrics_paths)
    plot_mean_max_action_probability_vs_reward(run_path=cross_play_path,
                                               max_action_probabilities=max_action_probabilities,
                                               player_returns_mean=player_returns_mean)
    plot_action_distribution(run_path=cross_play_path, max_action_probabilities=max_action_probabilities)
