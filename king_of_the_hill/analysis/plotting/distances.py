import json
import matplotlib.pyplot as plt
import numpy as np

# TODO: write a proper argparse
eval_path = 'king_of_the_hill/evaluation/run-1663376683/checkpoint-1300-2022-12-05-14-36-11'
metrics_path = f'{eval_path}/metrics.json'
ep_id = 0


def plot_players_distances(m: dict):
    if len(m['zaps'][ep_id]) != 2:
        raise NotImplementedError('Function is only designed for 2 agents')

    # Get all non-zero distances as zero indicates one of the players is dead
    player_0_to_1_distance = np.array([(i, d['player_0']['player_1'])
                                       for i, d in enumerate(m['distances'][ep_id])])

    timesteps = player_0_to_1_distance[:, 0]
    distances = player_0_to_1_distance[:, 1]

    fig = plt.figure(1, figsize=(8, 2))
    plt.plot(timesteps, distances)
    plt.xlabel('timesteps')
    plt.ylabel('Euclidean distance')
    plt.title('Distance between agents')

    out_path = f'{eval_path}/ep{ep_id}_distances'
    fig.savefig(out_path, bbox_inches='tight')
    return


if __name__ == "__main__":
    with open(metrics_path) as metrics_str:
        metrics = json.load(metrics_str)
    plot_players_distances(metrics)
