import json
import matplotlib.pyplot as plt
import numpy as np

# Plots distance between players

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


# Note this conversion applies transformations to the coordinates based on the orientation the player is facing.
# For details on how these were calculated see king_of_the_hill/docs/relative_position_notes.pdf
position_mappers = [
    lambda x, y: (x, -y),  # orientation 0: faces north
    lambda x, y: (y, x),  # orientation 1: faces east
    lambda x, y: (-x, y),  # orientation 2: faces south
    lambda x, y: (-y, -x)  # orientation 3: faces west
]


def relative_position(player_pos: [float, float], other_pos: [float, float], orientation: int) -> np.ndarray:
    if orientation < 0 or orientation > len(position_mappers):
        raise Exception(f'Unknown orientation {orientation}. Expected it to be in range of [0,{len(position_mappers)}]')
    mapper = position_mappers[orientation]
    return np.array(mapper(*other_pos)) - np.array(mapper(*player_pos))


def get_relative_positions(metrics: dict, ep_id: int, player_id: str, other_player_id: str):
    if len(metrics['positions'][ep_id][0]) != 2:
        raise NotImplementedError('Function is only designed for 2 agents')
    return np.array([relative_position(p[player_id], p[other_player_id], o[player_id])
                    for p, o in zip(metrics['positions'][ep_id], metrics['orientations'][ep_id])])


if __name__ == "__main__":
    # dummy_metrics = {'positions': [[{'player_0': [16, 3], 'player_1': [14, 4]},
    #                                 {'player_0': [16, 3], 'player_1': [14, 4]},
    #                                 {'player_0': [16, 3], 'player_1': [14, 4]},
    #                                 {'player_0': [16, 3], 'player_1': [14, 4]}]],
    #                  'orientations': [[{'player_0': 0}, {'player_0': 1}, {'player_0': 2}, {'player_0': 3}]]}
    # out = get_relative_positions(dummy_metrics, 0, 'player_0', 'player_1')
    # print(out)
    with open(metrics_path) as metrics_str:
        metrics = json.load(metrics_str)
    plot_players_distances(metrics)
