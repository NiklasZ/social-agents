import glob
import json
import os
from typing import List, Callable, Any, Optional

from numpy.random import randint
from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import zscore
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import explained_variance_score
from torch.nn.functional import one_hot
import inspect

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


def get_relative_player_positions(metrics: dict, ep_id: int, player_id: str, other_player_id: str):
    if len(metrics['positions'][ep_id][0]) != 2:
        raise NotImplementedError('Function is only designed for 2 agents')
    return np.array([relative_position(p[player_id], p[other_player_id], o[player_id])
                     for p, o in zip(metrics['positions'][ep_id], metrics['orientations'][ep_id])])


# {'center_x': hill_center_x, 'center_y': hill_center_y}

def get_relative_hill_positions(metrics: dict, ep_id: int, player_id: str):
    hill_position = (metrics['other']['hill']['center_x'], metrics['other']['hill']['center_y'])
    return np.array([relative_position(p[player_id], hill_position, o[player_id])
                     for p, o in zip(metrics['positions'][ep_id], metrics['orientations'][ep_id])])


def get_action_softmax(logit_array: np.ndarray, convert_to_one_hot: bool) -> np.ndarray:
    # Assume axis 0 is timesteps, axis 1 is actions
    soft_maxed = softmax(logit_array, axis=1)

    # If toggled, yields one-hot vectors where 1 is at the most probable action
    if convert_to_one_hot:
        most_probable_actions = np.argmax(soft_maxed, axis=1)
        one_hotted = np.zeros_like(logit_array)
        one_hotted[range(logit_array.shape[0]), most_probable_actions] = 1
        return one_hotted
    else:
        return soft_maxed


def action_to_one_hot(actions: List[dict], player_ids: List[str]) -> dict:
    one_hot_actions = {}

    for p in player_ids:
        # create a tensor with the data
        data_tensor = torch.tensor([a[p] for a in actions])
        # one-hot encode the data
        # FIXME this number of actions is a value that should be given by the environment, not a constant
        one_hot_actions[p] = one_hot(data_tensor, num_classes=9).numpy()
    return one_hot_actions


# Performs SVD on the covariance of 2 normalised matrices.
def get_covariance_svd(A: np.ndarray, B: np.ndarray):
    # Z-score
    A_z, B_z = zscore(np.nan_to_num(A), ddof=1), zscore(np.nan_to_num(B), ddof=1)
    # Get covariance
    cov_mat = np.nan_to_num(np.dot(A_z.T, B_z) / A_z.shape[0])
    # SVD on covariance matrix
    U, S, V_T = svd(cov_mat)
    return U, V_T.T, S


# Performs PLSC to get the shared neural activity. We then perform SVD to get the
# highest variance components (aka saliences) and use them to score the original activities.
def score_shared_activity_via_plsc(activations: List[np.ndarray], plsc_value=1) -> (np.ndarray, np.ndarray):
    if len(activations) != 2:
        raise NotImplementedError('Function is only designed for 2 agents')

    plsc_index = plsc_value - 1
    U, V, _ = get_covariance_svd(activations[0], activations[1])
    u_score = activations[0].dot(U[:, plsc_index])
    v_score = activations[1].dot(V[:, plsc_index])
    return u_score, v_score


def find_metrics_paths(run_id: str, run_path: str, policy_ids: List[str], checkpoint_range: List[int]) -> List[str]:
    metrics_paths = []
    for c in checkpoint_range:
        metrics_folder = '__VS__'.join([f'{run_id}.checkpoint-{c}.{p}' for p in policy_ids])
        matching_paths = glob.glob(os.path.join(run_path, "**", metrics_folder, "metrics.pkl"), recursive=True)
        if len(matching_paths) == 0:
            raise Exception(f'Could not find metrics file in path {run_path}..{metrics_folder}')
        elif len(matching_paths) > 1:
            raise Exception(
                f'Found more than one metrics file matching path {run_path}..{metrics_folder}.\n Found paths: {matching_paths}')
        else:
            metrics_paths.append(matching_paths[0])
    return metrics_paths


def get_kwargs_matching(function: Callable[..., Any], kwargs: dict) -> dict:
    """
    Filters a dictionary of keyword arguments to only include those that match the given function's signature.

    Args:
        function: The function whose signature to match.
        kwargs: The dictionary of keyword arguments to filter.

    Returns:
        A new dict
    """
    sig = inspect.signature(function)

    # Keep only the arguments that exist in the function's signature
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return filtered_kwargs


def get_var_explained_via_plsr(X: np.ndarray, y: np.ndarray, n_components: Optional[int] = None) -> float:
    if n_components is None:
        n_components = min(np.linalg.matrix_rank(X), np.linalg.matrix_rank(y))
    pls = PLSRegression(n_components=n_components, max_iter=1000)
    pls.fit(X, y)
    pred_y = pls.predict(X)
    return explained_variance_score(y, pred_y)


# Shifts array randomly by a range from the minimum to max length of the first dimension of the array.
def random_shift(arr: np.ndarray, min_shift=10) -> np.ndarray:
    rand_high = arr.shape[0]
    return np.roll(arr, randint(min_shift, rand_high))


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
