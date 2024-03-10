import copy
from typing import List
import numpy as np
# TODO code needs to be updated to use new evaluation metrics structure


def filter_by_deaths(metrics: dict, player_ids: List[str]) -> dict:
    """
    Only gets activations for timesteps where all players are alive
    @param metrics: metrics dict
    @param player_ids: ids to consider
    @return:
    """
    num_episodes = len(metrics['rewards'])
    output_metrics = copy.deepcopy(metrics)
    for i in range(num_episodes):
        alive_mask = [not any([k[p] for p in player_ids]) for k in metrics['deaths'][i]]
        alive_indices = np.where(alive_mask)

        for prop in [k for k in metrics.keys() if k not in ['activations', 'other']]:
            output_metrics[prop][i] = list(np.array(metrics[prop][i])[alive_indices])

        for p in player_ids:
            for layer in metrics['activations'][i][p]:
                output_metrics['activations'][i][p][layer] = metrics['activations'][i][p][layer][alive_indices]

        output_metrics['other'] = metrics['other']

    return output_metrics


# TODO refactor to use above function except for vision mask line
def filter_by_mutual_vision(metrics: dict, player_ids: List[str]) -> dict:
    """
    Only gets activations for timesteps where players can see each other
    @param metrics: metrics dict
    @param player_ids: ids to consider
    @return:
    """
    num_episodes = len(metrics['rewards'])
    output_metrics = copy.deepcopy(metrics)
    for i in range(num_episodes):
        vision_mask = [all([len(k[p]) for p in player_ids]) for k in metrics['visibilities'][i]]
        vision_indices = np.where(vision_mask)

        for prop in [k for k in metrics.keys() if k not in ['activations', 'other']]:
            output_metrics[prop][i] = list(np.array(metrics[prop][i])[vision_indices])

        for p in player_ids:
            for layer in metrics['activations'][i][p]:
                output_metrics['activations'][i][p][layer] = metrics['activations'][i][p][layer][vision_indices]

        output_metrics['other'] = metrics['other']

    return output_metrics
