import numpy as np


# TODO doc
def get_distances(infos: dict, deaths: dict):
    player_positions = {p: infos[p]['POSITION'] for p in infos}
    distances = {}

    for p_1, pos_1 in player_positions.items():
        distances[p_1] = {}
        for p_2, pos_2 in [(p, pos) for p, pos in player_positions.items() if p != p_1]:

            # We can only calculate the distance if both players are alive
            if not deaths[p_1] and not deaths[p_2]:
                distances[p_1][p_2] = np.linalg.norm(pos_1 - pos_2)
            # Otherwise set to NaN
            else:
                distances[p_1][p_2] = float('NaN')

    return distances
