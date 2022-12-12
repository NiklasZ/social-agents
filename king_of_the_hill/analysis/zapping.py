import numpy as np


def get_zaps(infos: dict, actions: dict) -> dict:
    player_positions = {p: infos[p]['POSITION'] for p in infos}
    player_orientations = {p: infos[p]['ORIENTATION'] for p in infos}
    player_zapping = {}

    for p in infos:
        player_zapping[p] = []
        # skip cases where the player isn't zapping
        if actions[p] != 7:
            continue

        # A position is defined by a (col,row) tuple, with (0,0) being in the top-left corner of the map.
        pos = player_positions[p]
        # Orientation ranges from 0 to 3, where:
        # 0 is facing north
        # 1 is facing east
        # 2 is facing south
        # 3 is facing west
        orient = player_orientations[p]

        # Each player has an observation around itself, where it can see:
        # 9 rows forwards
        # 1 row backwards
        # 5 columns to the left
        # 5 columns to the right

        # Depending on orientation,these yields the following distances a player can see.
        left_offsets = [-1, 0, -1, -2]
        right_offsets = [1, 2, 1, 0]
        down_offsets = [0, 1, 2, 1]
        up_offsets = [-2, -1, 0, -1]

        tip_offsets = np.array([(0, -3), (3, 0), (0, 3), (-3, 0)])

        # Zap box around a player
        top_left = (pos[0] + left_offsets[orient], pos[1] + up_offsets[orient])
        bottom_right = (pos[0] + right_offsets[orient], pos[1] + down_offsets[orient])
        tip = pos + tip_offsets[orient]

        # Loop over other players
        for p_2 in [i for i in player_positions if i != p]:
            x, y = player_positions[p_2]
            if (top_left[0] <= x <= bottom_right[0] and
                top_left[1] <= y <= bottom_right[1]) or \
                    np.all(tip == player_positions[p_2]):
                player_zapping[p].append(p_2)

    return player_zapping
