def get_visibilities(infos: dict, deaths: dict) -> dict:
    # Filter out dead players as they obviously can't be seen
    player_positions = {p: infos[p]['POSITION'] for p in infos}
    player_orientations = {p: infos[p]['ORIENTATION'] for p in infos}
    player_visibilities = {}
    for p in player_positions:
        player_visibilities[p] = []

        # If the player is dead, then they can't see anyone so we skip them.
        if deaths[p]:
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
        left_offsets = [-5, -1, -5, -9]
        right_offsets = [5, 9, 5, 1]
        down_offsets = [1, 5, 9, 5]
        up_offsets = [-9, -5, -1, -5]

        # Box around a player
        top_left = (pos[0] + left_offsets[orient], pos[1] + up_offsets[orient])
        bottom_right = (pos[0] + right_offsets[orient], pos[1] + down_offsets[orient])

        # Loop over other players that are not dead
        for p_2 in [i for i in player_positions if i != p and not deaths[i]]:
            x, y = player_positions[p_2]
            if top_left[0] < x < bottom_right[0] and top_left[1] < y < bottom_right[1]:
                player_visibilities[p].append(p_2)

    return player_visibilities
