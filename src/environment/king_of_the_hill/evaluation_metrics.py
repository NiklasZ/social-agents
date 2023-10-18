from typing import Optional, List
import numpy as np


def find_hill_center(map_string: str):
    """
    Finds the character position of the center of the hill. Note, we assume a circular or rectangular-like hill.
    @param map_string: a koth map string as shown in map.py
    @return: x,y character position
    """
    rows = map_string.split('\n')
    rows = [row.strip() for row in rows if row.strip()]  # Remove empty lines

    # G denotes hill tiles.
    g_positions = [(i, j) for i, row in enumerate(rows) for j, c in enumerate(row) if c == 'G']
    if not g_positions:
        return None

    min_i = min(i for i, _ in g_positions)
    max_i = max(i for i, _ in g_positions)
    min_j = min(j for _, j in g_positions)
    max_j = max(j for _, j in g_positions)

    center_i = (min_i + max_i) // 2
    center_j = (min_j + max_j) // 2

    return center_j, center_i  # x, y coordinates


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

        # TODO should avoid hard-coding this and rely on information from the environment settings
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


def get_hill_captures(ep_rewards: List[dict]) -> dict:
    """
    Count how many times the hill is captured by a team
    @param ep_rewards:
    @return:
    """
    # Special case where we are somehow in the first timestep and an agent already manages to capture the hill
    if len(ep_rewards) == 1:
        return {p: r > 0 for p, r in ep_rewards[-1].items()}

    # Normal case, where we are already 2+ timesteps into the episode.
    # Effectively, we count captures by checking if the reward increased
    return {p: r > ep_rewards[-2][p] and r == 1 for p, r in ep_rewards[-1].items()}


def get_positions(infos: dict, obs_x: float, obs_y: float):
    player_distances = {}
    for player_id, info in infos.items():
        raw_x = info['POSITION'][0]
        raw_y = info['POSITION'][1]
        player_distances[player_id] = [int(raw_x - obs_x / 8), int(raw_y - obs_y / 8)]
    return player_distances


def get_distances(infos: dict, deaths: dict) -> dict:
    """
    Calculates the relative distance between pairs of players
    @param infos:
    @param deaths:
    @return:
    """
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


# TODO painting hill event

def update_episode_step_metrics(env_episode_metrics: dict,
                                episode_metrics: dict,
                                general_metrics: dict,
                                obs: dict,
                                actions: dict,
                                rewards: dict,
                                infos: dict):
    if not len(env_episode_metrics):
        # nested array with following dimensions: [episode #], [timestep t], {player #: (other players it can see)}
        env_episode_metrics['visibilities'] = []
        # nested array with following dimensions: [episode #], [timestep t], {player #: other players they are zapping}
        env_episode_metrics['zaps'] = []
        # nested array with the following dimensions: [episode #], [timestep t], {player #: whether they captured the hill}
        env_episode_metrics['captures'] = []
        # nested array with following dimensions: [episode #], [timestep t], {player #: current position (col,row)}
        env_episode_metrics['positions'] = []
        # nested array with following dimensions: [episode #], [timestep t], {player #: current position (col,row)}
        env_episode_metrics['orientations'] = []
        # nested array with following dimensions: [episode #], [timestep t], {player #: current position (col,row)}
        env_episode_metrics['distances'] = []

    env_episode_metrics['visibilities'].append(get_visibilities(infos, episode_metrics['deaths'][-1]))
    env_episode_metrics['zaps'].append(get_zaps(infos, actions))
    env_episode_metrics['captures'].append(get_hill_captures(episode_metrics['rewards']))
    obs_position = general_metrics['other']['observation_position_px']
    env_episode_metrics['positions'].append(get_positions(infos, obs_position['x'], obs_position['y']))
    env_episode_metrics['orientations'].append({p: v['ORIENTATION'].tolist() for p, v in infos.items()})
    env_episode_metrics['distances'].append(get_distances(infos, episode_metrics['deaths'][-1]))


def update_episode_end_metrics(env_metrics: dict,
                               general_metrics: dict,
                               env_episode_metrics: dict,
                               episode_metrics: dict):
    if not len(env_metrics):
        env_metrics['visibilities'] = []
        env_metrics['zaps'] = []
        env_metrics['captures'] = []
        env_metrics['positions'] = []
        env_metrics['orientations'] = []
        env_metrics['distances'] = []

    env_metrics['visibilities'].append(env_episode_metrics['visibilities'])
    env_metrics['zaps'].append(env_episode_metrics['zaps'])
    env_metrics['captures'].append(env_episode_metrics['captures'])
    env_metrics['positions'].append(env_episode_metrics['positions'])
    env_metrics['orientations'].append(env_episode_metrics['orientations'])
    env_metrics['distances'].append(env_episode_metrics['distances']),

    # Get hill position of the environment
    hill_center_x, hill_center_y = find_hill_center(map_string=general_metrics['other']['map_string'])
    env_metrics['hill_position'] = {'center_x': hill_center_x, 'center_y': hill_center_y}


KOTH_EVAL_METRICS_FUNCS = {'episode_step': update_episode_step_metrics,
                           'episode_end': update_episode_end_metrics
                           }
