import re
from typing import List, T
import cv2
import numpy as np

from examples.rllib.utils import MeltingPotEnv


# Gets trailing number from a string.
# E.g "Bananas123" yields 123
def get_trailing_number(s: str) -> int:
    m = re.search(r'\d+$', s)
    if m:
        return int(m.group())
    else:
        raise Exception(f"No number found at the end of '{s}'")


# Maps a player name to a corresponding main policy.
#  E.g "player_0" --> "policy0"
def to_main_policy_name(player_name: str):
    regex_expr = r'^player_\d+$'
    m = re.search(regex_expr, player_name)
    if m:
        return f"policy{get_trailing_number(player_name)}"
    else:
        raise Exception(f'Expected player name matching {regex_expr}, but got {player_name}')


# Maps a policy name to a player
# E.g "policy0" --> "player_0"
# E.g "policy11_v0" --> "player_11"
def to_player_name(policy_name: str):
    regex_expr = r'^policy\d+(_v\d+)?$'
    m = re.search(regex_expr, policy_name)
    if m:
        return f"player_{get_trailing_number(policy_name.split('_')[0])}"
    else:
        raise Exception(f'Expected policy name matching {regex_expr}, but got {policy_name}')


# Note, only bothered implementing this in 2d case. More dimensions should be done recursively.
def flatten_list(l: List[List[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]


def get_world_rgb_bounding_box(env: MeltingPotEnv) -> (int, int, int, int):
    """
    Gets a bounding box around environment from the WORLD.RGB. The assumed coordinate system is that the origin (0,0)
    is the top-left corner of the image.
    @param env: a melting pot environment
    @return: top-left x,y coordinates, width of box, height of box
    """
    env.reset()
    action = env.action_space.sample()
    _, _, _, info = env.step(action)
    player_id = list(info.keys())[0]
    world_rgb_arr = info[player_id]['WORLD.RGB']

    # Assuming 'image' is your h x w x 3 image
    image = cv2.cvtColor(world_rgb_arr, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)  # Apply thresholding to get a binary image

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # You should get at least one contour, but there might be more (for instance, if your image contains multiple separate components).
    # Let's assume that the largest contour corresponds to the image contained within the border.
    # If that's not a valid assumption, you'll need to refine this approach.
    max_contour = max(contours, key=cv2.contourArea)

    # Get bounding box coordinates around the contour
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, w, h


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


def get_positions(infos: dict, obs_x: float, obs_y: float):
    player_distances = {}
    for player_id, info in infos.items():
        raw_x = info['POSITION'][0]
        raw_y = info['POSITION'][1]
        player_distances[player_id] = [int(raw_x - obs_x / 8), int(raw_y - obs_y / 8)]
    return player_distances
