import collections.abc
import json
import re
from typing import List, T
import cv2
import os

import numpy as np
from torch import Tensor

from src.deepmind.examples.rllib.utils import MeltingPotEnv
from src.deepmind.meltingpot.configs.bots import BotConfig

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
    _, _, _, _, info, _ = env.step(action)
    player_id = list(info.keys())[0]
    world_rgb_arr = info[player_id]['WORLD.RGB']
    env.reset()

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


def dict_to_argparser_args(d: dict) -> List[str]:
    """
    Converts a dictionary to a list of strings that can be passed to an argparser.
    @param d:
    @return:
    """
    args_list = []
    for key, value in d.items():
        key = '--' + key  # prepend '--' to make it a valid command-line argument
        if isinstance(value, bool):
            # For boolean, only add the key if the value is True
            if value:
                args_list.append(key)
        else:
            # For non-boolean, convert value to string and add both key and value
            args_list.extend([key, str(value)])

    return args_list


def remove_prefix_from_files_in(folder: str, prefix: str):
    """
    Removes a prefix from all files in a folder.
    @param folder: folder to remove prefix from
    @param prefix: prefix to remove
    @return:
    """
    assert os.path.isdir(folder), f"Folder {folder} does not exist"
    for filename in os.listdir(folder):
        os.rename(os.path.join(folder, filename), os.path.join(folder, filename.replace(prefix, '')))


def log_verbose(value, verbose):
    if verbose:
        print(value)


def get_property(root_obj: object, property_path: str) -> object:
    """
    Gets the content of a nested property of an object, including support for indexing. Some exmaples:
    get_content_path(model, 'lstm.weight_hh_l0.0.0') -> single parameter tensor (i.e one weight).
    @param root_obj: The root object from which to start retrieving content.
    @param property_path: A string representing the path to the desired content, using dot notation and indices.
    @return: The content at the specified path, if it exists.
    """
    obj = root_obj
    for part in property_path.split("."):
        # Handle index in path
        if part.isdigit():
            idx = int(part)
            if not isinstance(obj, (collections.abc.Sequence, Tensor, np.ndarray)):
                raise Exception(f"Attempted to access non-indexable object in '{property_path}' with index {part}.")
            if idx > len(obj):
                raise IndexError(f"Index {idx} is out of range for object in '{property_path}' of length {len(obj)}.")
            obj = obj[idx]
        # Handle attribute in path
        else:
            # Check if the object has the attribute
            if not hasattr(obj, part):
                raise AttributeError(f"Object path {property_path} does not have attribute '{part}'.")
            obj = getattr(obj, part)
    return obj


def check_json_has_fields(given_json: str, target_json: str):
    given_fields = set(json.loads(given_json))
    target_fields = set(json.loads(target_json))
    different_fields = target_fields.difference(given_fields)
    assert len(different_fields) == 0, (f"Given JSON '{given_json}' is missing the following fields: " +
                                        str(different_fields))
    
def botConfig2Dict(config: BotConfig):
    """
    Converts a bot config to a dictionary.
    @param config: a bot config
    @return: a dictionary representing the bot config
    """
    return {
        'model_path': config.model_path,
        'substrate': config.substrate,
        'puppeteer_builder': config.puppeteer_builder,
        'role': next(iter(config.roles)),
    }
