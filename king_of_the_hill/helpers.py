import re
from typing import List, T


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


# Note, only bothered implemnting this in 2d case. More dimensions should be done recursively.
def flatten_list(l: List[List[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]
