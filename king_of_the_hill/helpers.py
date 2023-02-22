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


# Note, only bothered implemnting this in 2d case. More dimensions should be done recursively.
def flatten_list(l: List[List[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]
