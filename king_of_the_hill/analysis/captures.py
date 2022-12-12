from typing import List

# TODO doc
def get_hill_captures(ep_rewards: List[dict]):
    # Special case where we are somehow in the first timestep and an agent already manages to capture the hill
    if len(ep_rewards) == 1:
        return {p: r > 0 for p, r in ep_rewards[-1].items()}

    # Normal case, where we are already 2+ timesteps into the episode.
    # Effectively, we count captures by checking if the reward increased
    return {p: r > ep_rewards[-2][p] and r == 1 for p, r in ep_rewards[-1].items()}
