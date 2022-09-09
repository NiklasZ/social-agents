from typing import List, Optional, Dict
from ray.rllib import BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID
from ray.tune import Callback

"""
Collects extra metrics during episode. A lot of these rely on the reward structure of the environment.
"""


# TODO add first hill tile painted
class MetricsCallbacks(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: Episode,
                         **kwargs):
        # We will use the `MultiAgentEpisode` object being passed into
        # all episode-related callbacks. It comes with a user_data property (dict),
        # which we can write arbitrary data into.

        # At the end of an episode, we'll transfer that data into the `hist_data`, and `custom_metrics`
        # properties to make sure our custom data is displayed in TensorBoard.

        # Step of first capture of the hill
        # we pick 1000 as the episode length in the case that the hill is never captured
        episode.user_data['first_capture'] = 1000
        episode.user_data['red_last_reward'] = 0
        episode.user_data['blue_last_reward'] = 0
        # Number of times hill is captured by colour
        episode.user_data['red_captures'] = 0
        episode.user_data['blue_captures'] = 0
        # # Set per-episode agent2-blocks counter (how many times has agent2 blocked agent1?).
        # episode.user_data["num_collisions"] = 0
        # # Set per-episode agent2-approaches
        # episode.user_data["a"] = 0
        pass

    def on_episode_step(self,
                        *,
                        worker,
                        base_env,
                        episode: Episode,
                        env_index,
                        **kwargs):
        # Red team reward
        r = episode.last_reward_for('player_0')
        # Blue team reward
        b = episode.last_reward_for('player_1')
        capture = False

        # Check for red capture
        if r > episode.user_data['red_last_reward']:
            episode.user_data['red_captures'] += 1
            capture = True

        episode.user_data['red_last_reward'] = r

        # Check for red capture
        if b > episode.user_data['blue_last_reward']:
            episode.user_data['blue_captures'] += 1
            capture = True

        episode.user_data['blue_last_reward'] = b

        # Check for first capture
        if capture and episode.length < episode.user_data['first_capture']:
            episode.user_data['first_capture'] = episode.length

    def on_episode_end(self,
                       *,
                       worker,
                       base_env,
                       policies,
                       episode: Episode,
                       env_index,
                       **kwargs):
        # Episode is done:
        # Write scalar values (sum over rewards) to `custom_metrics` and
        # time-series data (rewards per time step) to `hist_data`.
        # Both will be visible then in TensorBoard.
        episode.custom_metrics["first_capture_step"] = episode.user_data['first_capture']
        episode.custom_metrics["red_captures"] = episode.user_data['red_captures']
        episode.custom_metrics["blue_captures"] = episode.user_data['blue_captures']
        pass
