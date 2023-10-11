from typing import List, Optional, Dict
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.tune import Callback

"""
Collects extra metrics during episode. A lot of these rely on the reward structure of the environment.
"""


# TODO record actions taken (mean)
# TODO add first hill tile painted
# TODO track events through episode.last_info_for('player_0') rather than interpreting reward.
class MetricsCallbacks(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: EpisodeV2,
                         **kwargs):
        # We will use the `MultiAgentEpisode` object being passed into
        # all episode-related callbacks. It comes with a user_data property (dict),
        # which we can write arbitrary data into.

        # At the end of an episode, we'll transfer that data into the `hist_data`, and `custom_metrics`
        # properties to make sure our custom data is displayed in TensorBoard.

        # Step of first capture of the hill
        # we pick 1000 as the episode length in the case that the hill is never captured
        episode.user_data['first_capture'] = worker.env_context['lab2d_settings']['maxEpisodeLengthFrames'] # usually 1000
        episode.user_data['red_last_reward'] = 0
        episode.user_data['blue_last_reward'] = 0
        # Number of times hill is captured by colour
        episode.user_data['red_captures'] = 0
        episode.user_data['blue_captures'] = 0

    def on_episode_step(self,
                        *,
                        worker,
                        base_env,
                        episode: EpisodeV2,
                        env_index,
                        **kwargs):
        # Red team reward
        r = episode._agent_reward_history['player_0'][-1]
        # Blue team reward
        b = episode._agent_reward_history['player_1'][-1]
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
                       episode: EpisodeV2,
                       env_index,
                       **kwargs):
        # Episode is done:
        # Write scalar values (sum over rewards) to `custom_metrics` and
        # time-series data (rewards per time step) to `hist_data`.
        # Both will be visible then in TensorBoard.
        episode.custom_metrics["first_capture_step"] = episode.user_data['first_capture']
        episode.custom_metrics["red_captures"] = episode.user_data['red_captures']
        episode.custom_metrics["blue_captures"] = episode.user_data['blue_captures']
