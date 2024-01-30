from typing import List, Optional, Dict
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

"""
Collects extra metrics during episode. A lot of these rely on the reward structure of the environment.
"""


class MetricsCallback(DefaultCallbacks):
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

        episode.user_data['blue_picks_all'] = 0
        episode.user_data['red_picks_all'] = 0
        episode.user_data['blue_picks_own'] = 0
        episode.user_data['red_picks_own'] = 0
        episode.user_data['red_last_reward'] = 0
        episode.user_data['blue_last_reward'] = 0

    def on_episode_step(self,
                        *,
                        worker,
                        base_env,
                        episode: EpisodeV2,
                        env_index,
                        **kwargs):
        delta_red = episode._agent_reward_history['player_0'][-1]
        delta_blue = episode._agent_reward_history['player_1'][-1]
    
        match (int(delta_red), int(delta_blue)):
            #blue pick red, red pick red
            case (-1, 1):
                episode.user_data['red_picks_all'] += 1
                episode.user_data['blue_picks_all'] += 1
                episode.user_data['red_picks_own'] += 1
            #blue pick red, red pick blue
            case (-1, -1):
                episode.user_data['red_picks_all'] += 1
                episode.user_data['blue_picks_all'] += 1
            #blue pick red, red pick none
            case (-2, 1):
                episode.user_data['blue_picks_all'] += 1
            #blue pick blue, red pick red
            case (1, 1):
                episode.user_data['red_picks_all'] += 1
                episode.user_data['blue_picks_all'] += 1
                episode.user_data['red_picks_own'] += 1
                episode.user_data['blue_picks_own'] += 1
            #blue pick blue, red pick blue
            case (1, -1):
                episode.user_data['red_picks_all'] += 1
                episode.user_data['blue_picks_all'] += 1
                episode.user_data['blue_picks_own'] += 1
            #blue pick blue, red pick none
            case (0, 1):
                episode.user_data['blue_picks_all'] += 1
                episode.user_data['blue_picks_own'] += 1
            #blue pick none, red pick red
            case (1, 0):
                episode.user_data['red_picks_all'] += 1
                episode.user_data['red_picks_own'] += 1
            #blue pick none, red pick blue
            case (1, -2):
                episode.user_data['red_picks_all'] += 1
            #blue pick none, red pick none
            case (0, 0):
                pass
            case (_, _):
                raise Exception(f'Invalid reward values with delta_red={delta_red}, delta_blue={delta_blue}')
    
    def on_episode_end(self,
                       *,
                       worker,
                       base_env,
                       policies,
                       episode: EpisodeV2,
                       env_index,
                       **kwargs):
        # Episode is done:
        episode.custom_metrics["red_picks_all"] = episode.user_data['red_picks_all']
        episode.custom_metrics["blue_picks_all"] = episode.user_data['blue_picks_all']
        episode.custom_metrics["red_picks_own"] = episode.user_data['red_picks_own']
        episode.custom_metrics["blue_picks_own"] = episode.user_data['blue_picks_own']
        episode.custom_metrics["red_prop_own2all"] = episode.user_data['red_picks_own'] / episode.user_data['red_picks_all']
        episode.custom_metrics["blue_prop_own2all"] = episode.user_data['blue_picks_own'] / episode.user_data['blue_picks_all']
