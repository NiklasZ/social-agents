from typing import List, Optional, Dict
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.utils.typing import EnvType
from src.environment.wrappers import RGBObservationWrapper
from src.deepmind.examples.rllib.utils import MeltingPotEnv
from src.deepmind.meltingpot.utils.substrates.substrate import Substrate
from src.environment.wrappers import get_events_from_env_wrapper

"""
Collects extra metrics during episode. Some of these rely on the reward structure of the environment.
"""

class MetricsCallback(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: EpisodeV2,
                         **kwargs):


        # At the end of an episode, we'll transfer that data into the `hist_data`, and `custom_metrics`
        # properties to make sure our custom data is displayed in TensorBoard.

        episode.user_data['n_walls_ever_claimed'] = 0 
        episode.user_data['n_walls_destroyed'] = 0 
        episode.user_data['is_alive'] = [True] * len(policies) # Assume all agents are alive at the start of the episode
        episode.user_data['n_episodes_lived'] = [0] * len(policies) # Number of episodes lived by each agent

    def on_episode_step(self,
                        *,
                        worker,
                        base_env,
                        episode: EpisodeV2,
                        env_index,
                        **kwargs):
        
        #get the substrate
        sub_envs: List[EnvType] = base_env.get_sub_environments()
        assert(len(sub_envs) == 1, "There should be only one sub environment")
        observation_wrapper: RGBObservationWrapper = sub_envs[0]
        #get registered events
        events = get_events_from_env_wrapper(observation_wrapper)
        for event in events:
            event_type = event[0]
            # Update the number of walls ever claimed and destroyed
            if event_type == 'resources_stats':
                episode.user_data['n_walls_ever_claimed'] = event[1][2]
                episode.user_data['n_walls_destroyed'] = event[1][4]
            elif event_type == 'removal_due_to_sanctioning':
                dead_agent_index = int(event[1][4]) - 1
                episode.user_data['is_alive'][dead_agent_index] = False
        # Update the number of episodes lived by each agent
        for agent_index in range(len(episode.user_data['is_alive'])):
            if episode.user_data['is_alive'][agent_index]:
                episode.user_data['n_episodes_lived'][agent_index] += 1
        
    def on_episode_end(self,
                       *,
                       worker,
                       base_env,
                       policies,
                       episode: EpisodeV2,
                       env_index,
                       **kwargs):
        # Episode is done:
        episode.custom_metrics["n_walls_ever_claimed"] = episode.user_data['n_walls_ever_claimed']
        episode.custom_metrics["n_walls_destroyed"] = episode.user_data['n_walls_destroyed']
        for agent_index in range(len(episode.user_data['is_alive'])):
            episode.custom_metrics[f"n_episodes_lived_agent_{agent_index}"] = episode.user_data['n_episodes_lived'][agent_index]
        

