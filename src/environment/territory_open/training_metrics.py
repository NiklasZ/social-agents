from typing import List, Optional, Dict
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

"""
Collects extra metrics during episode. Some of these rely on the reward structure of the environment.
"""

"""
Notes about layering of env wrappers
- base_env argument in most methods in the MetricsCallback class is of type  
  <class 'ray.rllib.env.multi_agent_env.MultiAgentEnvWrapper'>
- Using the get_sub_environments() method on base_env returns a list of underlying environments
- The first element of the returned list is of type <class 'src.environment.wrappers.RGBObservationWrapper'>
- Using the get_dmlab2d_env() method on the RGBObservationWrapper returns an instance of
  <class 'src.deepmind.examples.rllib.utils.MeltingPotEnv'>
- Using the get_dmlab2d_env() method on the MeltingPotEnv returns an instance of
  <class 'meltingpot.utils.substrates.substrate.Substrate'>
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
        i_MultiAgentEnvWrapper = base_env.get_sub_environments()
        i_RGBObservationWrapper = i_MultiAgentEnvWrapper[0]
        i_MeltingPotEnv = i_RGBObservationWrapper.get_dmlab2d_env()
        substrate = i_MeltingPotEnv.get_dmlab2d_env()
        # get the registered events
        events = substrate.events()
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
        

