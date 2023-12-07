import os
from typing import Dict, List, Callable, Optional
from dm_env import TimeStep
import gymnasium as gym
from gymnasium import logger
from ray.rllib.utils.typing import MultiAgentDict

from src.deepmind.meltingpot.configs.bots import BotConfig
from src.deepmind.meltingpot.utils.policies.policy import Policy as MeltingPotPolicy
from src.deepmind.meltingpot.bot import get_config as get_bot_config, build as build_bot

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

def load_bot_configs(bot_names: List[str], starting_idx) -> List[BotConfig]:
    """
    Loads bots from a list of bot names.
    @param bot_names: a list of bot names
    @param starting_idx: the starting index of the bots
    @return: a list of bot configs
    """
    bot_configs = []
    for idx, bot_name in enumerate(bot_names):
        bot_config = get_bot_config(bot_name)
        bot_dict = botConfig2Dict(bot_config)
        bot_dict['name'] = bot_name
        bot_dict['player'] = f'player_{starting_idx + idx}'
        bot_configs.append(bot_dict)
    return bot_configs

def load_bot_policies(bot_names: List[str]) -> List[MeltingPotPolicy]:
    """
    Loads bots from a list of bot names.
    @param bot_names: a list of bot names
    @return: a list of bot policies
    """
    return [build_bot(bot_name) for bot_name in bot_names]

class Bots_Stepper:
    def __init__(
            self, 
            bots: List[MeltingPotPolicy], 
            bot_configs: List[BotConfig], 
            starting_idx: int):
        self.bots = bots
        self.bot_configs = bot_configs
        self.bot_names = [b_c['player'] for b_c in bot_configs]
        self.idx_offset = starting_idx
        self.prev_states = {}
        self.prev_actions = {}
        self.individual_timesteps = {}
        #init pre_states
        for bot, bot_name in zip(self.bots, self.bot_names):
            self.prev_states[bot_name] = bot.initial_state()
            self.prev_actions[bot_name] = 0

    def __update_timestep(self, timestep: TimeStep):
        for idx, bot_name in enumerate(self.bot_names):
            individual_timestep = TimeStep(
                step_type=timestep.step_type,
                reward=timestep.reward,
                discount=timestep.discount,
                observation=timestep.observation[idx + self.idx_offset]
            )
            self.individual_timesteps[bot_name] = individual_timestep

    def step(self, timestep: TimeStep):
        self.__update_timestep(timestep)
        for bot, bot_name in zip(self.bots, self.bot_names):
            action, state_out = bot.step(
                timestep=self.individual_timesteps[bot_name],
                prev_state=self.prev_states[bot_name]
            )
            self.prev_states[bot_name] = state_out
            self.prev_actions[bot_name] = action
    
    def get_actions(self):
        return self.prev_actions
    
    def get_states(self):
        return self.prev_states
