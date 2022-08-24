from typing import Union
from collections import deque

import numpy as np

from examples.rllib.utils import MeltingPotEnv
from gym import spaces


# TODO maybe I should shift these to config["multiagent"]["observation_fn"] instead
# Not really a wrapper as it inherits from MeltingPotEnv rather than gym.wrappers, but this is necessary
# as a regular wrapper won't work without using the melting pot adapter.
class RGBObservationWrapper(MeltingPotEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = get_rgb_space(self.observation_space)

    def reset(self, *args, **kwargs):
        # modify obs
        ob = super().reset()
        return get_rgb(ob)

    def step(self, *args, **kwargs):
        # modify obs
        ob, reward, done, info = super().step(*args, **kwargs)
        return get_rgb(ob), reward, done, info


def get_rgb(obs_dict: Union[dict, spaces.Dict]):
    return {p: obs_dict[p]['RGB'] for p in obs_dict.keys()}


def get_rgb_space(obs_dict: spaces.Dict) -> spaces.Dict:
    return spaces.Dict(get_rgb(obs_dict))


# Stacks k frames in a multi-agent environment
# Loosely inspired by ray/rllib/env/wrappers/atari_wrappers.py
class FrameStacker(RGBObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = kwargs['stack_frames'] if 'stack_frames' in kwargs else 4
        player_ids = self.observation_space.keys()
        self.frames = {p: deque([], maxlen=self.k) for p in player_ids}
        player_0 = list(player_ids)[0]
        H, W, C = self.observation_space[player_0].shape
        self.observation_space = spaces.Dict({p:
            spaces.Box(
                low=0,
                high=255,
                shape=(H, W, C * self.k),
                dtype=self.observation_space[player_0].dtype,
            ) for p in player_ids})

    def reset(self, *args, **kwargs):
        # modify obs
        obs = super().reset()
        for p in obs.keys():  # iterate players
            for _ in range(self.k):  # set each frame to be first observation
                self.frames[p].append(obs[p])
        return self._stack_obs()

    def step(self, *args, **kwargs):
        # modify obs
        obs, reward, done, info = super().step(*args, **kwargs)
        for p in obs.keys():  # iterate players
            self.frames[p].append(obs[p])
        return self._stack_obs(), reward, done, info

    def _stack_obs(self):
        return {p: np.concatenate(self.frames[p], axis=2) for p in self.frames.keys()}

# def gert
