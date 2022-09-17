from typing import Union
from collections import deque

import numpy as np
from ray.rllib.env import multi_agent_env

from examples.rllib.utils import MeltingPotEnv
from gym import spaces


# Not really a wrapper as it inherits from MeltingPotEnv rather than gym.wrappers, but this is necessary
# as a regular wrapper won't work without using the melting pot adapter.

# TODO define a base wrapepr class
class FullRGBObservationWrapper(multi_agent_env.MultiAgentEnv):
    def __init__(self,
                 env: MeltingPotEnv,
                 world_height: int,
                 world_width: int,
                 sprite_size=8):  # number of pixels a tile/sprite on a map is long
        self.world_height = world_height
        self.world_width = world_width
        self.sprite_size = sprite_size
        self._env = env
        self._agent_ids = self._env.get_agent_ids()
        player_ids = self._env.observation_space.keys()
        player_0 = list(player_ids)[0]
        # Replace observation height, width with world parameters.
        _, _, C = self._env.observation_space[player_0]['WORLD.RGB'].shape
        self.observation_space = spaces.Dict({p:
            spaces.Box(
                low=0,
                high=255,
                shape=(world_height * sprite_size, world_width * sprite_size, C),
                dtype=self._env.observation_space[player_0]['WORLD.RGB'].dtype,
            ) for p in player_ids})
        self.action_space = self._env.action_space
        super().__init__()

    def get_world_rgb(self, obs_dict: Union[dict, spaces.Dict]):
        # The WORLD.RGB doesn't exactly return the gameplay world. It actually pads it out into a larger image with
        # a lot of unused space. To avoid dealing with that, we crop to the gameplay world.
        return {
            p: obs_dict[p]['WORLD.RGB'][:self.world_height * self.sprite_size, -self.world_height * self.sprite_size:]
            for p in obs_dict.keys()}

    def reset(self, *args, **kwargs):
        # modify obs
        ob = self._env.reset()
        return self.get_world_rgb(ob)

    def step(self, *args, **kwargs):
        # modify obs
        ob, reward, done, info = self._env.step(*args, **kwargs)
        return self.get_world_rgb(ob), reward, done, info

    def close(self):
        """See base class."""
        self._env.close()

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    def render(self, mode: str) -> np.ndarray:
        return self._env.render(mode=mode)

    @property
    def metadata(self):
        return self._env.metadata

class RGBObservationWrapper(multi_agent_env.MultiAgentEnv):
    def __init__(self, env: MeltingPotEnv):
        self._env = env
        self._agent_ids = self._env.get_agent_ids()
        self.observation_space = get_rgb_space(self._env.observation_space)
        self.action_space = self._env.action_space
        super().__init__()

    def reset(self, *args, **kwargs):
        # modify obs
        ob = self._env.reset()
        return get_rgb(ob)

    def step(self, *args, **kwargs):
        # modify obs
        ob, reward, done, info = self._env.step(*args, **kwargs)
        return get_rgb(ob), reward, done, info

    def close(self):
        """See base class."""
        self._env.close()

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    def render(self, mode: str) -> np.ndarray:
        return self._env.render(mode=mode)

    @property
    def metadata(self):
        return self._env.metadata


def get_rgb(obs_dict: Union[dict, spaces.Dict]):
    return {p: obs_dict[p]['RGB'] for p in obs_dict.keys()}


def get_rgb_space(obs_dict: spaces.Dict) -> spaces.Dict:
    return spaces.Dict(get_rgb(obs_dict))


# Stacks k frames in a multi-agent environment
# Loosely inspired by ray/rllib/env/wrappers/atari_wrappers.py
class FrameStacker(multi_agent_env.MultiAgentEnv):
    def __init__(self, env: MeltingPotEnv, stack_frames=4):
        self._env = env
        self.k = stack_frames
        player_ids = self._env.observation_space.keys()
        self.frames = {p: deque([], maxlen=self.k) for p in player_ids}
        player_0 = list(player_ids)[0]
        H, W, C = self._env.observation_space[player_0].shape
        self.observation_space = spaces.Dict({p:
            spaces.Box(
                low=0,
                high=255,
                shape=(H, W, C * self.k),
                dtype=self._env.observation_space[player_0].dtype,
            ) for p in player_ids})
        self.action_space = self._env.action_space
        self._agent_ids = self._env.get_agent_ids()
        super().__init__()

    def reset(self, *args, **kwargs):
        # modify obs
        obs = self._env.reset()
        for p in obs.keys():  # iterate players
            for _ in range(self.k):  # set each frame to be first observation
                self.frames[p].append(obs[p])
        return self._stack_obs()

    def step(self, *args, **kwargs):
        # modify obs
        obs, reward, done, info = self._env.step(*args, **kwargs)
        for p in obs.keys():  # iterate players
            self.frames[p].append(obs[p])
        return self._stack_obs(), reward, done, info

    def close(self):
        """See base class."""
        self._env.close()

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    def render(self, mode: str) -> np.ndarray:
        return self._env.render(mode=mode)

    def _stack_obs(self):
        return {p: np.concatenate(self.frames[p], axis=2) for p in self.frames.keys()}

    @property
    def metadata(self):
        return self._env.metadata
