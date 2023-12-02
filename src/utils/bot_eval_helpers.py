import os
from typing import Dict, List, Callable, Optional
from dm_env import TimeStep
import gymnasium as gym
from gymnasium import logger
from gymnasium.wrappers.monitoring import video_recorder

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

def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

class CustomRecordVideo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """
    This is a custom RecordVideo wrapper that exposes the original TimeStep object.
    
    Original docstring:
    This wrapper records videos of rollouts.

    Usually, you only want to record episodes intermittently, say every hundredth episode.
    To do this, you can specify **either** ``episode_trigger`` **or** ``step_trigger`` (not both).
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed.
    By default, the recording will be stopped once a `terminated` or `truncated` signal has been emitted by the environment. However, you can
    also create recordings of fixed length (possibly spanning several episodes) by passing a strictly positive value for
    ``video_length``.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            disable_logger (bool): Whether to disable moviepy logger or not.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder: Optional[video_recorder.VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.frames = []
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
            timestep,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                if self.video_length > 0:
                    if self.recorded_frames > self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_recorder()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos, timestep

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment or as specified in kwargs."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()
