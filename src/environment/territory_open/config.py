TERRITORY_OPEN_DEFAULT_CONFIG = {
    'episode_length_frames': 1000,  # this config fixes the episode length to always be 1000 frames
    'minimumFramesPerEpisode': 1000,
     # determine how long it takes until a claimed territory yields rewards. Changes to this variable are not reflected
     # visually in the game, but it will affect the reward signal.
    'reward_delay': 25,
    'reward_probability': 0.01,  # probability of a claimed territory yielding a reward. E.g 1 is 100%
}
