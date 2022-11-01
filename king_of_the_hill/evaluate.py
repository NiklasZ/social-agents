from typing import Dict, Any, List, Tuple
import numpy as np
import ray
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from ray.rllib.agents.ppo import ppo
from king_of_the_hill.arg_parser import get_evaluation_args
from king_of_the_hill.generate_config import get_trailing_number
from gym.wrappers import RecordVideo
from datetime import datetime
from king_of_the_hill.wandb_api import get_api_key, download_evaluation_state
from king_of_the_hill.models import get_model
import os
import glob


def log_verbose(value, verbose):
    if verbose:
        print(value)


# Gets list of tuples containing policy id and player name
def get_names(config: Dict, run_config: Dict) -> List[Tuple[str, str]]:
    # Case for one default model
    if not run_config['multi_model']:
        return [('av', f'player_{i}') for i in range(config['env_config']['num_players'])]
    # Case for multiple models
    return [(p, f'player_{get_trailing_number(p)}') for p, _ in sorted(config['multiagent']['policies'].items())]


def evaluate(run_config: Dict):
    ray.init(local_mode=True)  # Run with local_mode=True to use debugger
    rllib_path = '/tmp/ray_results'
    checkpoint_folder = f"checkpoint_{run_config['checkpoint_number']:06}"
    checkpoint_name = f"checkpoint-{run_config['checkpoint_number']}"
    api_key = get_api_key()
    checkpoint_path, config, run_config = download_evaluation_state(rllib_path=rllib_path,
                                                                    run_config=run_config,
                                                                    api_key=api_key,
                                                                    checkpoint_folder=checkpoint_folder)
    print(f"Evaluating run: {run_config['rllib_run_name']}, id: {run_config['run_id']}")
    path = f"{checkpoint_path}/{checkpoint_name}"
    agent = ppo.PPOTrainer(config=config)
    agent.restore(
        path)  # If we have a lot of policies RLlib will dump a huge number of pickled policy files (.pkl) into the running directory.
    model_description = get_model(run_config['model_name'])
    env_constructor = model_description['env_constructor']
    test_env = env_constructor(config["env_config"])
    test_env.metadata['video.frames_per_second'] = run_config['recording_fps']
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    video_path = f"videos/{run_config['rllib_run_name']}/{checkpoint_name}-{timestamp}"
    recordable_env = RecordVideo(test_env, video_path, video_length=1000, name_prefix='')  # 1000 steps is one episode

    for episode in range(run_config['episodes']):
        print(f'==Episode {episode}==')
        done = False
        obs = recordable_env.reset()

        # If using RNN/LSTM, we need to track more information
        if config['model']['use_lstm']:
            prev_actions = {}
            states = {}
            for _, player in get_names(config, run_config):
                # Initial hidden and cell state
                states[player] = [np.zeros(config['model']['lstm_cell_size'], ) for _ in range(2)]
                prev_actions[player] = np.zeros(1, dtype='int32')

        if run_config['record_observations']:
            observation_recorders = {player: ImageEncoder(
                output_path=f"{video_path}/episode-{episode * 1000}-{player}.mp4",
                frame_shape=list(obs.values())[0].shape,
                frames_per_sec=run_config['recording_fps'],
                output_frames_per_sec=run_config['recording_fps']) for _, player in get_names(config, run_config)}
        while not done:
            # If using multiple models
            # if run_config['multi_model']:
            actions = {}
            # Sorted to ensure we always iterate policy ids in the same order
            for policy, player in get_names(config, run_config):
                # If using RNN/LSTM we need to keep track of hidden states
                if config['model']['use_lstm']:
                    action, state_out, _ = agent.compute_single_action(observation=obs[player],
                                                                       state=states[player],
                                                                       prev_action=prev_actions[player],
                                                                       policy_id=policy)
                    states[player] = state_out
                    prev_actions[player] = action
                else:
                    action = agent.compute_single_action(obs[player], policy_id=policy)
                log_verbose(f'{player}, {policy}, {action}', run_config['verbose'])
                actions[player] = action
            obs, reward, dones, _ = recordable_env.step(actions)
            done = all(dones.values())

            if run_config['record_observations']:
                for player in obs:
                    observation_recorders[player].capture_frame(obs[player])

        if run_config['record_observations']:
            for player in observation_recorders:
                observation_recorders[player].close()

        recordable_env.close_video_recorder()


if __name__ == "__main__":
    args = vars(get_evaluation_args())
    try:
        evaluate(args)
    # RLlib will sometimes dump a huge number of .pkl files into the running directory. We don't want to keep these.
    finally:
        cwd = os.getcwd()
        policy_file_list = glob.glob(f'{cwd}/*.pkl')
        for path in policy_file_list:
            os.remove(path)
