from typing import Dict, Any, List, Tuple
import numpy as np
import ray
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from ray.rllib.algorithms.ppo import PPO
from king_of_the_hill.analysis.captures import get_hill_captures
from king_of_the_hill.analysis.distances import get_distances
from king_of_the_hill.analysis.visibility import get_visibilities
from king_of_the_hill.analysis.zapping import get_zaps
from king_of_the_hill.arg_parser import get_evaluation_args
from king_of_the_hill.generate_config import get_trailing_number
from gym.wrappers import RecordVideo
from datetime import datetime
from king_of_the_hill.wandb_api import get_api_key, download_evaluation_state
from king_of_the_hill.models import get_model
import os
import glob
import json


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
    # TODO use workers for evaluation
    ray.init(local_mode=run_config['debug_mode'])  # Run with local_mode=True to use debugger
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
    agent = PPO(config=config)
    agent.restore(
        path)  # If we have a lot of policies RLlib will dump a huge number of pickled policy files (.pkl) into the running directory.
    model_description = get_model(run_config['model_name'])
    env_constructor = model_description['env_constructor']
    test_env = env_constructor(config["env_config"])
    test_env.metadata['video.frames_per_second'] = run_config['recording_fps']
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    eval_path = f"evaluation/{run_config['rllib_run_name']}/{checkpoint_name}-{timestamp}"
    video_path = f"{eval_path}/videos"
    recordable_env = RecordVideo(test_env, video_path, video_length=1000, name_prefix='')  # 1000 steps is one episode

    # Evaluation metrics to track
    metrics = {
        # nested array with the following dimensions: [episode #], [timestep t], {player #: action at timestep t}
        'rewards': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: reward at timestep t}
        'actions': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: (other players it can see)}
        'visibilities': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: current position (col,row)}
        'positions': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: current position (col,row)}
        'distances': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: {other player #: distance}}
        'deaths': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: other players they are zapping}
        'zaps': [],
        # nested array with the following dimensions: [episode #], [timestep t], {player #: whether they captured the hill}
        'captures': []
        # TODO painting hill event
    }

    for episode in range(run_config['episodes']):
        print(f'==Episode {episode}==')
        done = False
        obs = recordable_env.reset()

        ep_rewards = []
        ep_actions = []
        ep_visibilities = []
        ep_positions = []
        ep_deaths = []
        ep_zaps = []
        ep_distances = []
        ep_captures = []

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

            obs, reward, dones, infos = recordable_env.step(actions)

            ep_rewards.append({p: float(r.item()) for p, r in reward.items()})  # Map numpy array value to regular float
            ep_actions.append({p: int(a.item()) for p, a in actions.items()})
            ep_visibilities.append(get_visibilities(infos))
            ep_positions.append({p: v['POSITION'].tolist() for p, v in infos.items()})
            deaths = {p: (np.max(ob) == 0).item() for p, ob in obs.items()}
            ep_deaths.append(deaths)  # if a player is given an empty obs, it has recently been killed
            ep_zaps.append(get_zaps(infos, actions))
            ep_distances.append(get_distances(infos, deaths))
            ep_captures.append(get_hill_captures(ep_rewards))

            done = all(dones.values())

            if run_config['record_observations']:
                for player in obs:
                    observation_recorders[player].capture_frame(obs[player])

        if run_config['record_observations']:
            for player in observation_recorders:
                observation_recorders[player].close()

        metrics['rewards'].append(ep_rewards)
        metrics['actions'].append(ep_actions)
        metrics['visibilities'].append(ep_visibilities)
        metrics['positions'].append(ep_positions)
        metrics['deaths'].append(ep_deaths)
        metrics['zaps'].append(ep_zaps)
        metrics['distances'].append(ep_distances)
        metrics['captures'].append(ep_captures)

        recordable_env.close_video_recorder()

        # Record metrics every few eval steps. Just in case we interrupt evaluation at some point
        if episode % 5 == 0 or episode == run_config['episodes'] - 1:
            with open(f'{eval_path}/metrics.json', 'w') as convert_file:
                convert_file.write(json.dumps(metrics))


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
