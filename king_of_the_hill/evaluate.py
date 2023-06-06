from typing import Dict, Any, List, Tuple
import numpy as np
import ray
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from ray.rllib.algorithms.ppo import PPO
from king_of_the_hill.activations import ActivationExtractor
from king_of_the_hill.analysis.captures import get_hill_captures
from king_of_the_hill.analysis.distances import get_distances
from king_of_the_hill.analysis.visibility import get_visibilities
from king_of_the_hill.analysis.zapping import get_zaps
from king_of_the_hill.arg_parser import get_evaluation_args
from king_of_the_hill.helpers import get_trailing_number, to_player_name
from gym.wrappers import RecordVideo
from datetime import datetime
from king_of_the_hill.wandb_api import get_api_key, download_evaluation_state, make_dir_if_not_exists
from king_of_the_hill.models import get_model
import os
import glob
import json
import warnings
import pickle

# Ignoring deprecations because 9/10 times it is 3rd libraries complaining about each other.
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def get_eval_path(run_config: Dict) -> str:
    return f"evaluation/{run_config['eval_folder']}/{run_config['rllib_run_name']}".replace('//',
                                                                                            '/')  # replace extra / if eval_folder is not used.


def evaluate(run_config: Dict):
    # TODO use workers for evaluation
    ray.init(local_mode=run_config['debug_mode'])  # Run with local_mode=True to use debugger
    rllib_path = run_config['rllib_path']
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    multi_checkpoint = bool(run_config['checkpoint_range'])

    # Pick either one or many checkpoints based on arg
    if multi_checkpoint:
        checkpoint_nos = list(range(*run_config['checkpoint_range']))
    else:
        checkpoint_nos = [run_config['checkpoint_number']]

    checkpoint_folders = [f"checkpoint_{c:06}" for c in checkpoint_nos]
    checkpoint_names = [f"checkpoint-{c}" for c in checkpoint_nos]

    api_key = get_api_key()
    checkpoint_paths, config, run_config = download_evaluation_state(rllib_path=rllib_path,
                                                                     run_config=run_config,
                                                                     api_key=api_key,
                                                                     checkpoint_folders=checkpoint_folders)
    print(f"Evaluating run: {run_config['rllib_run_name']}, id: {run_config['run_id']}")

    for checkpoint_path, checkpoint_name in zip(checkpoint_paths, checkpoint_names):
        print(f'\nEvaluating {checkpoint_name}...')
        agent_path = f"{checkpoint_path}/{checkpoint_name}"

        base_eval_path = get_eval_path(run_config)
        if multi_checkpoint:
            eval_path = f"{base_eval_path}/checkpoint-range-{checkpoint_nos[0]}-{checkpoint_nos[-1]}-{timestamp}/{checkpoint_name}"
        else:
            eval_path = f"{base_eval_path}/{checkpoint_name}-{timestamp}"
        evaluate_agent(agent_path=agent_path, eval_path=eval_path, config=config, run_config=run_config)

    ray.shutdown()


def evaluate_agent(agent_path: str, eval_path: str, config: Dict, run_config: Dict):
    agent = PPO(config=config)
    # If we have a lot of policies RLlib will dump a huge number of pickled policy files (.pkl) into the running directory.
    agent.restore(agent_path)

    if run_config['multi_policy']:
        for pair in run_config['multi_policy']:
            evaluate_policies(agent=agent,
                              eval_path=eval_path,
                              policy_names=pair,
                              config=config,
                              run_config=run_config)
    else:
        policy_names = list(agent.workers.local_worker().policy_dict.keys())
        num_players = config['env_config']['num_players']
        if len(policy_names) != num_players:
            raise Exception(
                f'Expected to have exactly 1 policy per player but have {num_players} players with policies {policy_names}')
        evaluate_policies(agent=agent,
                          eval_path=eval_path,
                          policy_names=policy_names,
                          config=config,
                          run_config=run_config)


def evaluate_policies(agent: PPO,
                      policy_names: List[str],
                      eval_path: str,
                      config: Dict,
                      run_config: Dict):
    print(f'Evaluating policies: {policy_names}')
    matchup_path = f"{eval_path}/{'__VS__'.join(policy_names)}"
    player_names = [to_player_name(p) for p in policy_names]

    if run_config['activations']:
        activation_extractor = ActivationExtractor()
        activation_extractor.register_activation_hooks(agent, run_config['activations'])

    model_description = get_model(run_config['model_name'])
    env_constructor = model_description['env_constructor']
    test_env = env_constructor(config["env_config"])
    test_env.metadata['video.frames_per_second'] = run_config['recording_fps']
    video_path = f"{matchup_path}/videos"
    make_dir_if_not_exists(matchup_path)
    if run_config['record_episodes']:
        # FIXME make this depend on env setting rather than a hard-coded value
        test_env = RecordVideo(test_env, video_path, video_length=1000, name_prefix='')  # 1000 steps is one episode

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
        'orientations': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: current position (col,row)}
        'distances': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: {other player #: distance}}
        'deaths': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: other players they are zapping}
        'zaps': [],
        # nested array with the following dimensions: [episode #], [timestep t], {player #: whether they captured the hill}
        'captures': [],
        # nested array with the following dimensions: [episode #], {player #: [timestep t] [activations]}
        'activations': []
        # TODO painting hill event
    }

    for episode in range(run_config['episodes']):
        print(f'==Episode {episode}==')
        done = False
        obs = test_env.reset()

        ep_rewards = []
        ep_actions = []
        ep_visibilities = []
        ep_positions = []
        ep_orientations = []
        ep_deaths = []
        ep_zaps = []
        ep_distances = []
        ep_captures = []

        # If using RNN/LSTM, we need to track more information
        if config['model']['use_lstm']:
            prev_actions = {}
            states = {}
            for player in player_names:
                # Initial hidden and cell state
                states[player] = [np.zeros(config['model']['lstm_cell_size'], ) for _ in range(2)]
                prev_actions[player] = np.zeros(1, dtype='int32')

        if run_config['record_observations']:
            observation_recorders = {player: ImageEncoder(
                output_path=f"{video_path}/episode-{episode * 1000}-{player}.mp4",
                frame_shape=list(obs.values())[0].shape,
                frames_per_sec=run_config['recording_fps'],
                output_frames_per_sec=run_config['recording_fps']) for player in player_names}
        while not done:
            actions = {}
            # Sorted to ensure we always iterate policy ids in the same order
            for policy, player in zip(policy_names, player_names):

                # If storing activations, indicate the current player
                if run_config['activations']:
                    activation_extractor.set_current_player(player)

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

            obs, reward, dones, infos = test_env.step(actions)

            deaths = {p: (np.max(ob) == 0).item() for p, ob in obs.items()}

            ep_rewards.append({p: float(r.item()) for p, r in reward.items()})  # Map numpy array value to regular float
            ep_actions.append({p: int(a.item()) for p, a in actions.items()})
            ep_visibilities.append(get_visibilities(infos, deaths))
            ep_positions.append({p: v['POSITION'].tolist() for p, v in infos.items()})
            ep_orientations.append({p: v['ORIENTATION'].tolist() for p, v in infos.items()})
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

        if run_config['activations']:
            len(activation_extractor.activations[player_names[0]])
            ep_activations = {}
            for player in player_names:
                ep_activations[player] = activation_extractor.activations[player]
            activation_extractor.clear_activations()
            metrics['activations'].append(ep_activations)

        metrics['rewards'].append(ep_rewards)
        metrics['actions'].append(ep_actions)
        metrics['visibilities'].append(ep_visibilities)
        metrics['positions'].append(ep_positions)
        metrics['orientations'].append(ep_orientations)
        metrics['deaths'].append(ep_deaths)
        metrics['zaps'].append(ep_zaps)
        metrics['distances'].append(ep_distances)
        metrics['captures'].append(ep_captures)

        if run_config['record_episodes']:
            test_env.close_video_recorder()

        # Record metrics every few eval steps. Just in case we interrupt evaluation at some point
        if episode % 5 == 0 or episode == run_config['episodes'] - 1:
            with open(f'{matchup_path}/metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_eval_and_clean(args: dict):
    try:
        evaluate(args)
    # RLlib will sometimes dump a huge number of .pkl files into the running directory. We don't want to keep these.
    finally:
        cwd = os.getcwd()
        policy_file_list = glob.glob(f'{cwd}/*.pkl')
        for path in policy_file_list:
            os.remove(path)


if __name__ == "__main__":
    args = vars(get_evaluation_args())
    run_eval_and_clean(args)
