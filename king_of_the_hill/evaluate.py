import copy
import time
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
from king_of_the_hill.helpers import get_trailing_number, to_player_name, get_world_rgb_bounding_box, find_hill_center, \
    get_positions
from gym.wrappers import RecordVideo
from datetime import datetime
from king_of_the_hill.maps import PLAYABLE_MAPS
from king_of_the_hill.wandb_api import get_api_key, download_evaluation_state, make_dir_if_not_exists
from king_of_the_hill.models import get_model, MODELS
import os
import glob
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


def get_eval_path(run_config: Dict, timestamp: str = '') -> str:
    return f"evaluation/{run_config['eval_folder']}/{timestamp}".replace('//',
                                                                         '/')  # replace extra / if eval_folder is not used.

def evaluate(run_config: Dict):
    # TODO use workers for evaluation
    ray.init(local_mode=run_config['debug_mode'])  # Run with local_mode=True to use debugger
    rllib_path = run_config['rllib_path']
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

    agent_paths = []
    agent_dicts = []

    for policy_info in run_config['policies']:
        checkpoint_folder = f"checkpoint_{policy_info['checkpoint']:06}"
        checkpoint_name = f"checkpoint-{policy_info['checkpoint']}"
        run_id = policy_info['run_id']

        api_key = get_api_key()

        checkpoint_path, agent_config, agent_run_config = download_evaluation_state(rllib_path=rllib_path,
                                                                                    run_config=copy.deepcopy(
                                                                                        run_config),
                                                                                    run_id=run_id,
                                                                                    api_key=api_key,
                                                                                    checkpoint_folder=checkpoint_folder)
        agent_path = f"{checkpoint_path}/{checkpoint_name}"
        agent_dicts.append({
                               'checkpoint_name': checkpoint_name,
                               'rllib_run_name': agent_run_config['rllib_run_name'],
                               'run_config': agent_run_config,
                               'config': agent_config,
                           } | policy_info)
        agent_paths.append(agent_path)

    model_names = {a_d['run_config']['model_name'] for a_d in agent_dicts}
    if len(model_names) != 1:
        raise NotImplementedError(
            'Presently, agents may only be evaluated against each other if they use the same model. '
            f'Found models: f{model_names}')

    base_eval_path = get_eval_path(run_config=run_config, timestamp=timestamp)
    evaluate_agents(agent_paths=agent_paths,
                    agent_dicts=agent_dicts,
                    eval_path=base_eval_path)

    ray.shutdown()


def evaluate_agents(agent_paths: List[str],
                    agent_dicts: List[Dict],
                    eval_path: str):
    # Restore agents
    loaded_agents = {}
    agents = []
    # TODO we should try to save some memory by dropping unused policies from agent mappings
    for agent_path, agent_dict in zip(agent_paths, agent_dicts):
        # Often we have policies from the same agent play against each other.
        # Instantiating the same policies multiple times is wasteful, so we check if the agent has already been loaded.
        if agent_path in loaded_agents:
            agents.append(loaded_agents[agent_path])
        else:
            agent = PPO(config=agent_dict['config'])
            # If we have a lot of policies RLlib will dump a huge number of pickled policy files (.pkl) into the running directory.
            agent.restore(agent_path)
            agents.append(agent)
            loaded_agents[agent_path] = agent

    current_time = time.time()
    activations_names = agent_dicts[0]['run_config']['activations']
    env_config = agent_dicts[0]['config']['env_config']
    # Assume agents have the same model for now
    model_description = MODELS[agent_dicts[0]['run_config']['model_name']]
    recording_fps = agent_dicts[0]['run_config']['recording_fps']
    record_episodes = agent_dicts[0]['run_config']['record_episodes']
    record_observations = agent_dicts[0]['run_config']['record_observations']
    episode_length = agent_dicts[0]['config']['horizon']
    env_constructor = model_description['env_constructor']
    map_string = PLAYABLE_MAPS[agent_dicts[0]['run_config']['map_name']]['map']
    num_episodes = agent_dicts[0]['run_config']['episodes']
    verbose = agent_dicts[0]['run_config']['verbose']

    policy_names = [f"{a['rllib_run_name']}.{a['checkpoint_name']}.{a['policy_id']}" for a in agent_dicts]
    print(f"Evaluating policies: {','.join(policy_names)}")
    matchup_path = f"{eval_path}/{'__VS__'.join(policy_names)}"
    player_names = []
    for agent_dict in agent_dicts:
        agent_dict['player'] = to_player_name(agent_dict['policy_id'])
        player_names.append(agent_dict['player'])

    if activations_names:
        activation_extractor = ActivationExtractor()
        for a, d in zip(agents, agent_dicts):
            activation_extractor.register_activation_hooks(a, d['policy_id'], activations_names)

    test_env = env_constructor(env_config)
    test_env.metadata['video.frames_per_second'] = recording_fps
    video_path = f"{matchup_path}/videos"
    make_dir_if_not_exists(matchup_path)
    if record_episodes:
        test_env = RecordVideo(test_env, video_path, video_length=episode_length, name_prefix='')

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
        'activations': [],
        # for other information that does not need to be recorded per episode
        'other': {}
        # TODO painting hill event
    }

    # Copy all agent info that is simple string identifiers.
    metrics['other']['agent_info'] = [{k: v for k, v in a_d.items() if isinstance(v, str)} for a_d in
                                      agent_dicts]

    # Get bounding box around environment
    obs_x, obs_y, obs_width, obs_height = get_world_rgb_bounding_box(env_constructor(env_config))
    metrics['other']['observation_position_px'] = {'x': obs_x, 'y': obs_y, 'width': obs_width, 'height': obs_height}

    # Get hill position of the environment
    hill_center_x, hill_center_y = find_hill_center(map_string=map_string)
    metrics['other']['hill'] = {'center_x': hill_center_x, 'center_y': hill_center_y}

    for episode in range(num_episodes):
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
        if model_description['model_config']['use_lstm']:
            prev_actions = {}
            states = {}
            for player in player_names:
                # Initial hidden and cell state
                states[player] = [np.zeros(model_description['model_config']['lstm_cell_size'], ) for _ in range(2)]
                prev_actions[player] = np.zeros(1, dtype='int32')

        if record_observations:
            observation_recorders = {player: ImageEncoder(
                output_path=f"{video_path}/episode-{episode * 1000}-{player}.mp4",
                frame_shape=list(obs.values())[0].shape,
                frames_per_sec=recording_fps,
                output_frames_per_sec=recording_fps) for player in player_names}
        while not done:
            actions = {}
            for agent, a_dict in zip(agents, agent_dicts):
                player = a_dict['player']
                # If storing activations, indicate the current player
                if activations_names:
                    activation_extractor.set_current_player(player)

                # If using RNN/LSTM we need to keep track of hidden states
                if model_description['model_config']['use_lstm']:
                    action, state_out, _ = agent.compute_single_action(observation=obs[player],
                                                                       state=states[player],
                                                                       prev_action=prev_actions[player],
                                                                       policy_id=a_dict['policy_id'])
                    states[player] = state_out
                    prev_actions[player] = action
                else:
                    action = agent.compute_single_action(obs[player], policy_id=a_dict['policy_id'])

                log_verbose(f"{player}, {a_dict['policy_id']}, {action}", verbose)
                actions[player] = action

            obs, reward, dones, infos = test_env.step(actions)

            deaths = {p: (np.max(ob) == 0).item() for p, ob in obs.items()}

            ep_rewards.append({p: float(r.item()) for p, r in reward.items()})  # Map numpy array value to regular float
            ep_actions.append({p: int(a.item()) for p, a in actions.items()})
            ep_visibilities.append(get_visibilities(infos, deaths))
            ep_positions.append(get_positions(infos, obs_x, obs_y))
            ep_orientations.append({p: v['ORIENTATION'].tolist() for p, v in infos.items()})
            ep_deaths.append(deaths)  # if a player is given an empty obs, it has recently been killed
            ep_zaps.append(get_zaps(infos, actions))
            ep_distances.append(get_distances(infos, deaths))
            ep_captures.append(get_hill_captures(ep_rewards))

            done = all(dones.values())

            if record_observations:
                for player in obs:
                    observation_recorders[player].capture_frame(obs[player])

        if record_observations:
            for player in observation_recorders:
                observation_recorders[player].close()

        if activations_names:
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

        if record_episodes:
            test_env.close_video_recorder()

        # Record metrics every few eval steps. Just in case we interrupt evaluation at some point
        if episode % 5 == 0 or episode == num_episodes - 1:
            with open(f'{matchup_path}/metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Finished pair evaluation in {time.time() - current_time:.2f}s')


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
