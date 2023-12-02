import copy
import time
from typing import Dict, List, Tuple
import numpy as np
import dm_env
import ray
from ray.rllib import Policy
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

from datetime import datetime
import os
import glob
import warnings
import pickle
import wandb

from environment.helpers import validate_environment_roles
from evaluation.observation_recorder import ObservationRecorder
from src.environment.configs import PLAYABLE_MAPS, ENV_EVAL_METRIC_FUNCS, MELTING_POT_CONFIG_FUNCS
from src.environment.wrappers import RenderCropper
from src.evaluation.activations import ActivationExtractor, NeuronScorer
from src.evaluation.arg_parser import get_evaluation_args
from src.models.model_catalog import MODELS
from src.utils.helpers import get_trailing_number, to_player_name, get_world_rgb_bounding_box, \
    remove_prefix_from_files_in
from src.utils.bot_eval_helpers import load_bot_configs, load_bot_policies, Bots_Stepper, CustomRecordVideo as RecordVideo
from src.utils.wandb_api import download_evaluation_state, make_dir_if_not_exists
from utils.helpers import log_verbose

# Ignoring deprecations because 9/10 times it is 3rd libraries complaining about each other.
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_eval_path(run_config: Dict, env_name: str, timestamp: str = '') -> str:
    # replace extra / if eval_folder is not used.
    return f"eval_games/{env_name}/{run_config['eval_folder']}/{timestamp}".replace('//', '/')


def evaluate(run_config: Dict):
    # TODO use workers for evaluation
    ray.init(local_mode=run_config['debug_mode'])  # Run with local_mode=True to use debugger
    rllib_path = run_config['rllib_path']
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

    agent_paths = []
    agent_dicts = []

    # Extract meltingpot bot names if provided
    bot_names = []
    if run_config['meltingpot_bots']:
        for bot_config in run_config['meltingpot_bots']:
            bot_names.append(bot_config['bot_name'])

    for idx, player_info in enumerate(run_config['players']):
        checkpoint_folder = f"checkpoint_{player_info['checkpoint']:06}"
        checkpoint_name = f"checkpoint-{player_info['checkpoint']}"
        run_id = player_info['run_id']

        checkpoint_path, agent_config, agent_run_config = download_evaluation_state(rllib_path=rllib_path,
                                                                                    run_config=copy.deepcopy(
                                                                                        run_config),
                                                                                    run_id=run_id,
                                                                                    checkpoint_folder=checkpoint_folder)

        a_d = {'checkpoint_name': checkpoint_name,
               'rllib_run_name': agent_run_config['rllib_run_name'],
               'run_config': agent_run_config,
               'config': agent_config,
               'player': f'player_{idx}'
               } | player_info

        # Sanity-check that the role of the agent matches the role it was trained on.
        evaluation_role = training_role = a_d['role']
        if a_d['run_config']['model_sharing'] == 'independent':
            agent_training_idx = get_trailing_number(a_d['policy_id'])
            training_role = a_d['run_config']['agent_roles'][agent_training_idx]

        elif a_d['run_config']['model_sharing'] == 'one-per-role':
            training_role = a_d['policy_id'].split('_')[1]

        assert training_role == evaluation_role, \
            f'Expected training role {training_role} to match evaluation role {evaluation_role} for player {idx}'
        agent_dicts.append(a_d)
        agent_paths.append(checkpoint_path)

    model_names = {a_d['run_config']['model_name'] for a_d in agent_dicts}
    if len(model_names) != 1:
        raise NotImplementedError(
            'Presently, agents may only be evaluated against each other if they use the same model. '
            f'Found models: f{model_names}')

    env_names = list({a_d['run_config']['env'] for a_d in agent_dicts})
    if len(env_names) != 1:
        raise Exception(
            'Presently, agents may only be evaluated against each other if they use the same environment. '
            f'Found environments: f{env_names}')

    validate_environment_roles(env_names[0], [a_d['role'] for a_d in agent_dicts])

    base_eval_path = get_eval_path(run_config=run_config, timestamp=timestamp, env_name=agent_run_config['env'])
    evaluate_agents(agent_paths=agent_paths,
                    agent_dicts=agent_dicts,
                    eval_path=base_eval_path,
                    bot_names=bot_names)

    ray.shutdown()
    wandb.finish()


def evaluate_agents(agent_paths: List[str],
                    agent_dicts: List[Dict],
                    eval_path: str,
                    bot_names: List[str]):
    
    # Load meltingpot bots if provided
    bots = load_bot_policies(bot_names)
    bot_dicts = load_bot_configs(bot_names, starting_idx=len(agent_paths))

    # Restore agent policies
    agents: List[Policy] = []
    for agent_path, agent_dict in zip(agent_paths, agent_dicts):
        policy = Policy.from_checkpoint(f"{agent_path}/{agent_dict['policy_id']}")
        # If we have a lot of policies RLlib will dump a huge number of pickled policy files (.pkl) into the running directory.
        agents.append(policy)

    current_time = time.time()
    run_config = agent_dicts[0]['run_config']
    activation_paths = run_config['activations']
    score_activations = run_config['score_for_dormant_activations']

    # We regenerate the environment configuration here as both agents may not have been trained on the exact same
    # environment. We use the map and extra settings of the first agent, but the roles are obtained from each
    # agent's actual role they had during training.
    map_string = PLAYABLE_MAPS[run_config['env']][run_config['map_name']]['map']
    # agent_roles = [a['run_config']['agent_roles'][get_trailing_number(a['policy_id'])] for a in agent_dicts]
    env_config = MELTING_POT_CONFIG_FUNCS[run_config['env']](
        roles=[a_d['role'] for a_d in agent_dicts] + [b_d['role'] for b_d in bot_dicts],
        play_map=map_string,
        env_settings=run_config['env_settings'])

    # Assume agents have the same model for now
    model_description = MODELS[agent_dicts[0]['run_config']['model_name']]
    recording_fps = run_config['recording_fps']
    record_episodes = run_config['record_episodes']
    record_observations = run_config['record_observations']
    episode_length = agent_dicts[0]['config']['max_episode_steps']
    env_constructor = model_description['env_constructor']
    num_episodes = run_config['episodes']
    verbose = run_config['verbose']
    env_name = run_config['env']

    policy_names = [f"{a['rllib_run_name']}.{a['checkpoint_name']}.{a['policy_id']}" for a in agent_dicts] + \
        [f"{b['name']}" for b in bot_dicts]
    print(f"Evaluating policies: {', '.join(policy_names)} in {env_name}")
    matchup_path = f"{eval_path}/{'__VS__'.join(policy_names)}"
    player_names = [a_d['player'] for a_d in agent_dicts] 
    bot_names = [b_d['player'] for b_d in bot_dicts]
    player_and_bot_names = player_names + bot_names

    # If set, store activations from final layers
    if activation_paths:
        activation_extractor = ActivationExtractor()
        for a, d in zip(agents, agent_dicts):
            activation_extractor.register_activation_hooks(a, activation_paths)

    # If set, score activations for every neuron in net, relative to layer
    if score_activations:
        neuron_scorer = NeuronScorer()
        for a, d in zip(agents, agent_dicts):
            neuron_scorer.register_activation_hooks(a.get_policy(d['policy_id']).model, d['player'])

    test_env = env_constructor(env_config)
    test_env.metadata['render_fps'] = recording_fps
    video_path = f"{matchup_path}/videos"
    make_dir_if_not_exists(matchup_path)
    if record_episodes:
        test_env = RenderCropper(test_env)
        test_env = RecordVideo(test_env, video_path, video_length=episode_length, name_prefix='', disable_logger=True)

    # General evaluation metrics to track
    metrics = {
        # nested array with the following dimensions: [episode #], [timestep t], {player #: action at timestep t}
        'rewards': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: reward at timestep t}
        'actions': [],
        # nested array with following dimensions: [episode #], [timestep t], {player #: {other player #: distance}}
        'deaths': [],
        # nested array with the following dimensions: [episode #], {player #: [timestep t] [activations]}
        'activations': [],
        # for other information that does not need to be recorded per episode
        'other': {},
        # activation scores for each neuron relative to its layer (used to determine dormant neurons later)
        'relative_activation_scores': {}
    }
    # Metrics specific to the environment
    env_metrics = {}

    # Copy all agent info that is simple string identifiers.
    metrics['other']['agent_info'] = [{k: v for k, v in a_d.items() if isinstance(v, str)} for a_d in
                                      agent_dicts]

    # Get bounding box around environment
    obs_x, obs_y, obs_width, obs_height = get_world_rgb_bounding_box(env_constructor(env_config))
    metrics['other']['observation_position_px'] = {'x': obs_x, 'y': obs_y, 'width': obs_width, 'height': obs_height}

    metrics['other']['map_string'] = map_string

    if activation_paths:
        metrics['other']['activation_paths'] = activation_paths

    #a special timestep object used for bots
    timestep : dm_env.TimeStep = None

    for episode in range(num_episodes):
        print(f'==Episode {episode}==')
        done = False
        obs, _, timestep = test_env.reset()

        # init a class instance for stepping meltingpot provided bots
        bots_stepper = Bots_Stepper(
            bots=bots, 
            bot_configs=bot_dicts, 
            starting_idx=len(agent_dicts))

        if record_episodes:
            test_env.start_video_recorder()

        episode_metrics = {
            'rewards': [],
            'actions': [],
            'deaths': [],
        }
        env_episode_metrics = {}

        # If using RNN/LSTM, we need to track more information
        if any(isinstance(agent.model, RecurrentNetwork) for agent in agents):
            prev_actions = {}
            states = {}
            for agent, player in zip(agents, player_names):
                # Initial hidden and cell state
                states[player] = agent.model.get_initial_state()
                prev_actions[player] = 0

        if record_observations:
            observation_recorder = ObservationRecorder(video_folder=video_path,
                                                       video_prefix=f'episode{episode}',
                                                       frame_shapes={p: obs[p].shape for p in player_and_bot_names},
                                                       frames_per_sec=recording_fps)
        while not done:
            actions = {}
            for agent, a_dict in zip(agents, agent_dicts):
                player = a_dict['player']
                # If storing activations, indicate the current player
                if activation_paths:
                    activation_extractor.set_current_player(player)

                # If scoring activations, indicate the current player
                if score_activations:
                    neuron_scorer.set_current_player(player)

                # If using RNN/LSTM we need to keep track of hidden states
                if isinstance(agent.model, RecurrentNetwork):
                    action, state_out, _ = agent.compute_single_action(obs=obs[player],
                                                                       state=states[player],
                                                                       prev_action=prev_actions[player])
                    states[player] = state_out
                    prev_actions[player] = action
                else:
                    action = agent.compute_single_action(obs[player], policy_id=a_dict['policy_id'])

                log_verbose(f"{player}, {a_dict['policy_id']}, {action}", verbose)
                actions[player] = action

            # Step bots
            bots_stepper.step(timestep)
            actions.update(bots_stepper.get_actions())
            states.update(bots_stepper.get_states())

            obs, reward, terminated, truncated, infos, timestep = test_env.step(actions)

            deaths = {p: (np.max(ob) == 0).item() for p, ob in obs.items()}

            # General step metrics
            episode_metrics['rewards'].append({p: float(r.item()) for p, r in reward.items()})
            episode_metrics['actions'].append({p: int(a.item()) if isinstance(a, np.int32) else a for p, a in
                                               actions.items()})
            episode_metrics['deaths'].append(deaths)  # if a player is given an empty obs, it has recently been killed

            # Environment step metrics
            ENV_EVAL_METRIC_FUNCS[env_name]['episode_step'](env_episode_metrics=env_episode_metrics,
                                                            episode_metrics=episode_metrics,
                                                            general_metrics=metrics,
                                                            obs=obs,
                                                            actions=actions,
                                                            rewards=reward,
                                                            infos=infos)

            done = all(terminated.values()) or all(truncated.values())

            if record_episodes:
                test_env.video_recorder.capture_frame()

            if record_observations:
                observation_recorder.capture_frame(obs)

        if record_observations:
            observation_recorder.close()

        if activation_paths:
            len(activation_extractor.activations[player_names[0]])
            ep_activations = {}
            for player in player_names:
                ep_activations[player] = activation_extractor.activations[player]
            activation_extractor.clear_activations()
            metrics['activations'].append(ep_activations)

        if score_activations:
            metrics['relative_activation_scores'] = neuron_scorer.activation_scores

        # General episode metrics
        metrics['rewards'].append(episode_metrics['rewards'])
        metrics['actions'].append(episode_metrics['actions'])
        metrics['deaths'].append(episode_metrics['deaths'])

        # Environment episode metrics
        ENV_EVAL_METRIC_FUNCS[env_name]['episode_end'](env_metrics=env_metrics,
                                                       general_metrics=metrics,
                                                       env_episode_metrics=env_episode_metrics,
                                                       episode_metrics=episode_metrics)

        if record_episodes:
            test_env.close_video_recorder()
            # minor cleanup as gymnasium adds a '-' even if there is no prefix
            remove_prefix_from_files_in(folder=video_path, prefix='-')

        all_metrics = {'general_metrics': metrics, 'env_metrics': env_metrics}
        # Record metrics every few eval steps. Just in case we interrupt evaluation at some point
        if episode % 5 == 0 or episode == num_episodes - 1:
            with open(f'{matchup_path}/metrics.pkl', 'wb') as f:
                pickle.dump(all_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

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
