from collections import defaultdict
from copy import copy
from typing import Dict, Any, List, Tuple
import ray
import numpy as np
from ray.actor import ActorHandle
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.evaluation import Episode
from scipy.special import softmax
from scipy.stats import zscore

from utils.helpers import get_player_idx

"""
This module allows training agents in a population together. It works by maintaining a training population of k policies
where each policy is trained against frozen variants of other policies in the population. The frozen policies are
generated once per curriculum step. The conditions for a curriculum step to end are based on whether the enough policies
in the training population have improved sufficiently. 
"""

@ray.remote
class PopulationManager:
    """
    This class is responsible for managing the population of agents during training.
    """

    def __init__(self, config: Dict[str, Any], run_config: Dict[str, Any]):
        assert run_config[
                   'model_sharing'] == 'independent', 'PopulationManager only supports independent model sharing'

        # TODO verify num episodes per training iteration is divisible by population per
        population_per = run_config['population_self_play']['population_per']
        self.num_players = len(run_config['agent_roles'])
        assert self.num_players == 2, 'PopulationManager only supports 2 player games'
        self.min_games_against_new_opponent = run_config['population_self_play']['min_games_against_new_opponent']
        self.recent_reward_window = run_config['population_self_play']['recent_reward_window']
        self.policy_improvement_factor = run_config['population_self_play']['policy_improvement_factor']
        self.sufficient_improved_population_ratio = run_config['population_self_play'][
            'sufficient_improved_population_ratio']
        self.curriculum_end_timesteps = run_config['population_self_play']['curriculum_end_timesteps']

        # Set up the initial populations of policies
        # Training: all policies that should be trained
        # Frozen: policies that are frozen and never trained
        # Paused: policies that are temporarily frozen and will resume training later
        # Stopped: policies that have been removed from training and are no longer used
        self.training_population = {f'policy_{p}_p{i}' for p in range(self.num_players) for i in range(population_per)}
        self.frozen_population = set()
        self.paused_training_population = set()
        self.stopped_training_population = set()

        # dict of {trainable policy_id: {frozen policy_id: [float]}} # reward for games played against each opponent
        self.play_stats = {p: defaultdict(list) for p in self.training_population}
        self.training_episodes = {}  # dict of {episode_id: {agent_id: policy_id}}
        self.curriculum_iteration = 1  # which iteration of frozen policies we are on
        # Tracks the rewards obtained per curriculum iteration
        self.curriculum_rewards = {self.curriculum_iteration: {p: [] for p in self.training_population}}
        # the estimated initial rewards a policy has against every opponent at the start of a curriculum iteration
        # dict of {curriculum_iteration: {trainable policy_id: {frozen policy_id: [float]}}}
        self.curriculum_initial_reward_estimates = {self.curriculum_iteration: {}}
        # the final training step for the curriculum iteration
        self.curriculum_ends = {}
        # number of games each policy played per curriculum iteration
        self.curriculum_games_played = {self.curriculum_iteration: {p: 0 for p in self.training_population}}

        self.evaluation_episodes = {}  # dict of {episode_id: {agent_id: policy_id}}}
        # dict of {trainable policy_id: {frozen policy_id: [float]}}
        self.evaluation_stats = {p: defaultdict(list) for p in self.training_population}
        # List of dicts of {players: {agent_id: policy_id}, state: 'pending'/'ongoing'/'complete', episode: episode_id}
        self.evaluation_pairings = []
        # For some reason Rllib oversamples episodes during evaluation so we need to keep track of "dud" episodes
        # that we don't want to measure.
        self.dud_episodes = set()

        # TODO is there some way to update the number of episodes per training iteration during training?
        # TODO may be possible via algorithm.merge_algorithm_configs()

    def filter_by_in_training(self, d: Dict[str, Any]):
        """
        Filters a dictionary of policies by those that are in training.
        @param d:
        @return:
        """
        result = {k: v for k, v in d.items() if k in self.training_population}
        # assert len(result) > 0, f'No policies in training according to {d} and {self.training_population}'
        return result

    def get_training_matchup(self, episode_id: str, agent_id: str) -> str:
        """
        For a given episode, yield a player policy configuration (generating one if it doesn't exist).
        @param episode_id:
        @param agent_id:
        @return:
        """
        if episode_id not in self.training_episodes:
            games_per_active_policy = self.filter_by_in_training(
                self.curriculum_games_played[self.curriculum_iteration])

            # Randomly pick a policy to train with the lowest number of games played
            min_games = min(games_per_active_policy.values())
            policies_with_min_games = [p for p, games in games_per_active_policy.items() if games == min_games]
            main_policy = np.random.choice(policies_with_min_games)

            # Choose the player corresponding to the policy
            main_agent_idx = get_player_idx(main_policy)
            self.training_episodes[episode_id] = {}
            self.training_episodes[episode_id][f'player_{main_agent_idx}'] = main_policy

            # Assign a policy for each other agent
            for idx in range(self.num_players):
                if idx == main_agent_idx:
                    continue
                self.training_episodes[episode_id][f'player_{idx}'] = self.sample_opponent_for(main_policy, idx)

        return self.training_episodes[episode_id][agent_id]

    def generate_evaluation_matchups(self):
        for trainable_policy in self.training_population:
            for frozen_policy in self.evaluation_stats[trainable_policy]:
                eval_games_to_play = self.min_games_against_new_opponent - len(
                    self.evaluation_stats[trainable_policy][frozen_policy])
                if eval_games_to_play > 0:
                    self.evaluation_pairings += [{'players':
                                                      {f'player_{get_player_idx(trainable_policy)}': trainable_policy,
                                                       f'player_{get_player_idx(frozen_policy)}': frozen_policy},
                                                  'episode': None,
                                                  'state': 'pending'
                                                  } for i in range(eval_games_to_play)]

    def get_num_evaluations_in_state(self, state: str) -> int:
        return len([e for e in self.evaluation_pairings if e['state'] == state])

    def get_evaluation_matchup(self, episode_id: str, agent_id: str) -> str:
        """
        For a given episode, yield a player policy configuration (generating one if it doesn't exist).
        @param episode_id:
        @param agent_id:
        @return:
        """
        if episode_id not in self.evaluation_episodes:
            # If Rllib oversamples, yield main ids and mark the episode as a dud.
            if self.get_num_evaluations_in_state('pending') == 0:
                # get first two frozen policies for a dummy game
                self.evaluation_episodes[episode_id] = {f'player_{idx}': policy_id for idx, policy_id in
                                                        enumerate(list(self.frozen_population)[:2])}
                self.dud_episodes.add(episode_id)
            else:
                first_free_pairing = next(e for e in self.evaluation_pairings if e['state'] == 'pending')
                self.evaluation_episodes[episode_id] = first_free_pairing['players']
                first_free_pairing['state'] = 'ongoing'
                first_free_pairing['episode'] = episode_id
        return self.evaluation_episodes[episode_id][agent_id]

    def get_sampling_probabilities(self, main_policy_id: str, opponent_player_idx: int) -> Tuple[
        List[str], np.ndarray, np.ndarray, float]:
        """
        Determines the sampling probabilities based on the recent rewards of the main policy vs all opponents.
        @param main_policy_id:
        @param opponent_player_idx:
        @return: opponent policy ids, reward estimate against opponents, sampling probabilities, reward_offset
        """
        policies_matching_role = [p for p in self.frozen_population if get_player_idx(p) == opponent_player_idx]

        policy_rewards = [self.curriculum_initial_reward_estimates[self.curriculum_iteration][main_policy_id][p] for p
                          in policies_matching_role]
        # Offset rewards to be non-negative. It makes other calculations easier.
        min_reward = min(policy_rewards)
        reward_offset = min_reward * -1 if min_reward < 0 else 0
        policy_weights = np.array(policy_rewards) + reward_offset
        # TODO consider multiple implementation variants for sampling opponents
        probs = self.weight_by_worst_rewards(policy_weights)
        return policies_matching_role, policy_weights, probs, reward_offset

    def weight_by_worst_rewards(self, policy_rewards: List[float]) -> np.ndarray:
        """
        Returns an array probability weights which favour policy matchups with the worst rewards.
        @param policy_rewards:
        @return:
        """
        # TODO add softmax temperature as a parameter.
        # TODO See: https://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html
        return softmax(-zscore(policy_rewards))

    def sample_opponent_for(self, main_policy_id: str, opponent_player_idx: int) -> str:
        """
        Samples an opponent for a given the main policy and desired opponent_player_idx.
        @param main_policy_id
        @param opponent_player_idx:
        @return: policy ID of the opponent
        """
        policies_matching_role, _, probabilities, _ = self.get_sampling_probabilities(main_policy_id,
                                                                                      opponent_player_idx)
        if any(np.isnan(p) for p in probabilities):
            return np.random.choice(policies_matching_role)
        return np.random.choice(policies_matching_role, p=probabilities)

    def add_new_frozen_policy(self, policy_id: str):
        """
        Adds a new frozen policy to the frozen opponent population.
        @param policy_id:
        @return:
        """
        self.frozen_population.add(policy_id)
        for p in self.training_population:
            # Only create matchups for policies that play different roles
            if get_player_idx(p) != get_player_idx(policy_id):
                self.play_stats[p][policy_id] = []
                self.evaluation_stats[p][policy_id] = []

    def apply_episode_result(self, episode_id: str, result: dict):
        """
        Updates play metrics based on the result of an episode.
        @param episode_id:
        @param result:
        @return:
        """
        # Case 1: dud episode
        if episode_id in self.dud_episodes:
            return

        policy_ids = [r[1] for r in result.keys()]
        matches = [p for p in policy_ids if p in self.training_population or p in self.paused_training_population]

        # Handle an edge case where a policy could be scheduled for an episode
        # but is then paused or stopped before the episode is played.
        if len(matches) == 0:
            stopped_policies = [p for p in policy_ids if p in self.stopped_training_population]
            if len(stopped_policies) == 1:
                print(f'Found a stopped policy {stopped_policies[0]} in episode {episode_id}. Dropping result')
                return
            else:
                raise Exception(f'Could not match episode {episode_id} with policies {policy_ids}'
                                f'to any training ({self.training_population}) '
                                f'or paused ({self.paused_training_population}) policies.')
        else:
            main_policy = matches[0]

        other_policies = [p for p in policy_ids if p not in self.training_population]
        main_policy_reward = result[(f'player_{get_player_idx(main_policy)}', main_policy)]

        # If training update curriculum metrics
        if episode_id in self.training_episodes:
            self.curriculum_games_played[self.curriculum_iteration][main_policy] += 1
            self.curriculum_rewards[self.curriculum_iteration][main_policy].append(main_policy_reward)
            for o in other_policies:
                self.play_stats[main_policy][o].append(main_policy_reward)
        # If evaluation update evaluation metrics
        elif episode_id in self.evaluation_episodes:
            for o in other_policies:
                self.evaluation_stats[main_policy][o].append(main_policy_reward)
            paired_episode = [e for e in self.evaluation_pairings if e['episode'] == episode_id][0]
            paired_episode['state'] = 'complete'
        else:
            raise Exception(f'Could not match episode {episode_id} with training or evaluation episodes.')

    def check_update_training_population(self):
        """
        Checks if policies in the training population have improved sufficiently to pause their training.
        @return:
        """

        curriculum_rewards = self.filter_by_in_training(self.curriculum_rewards[self.curriculum_iteration])

        # Check if any of the agents have improved sufficiently to pause their training.
        for policy_id, reward in curriculum_rewards.items():
            initial_reward, current_reward, required_reward = self.get_reward_targets(policy_id)
            # If the reward has improved enough, we pause the training for that policy.
            if (len(curriculum_rewards[policy_id]) >= self.recent_reward_window and
                    current_reward > required_reward):
                self.training_population.remove(policy_id)
                self.paused_training_population.add(policy_id)

    def get_reward_targets(self, policy_id: str) -> Tuple[float, float, float]:
        """
        Returns the reward targets for a policy in the curriculum
        @param policy_id:
        @return:
        """
        curriculum_rewards = self.filter_by_in_training(self.curriculum_rewards[self.curriculum_iteration])
        opponent_player_idx = [i for i in range(self.num_players) if i != get_player_idx(policy_id)][0]
        _, opponent_rewards, probabilities, reward_offset = self.get_sampling_probabilities(policy_id,
                                                                                            opponent_player_idx)
        initial_reward = opponent_rewards.dot(
            probabilities)  # get an estimate of the expected reward against all opponents
        # We compare the initial reward and requirement improvement to the average reward over the last episodes.
        # We have to ensure the reward is always non-negative.
        assert initial_reward >= 0, f'Initial reward for {policy_id} is negative: {initial_reward}'
        required_reward = initial_reward * self.policy_improvement_factor
        current_reward = np.average(curriculum_rewards[policy_id][-self.recent_reward_window:]) + reward_offset
        return initial_reward, current_reward, required_reward

    def calculate_initial_curriculum_reward_estimates(self):
        reward_estimates = {policy_id: defaultdict(float) for policy_id in self.training_population}
        for policy_id in self.training_population:
            player_idx = get_player_idx(policy_id)
            opponents = [p for p in self.frozen_population if get_player_idx(p) != player_idx]
            for opp in opponents:
                training_rewards = self.play_stats[policy_id][opp]
                evaluation_rewards = self.evaluation_stats[policy_id][opp]
                rewards = evaluation_rewards + training_rewards

                if len(rewards) < self.min_games_against_new_opponent:
                    raise Exception(f'{policy_id} has not played enough games played against opponent {opp} '
                                    f'to calculate reward estimate')
                reward_estimates[policy_id][opp] = np.average(rewards[-self.min_games_against_new_opponent:])

        self.curriculum_initial_reward_estimates[self.curriculum_iteration] = reward_estimates

    def check_should_end_curriculum_iteration(self, training_iteration: int) -> bool:
        """
        Checks if the training population has improved sufficiently to end the current curriculum iteration.
        @param training_iteration:
        @return:
        """

        # Check if all agents have improved sufficiently
        if len(self.training_population) == 0:
            return True

        num_finished_policies = len(self.paused_training_population)
        num_training_policies = len(self.training_population)
        ratio_finished = num_finished_policies / (num_finished_policies + num_training_policies)

        # If enough agents have improved sufficiently, set a timestep limit for the stragglers
        if self.sufficient_improved_population_ratio <= ratio_finished:
            if self.curriculum_iteration not in self.curriculum_ends:
                self.curriculum_ends[self.curriculum_iteration] = training_iteration + self.curriculum_end_timesteps
            else:
                return training_iteration >= self.curriculum_ends[self.curriculum_iteration]

        return False

    def increment_curriculum_iteration(self):
        """
        Increments the curriculum iteration.
        @return:
        """

        print(f'Incrementing curriculum iteration to {self.curriculum_iteration + 1}')
        print(f'Current training population: {self.training_population}, '
              f'Current paused population: {self.paused_training_population},'
              f'Current stopped population: {self.stopped_training_population}')
        # Prune any policies that have not met the training goal
        for p in self.training_population:
            self.stopped_training_population.add(p)
        # Return all paused policies to the training population
        self.training_population = copy(self.paused_training_population)
        self.paused_training_population = set()

        # Increment the curriculum iteration
        self.curriculum_iteration += 1
        self.curriculum_games_played[self.curriculum_iteration] = {p: 0 for p in self.training_population}
        self.curriculum_rewards[self.curriculum_iteration] = {p: [] for p in self.training_population}
        self.curriculum_initial_reward_estimates[self.curriculum_iteration] = {}

        # Clear evaluation matchups
        self.evaluation_pairings = []

        print(f'New training population: {self.training_population}, '
              f'New paused population: {self.paused_training_population},'
              f'New stopped population: {self.stopped_training_population}')

    def get_attr(self, param_name):
        return getattr(self, param_name)


class PopulationSelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.pop_manager = ray.get_actor('population_manager')

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """
        Creates the initial opponent population to train against.
        @param algorithm:
        @param kwargs:
        @return:
        """
        print('Initialising population-based self-play...')
        training_ids = ray.get(self.pop_manager.get_attr.remote('training_population'))
        # Set the initial frozen policies
        for t in training_ids:
            clone_and_add_policy(self.pop_manager, algorithm, t)
        print(f'Starting policies: {list_current_policies(algorithm)}')

        print(f'Evaluating current agents against initial opponent policies...')
        self._evaluate_new_agents(algorithm)
        ray.get(self.pop_manager.calculate_initial_curriculum_reward_estimates.remote())
        print('Evaluation complete.')

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        ray.get(self.pop_manager.check_update_training_population.remote())
        should_end_curriculum = ray.get(self.pop_manager.check_should_end_curriculum_iteration.remote(
            algorithm.training_iteration))
        if should_end_curriculum:
            ray.get(self.pop_manager.increment_curriculum_iteration.remote())
            curriculum_iteration = ray.get(self.pop_manager.get_attr.remote('curriculum_iteration'))
            print(f'Transitioning to curriculum iteration {curriculum_iteration}')
            training_ids = ray.get(self.pop_manager.get_attr.remote('training_population'))
            # Set the initial frozen policies
            print(f'Adding new frozen policies for {training_ids}')
            for t in training_ids:
                clone_and_add_policy(self.pop_manager, algorithm, t)

            print(f'Currently registered rllib policies: {list_current_policies(algorithm)}')

            print(f'Evaluating current agents against new frozen policies...')
            self._evaluate_new_agents(algorithm)
            ray.get(self.pop_manager.calculate_initial_curriculum_reward_estimates.remote())
            print('Evaluation complete.')

        metrics = calculate_metrics(self.pop_manager)

        # Add custom metrics
        for m in metrics:
            result[m] = metrics[m]

    def _evaluate_new_agents(self, algo: Algorithm):
        """
        Evaluate training agents against new opponents so we get a baseline reward for our sampling.
        @param algo:
        @return:
        """
        # We shouldn't be using regular workers for evaluation, but rllib just can't set the evaluation workers properly.
        eval_workers = algo.workers
        # Update eval workers with most recent training state.
        eval_workers.sync_weights(from_worker_or_learner_group=algo.workers.local_worker())

        # Generate eval games for workers to use
        ray.get(self.pop_manager.generate_evaluation_matchups.remote())

        # Set policy mapping function using evaluation for workers.
        eval_mapping_func = lambda w: w.set_policy_mapping_fn(evaluation_policy_mapping_fn)
        eval_workers.foreach_worker(eval_mapping_func)

        def evaluations_remaining(pm: ActorHandle) -> int:
            return (ray.get(pm.get_num_evaluations_in_state.remote('ongoing')) +
                    ray.get(pm.get_num_evaluations_in_state.remote('pending')))

        while evaluations_remaining(self.pop_manager) > 0:
            print(f'{evaluations_remaining(self.pop_manager)} evaluations remaining...')
            eval_workers.foreach_worker(func=lambda w: w.sample())

        training_mapping_func = lambda w: w.set_policy_mapping_fn(training_policy_mapping_fn)
        eval_workers.foreach_worker(training_mapping_func)

    def on_episode_end(self, **kwargs, ) -> None:
        episode: Episode = kwargs['episode']
        self.pop_manager.apply_episode_result.remote(episode.episode_id, episode.agent_rewards)


def training_policy_mapping_fn(agent_id: str, episode: Episode, **kwargs):
    pop_manager = ray.get_actor('population_manager')
    policy_id = ray.get(pop_manager.get_training_matchup.remote(episode.episode_id, agent_id))
    return policy_id


def evaluation_policy_mapping_fn(agent_id: str, episode: Episode, **kwargs):
    pop_manager = ray.get_actor('population_manager')
    policy_id = ray.get(pop_manager.get_evaluation_matchup.remote(episode.episode_id, agent_id))
    return policy_id


def clone_and_add_policy(pop_manager: ActorHandle, algorithm: Algorithm, main_policy_id: str):
    """
    Clones a policy and adds it to the algorithm. Note that the cloned policy is not added to the training population.
    Therefore, it is effectively frozen and will not be trained.
    @param pop_manager:
    @param algorithm:
    @param main_policy_id:
    @return:
    """
    curriculum_iteration = ray.get(pop_manager.get_attr.remote('curriculum_iteration'))
    new_pol_id = f"{main_policy_id}_v{curriculum_iteration}"
    pop_manager.add_new_frozen_policy.remote(new_pol_id)

    temp_env = algorithm.env_creator(algorithm.config['env_config'])
    main_state = algorithm.get_policy(main_policy_id).get_state()
    algorithm.add_policy(
        policy_id=new_pol_id,
        policy_cls=type(algorithm.get_policy(main_policy_id)),
        policy_state=main_state,
        observation_space=temp_env.observation_space[f'player_{get_player_idx(main_policy_id)}'],
        action_space=temp_env.action_space[f'player_{get_player_idx(main_policy_id)}'],
    )
    # We need to sync the just copied local weights to all the
    # remote workers as well.
    algorithm.workers.sync_weights(policies=[new_pol_id])
    print(f'Created new policy copy {new_pol_id}')

    def _set(worker):
        worker.set_policy_mapping_fn(training_policy_mapping_fn)
        training_population = ray.get(pop_manager.get_attr.remote('training_population'))
        worker.set_is_policy_to_train(list(training_population))

    algorithm.workers.foreach_worker(_set)


def list_current_policies(algorithm: Algorithm) -> List[str]:
    return list(sorted(algorithm.workers.local_worker().policy_map.keys()))


def has_at_least_one_policy_per_player(policy_ids: List[str], num_players: int):
    """
    Checks if there is at least one policy per player.
    @param policy_ids:
    @param num_players:
    @return:
    """
    return all([any([get_player_idx(p) == i for p in policy_ids]) for i in range(num_players)])


def calculate_metrics(pop_manager: ActorHandle):
    """
    Calculates metrics to log to wandb.
    @param pop_manager:
    @return:
    """
    curriculum_iteration = ray.get(pop_manager.get_attr.remote('curriculum_iteration'))
    training_population = ray.get(pop_manager.get_attr.remote('training_population'))
    paused_training_population = ray.get(pop_manager.get_attr.remote('paused_training_population'))
    stopped_training_population = ray.get(pop_manager.get_attr.remote('stopped_training_population'))
    frozen_population = ray.get(pop_manager.get_attr.remote('frozen_population'))
    play_stats = ray.get(pop_manager.get_attr.remote('play_stats'))
    curriculum_iteration_games_played = ray.get(pop_manager.get_attr.remote('curriculum_games_played'))
    curriculum_rewards = ray.get(pop_manager.get_attr.remote('curriculum_rewards'))
    curriculum_initial_rewards = ray.get(pop_manager.get_attr.remote('curriculum_initial_reward_estimates'))
    recent_reward_window = ray.get(pop_manager.get_attr.remote('recent_reward_window'))

    # Get target rewards for each policy
    target_rewards = {}
    for policy_id in training_population:
        initial_reward, current_reward, required_reward = ray.get(pop_manager.get_reward_targets.remote(policy_id))
        target_rewards[f'{policy_id}_target_rewards'] = {'initial_reward': initial_reward,
                                                         'current_reward': current_reward,
                                                         'required_reward': required_reward}

    def avg_reward(rewards: List[float]):
        return np.average(rewards[-recent_reward_window:])

    avg_reward_per_opponent = {f'{policy_id}_reward': {opp_id: avg_reward(play_stats[policy_id][opp_id])
                                                       for opp_id in play_stats[policy_id]}
                               for policy_id in play_stats}
    games_per_opponent = {f'{policy_id}_games_played': {opp_id: len(play_stats[policy_id][opp_id])
                                                        for opp_id in play_stats[policy_id]}
                          for policy_id in play_stats}

    curriculum_avg_rewards = {policy_id: avg_reward(curriculum_rewards[curriculum_iteration][policy_id])
                              for policy_id in curriculum_rewards[curriculum_iteration]}

    games_played = defaultdict(int)
    for c in curriculum_iteration_games_played:
        for p in curriculum_iteration_games_played[c]:
            games_played[p] += curriculum_iteration_games_played[c][p]

    metrics = {'curriculum_iteration': curriculum_iteration,
               'training_population_size': len(training_population),
               'paused_training_population_size': len(paused_training_population),
               'stopped_training_population_size': len(stopped_training_population),
               'frozen_population_size': len(frozen_population),
               'games_played': games_played,
               'curriculum_avg_rewards': curriculum_avg_rewards,
               'curriculum_initial_reward_estimates': curriculum_initial_rewards[curriculum_iteration],
               } | avg_reward_per_opponent | games_per_opponent | target_rewards

    return metrics
