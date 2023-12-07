from collections import defaultdict
from copy import copy
from typing import Dict, Any, List
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
        # self.games_played = {p: 0 for p in self.training_population}
        self.training_episodes = {}  # dict of {episode_id: {agent_id: policy_id}}
        self.curriculum_iteration = 1  # which iteration of frozen policies we are on
        # Tracks the rewards obtained per curriculum iteration
        self.curriculum_rewards = {self.curriculum_iteration: {p: [] for p in self.training_population}}
        # the initial rewards a policy has at the start of a curriculum iteration
        self.curriculum_initial_rewards = {self.curriculum_iteration: {}}
        # the final training step for the curriculum iteration
        self.curriculum_ends = {}
        # number of games each policy played per curriculum iteration
        self.curriculum_games_played = {self.curriculum_iteration: {p: 0 for p in self.training_population}}

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

    def get_training_matchup(self, episode_id: str, agent_id: str):
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

    def sample_opponent_for(self, main_policy_id: str, player_idx: int) -> str:
        """
        Samples an opponent for a given the main policy and desired player_idx.
        @param main_policy_id
        @param player_idx:
        @return: policy ID of the opponent
        """
        # TODO consider multiple implementation variants for sampling opponents
        policies_matching_role = [p for p in self.frozen_population if get_player_idx(p) == player_idx]

        policy_weights = []
        for p in policies_matching_role:
            recent_rewards = self.play_stats[main_policy_id][p][-self.recent_reward_window:]

            # If there exists a policy that not enough games have been played against, pick it
            if len(recent_rewards) < self.min_games_against_new_opponent:
                return p
            # Otherwise let the weight be the average of the recent rewards.
            avg_reward = np.average(recent_rewards)
            policy_weights.append(avg_reward)

        # Derive probabilities from the average rewards via normalisation and softmax
        probs = softmax(-zscore(policy_weights))
        # In some cases (e.g when we have one opponent or all policies have the same reward), the z-score will be NaN.
        # This is not invalid, but simply a consequence of statistics (e.g not enough samples or 0 STD dev).
        # In such cases we sample uniformly instead.
        if any(np.isnan(p) for p in probs):
            return np.random.choice(policies_matching_role)
        return np.random.choice(policies_matching_role, p=probs)

    def add_new_frozen_policy(self, policy_id: str):
        """
        Adds a new frozen policy to the frozen opponent population.
        @param policy_id:
        @return:
        """
        self.frozen_population.add(policy_id)
        for p in self.training_population:
            self.play_stats[p][policy_id] = []

    def apply_episode_result(self, episode_id: str, result: dict):
        """
        Updates play metrics based on the result of an episode.
        @param episode_id:
        @param result:
        @return:
        """
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

        self.curriculum_games_played[self.curriculum_iteration][main_policy] += 1
        self.curriculum_rewards[self.curriculum_iteration][main_policy].append(main_policy_reward)
        for o in other_policies:
            self.play_stats[main_policy][o].append(main_policy_reward)

    def check_update_training_population(self):
        """
        Checks if policies in the training population have improved sufficiently to pause their training.
        @return:
        """

        curriculum_rewards = self.filter_by_in_training(self.curriculum_rewards[self.curriculum_iteration])
        curriculum_initial_rewards = self.filter_by_in_training(
            self.curriculum_initial_rewards[self.curriculum_iteration])

        # Estimate the initial reward of a curriculum iteration
        for policy_id, rewards in curriculum_rewards.items():
            if policy_id not in curriculum_initial_rewards and len(rewards) >= self.recent_reward_window:
                avg_reward = np.average(rewards[-self.recent_reward_window:])
                curriculum_initial_rewards[policy_id] = avg_reward
                self.curriculum_initial_rewards[self.curriculum_iteration][policy_id] = avg_reward

        # If enough rewards have been sampled to estimate the initial reward for all training agents,
        # Then we check if any of the agents have improved sufficiently to pause their training.
        if len(curriculum_initial_rewards) == len(self.training_population):
            for policy_id, initial_reward in curriculum_initial_rewards.items():

                # We offset by the minimum reward to ensure our rewards are always non-negative.
                min_reward = min(curriculum_rewards[policy_id])
                reward_offset = min_reward * -1 if min_reward < 0 else 0
                # We compare the initial reward and requirement improvement to the average reward over the last episodes.
                required_reward = (initial_reward + reward_offset) * self.policy_improvement_factor
                current_reward = np.average(curriculum_rewards[policy_id][-self.recent_reward_window:]) + reward_offset
                # If the reward has improved enough, we pause the training for that policy.
                if current_reward > required_reward:
                    self.training_population.remove(policy_id)
                    self.paused_training_population.add(policy_id)

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
        self.curriculum_initial_rewards[self.curriculum_iteration] = {}

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

        metrics = calculate_metrics(self.pop_manager)

        # Add custom metrics
        for m in metrics:
            result[m] = metrics[m]

    def on_episode_end(self, **kwargs, ) -> None:
        episode: Episode = kwargs['episode']
        self.pop_manager.apply_episode_result.remote(episode.episode_id, episode.agent_rewards)


def training_policy_mapping_fn(agent_id: str, episode: Episode, **kwargs):
    pop_manager = ray.get_actor('population_manager')
    policy_id = ray.get(pop_manager.get_training_matchup.remote(episode.episode_id, agent_id))
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
        policy_mapping_fn=training_policy_mapping_fn,
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


def list_current_policies(algorithm: Algorithm):
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
    curriculum_initial_rewards = ray.get(pop_manager.get_attr.remote('curriculum_initial_rewards'))
    recent_reward_window = ray.get(pop_manager.get_attr.remote('recent_reward_window'))

    def avg_reward(rewards: List[float]):
        return np.average(rewards[-recent_reward_window:])

    avg_reward_per_opponent = {f'{policy_id}_reward': {opp_id: avg_reward(play_stats[policy_id][opp_id])
                                                       for opp_id in play_stats[policy_id]}
                               for policy_id in play_stats}
    games_per_opponent = {f'{policy_id}_games_played': {opp_id: len(play_stats[policy_id][opp_id])
                                                        for opp_id in play_stats[policy_id]}
                          for policy_id in play_stats}

    # TODO join all rewards together before averaging for smoother charts
    curriculum_avg_rewards = {policy_id: avg_reward(curriculum_rewards[curriculum_iteration][policy_id])
                              for policy_id in curriculum_rewards[curriculum_iteration]}

    games_played = defaultdict(int)
    for c in curriculum_iteration_games_played:
        for p in curriculum_iteration_games_played[c]:
            games_played[p] += curriculum_iteration_games_played[c][p]

    # TODO filter out irrelevant opponents
    metrics = {'curriculum_iteration': curriculum_iteration,
               'training_population_size': len(training_population),
               'paused_training_population_size': len(paused_training_population),
               'stopped_training_population_size': len(stopped_training_population),
               'frozen_population_size': len(frozen_population),
               'games_played': games_played,
               'curriculum_avg_rewards': curriculum_avg_rewards,
               'curriculum_initial_rewards': curriculum_initial_rewards[curriculum_iteration],
               } | avg_reward_per_opponent | games_per_opponent

    return metrics
