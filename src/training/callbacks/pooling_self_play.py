from typing import Dict, Any, List, Callable, Union, Optional
from ray.actor import ActorHandle
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.worker_set import WorkerSet
import ray
import numpy as np
from ray.rllib.examples.policy.random_policy import RandomPolicy
from scipy.special import softmax
from scipy.stats import zscore

from src.utils.helpers import get_trailing_number, flatten_list


@ray.remote
class PolicyPoolManager:
    def __init__(self, config: Dict[str, Any], run_config: Dict[str, Any]):
        self.main_policy_ids = list(config['multiagent']['policies'].keys())

        assert len(self.main_policy_ids), 'This training method has been designed for exactly two main policies. ' \
                                          f'Received: {self.main_policy_ids}'
        # Controls how often to sample from the pool of already mastered opponents during training.
        self.mastered_pool_max_prob = run_config['pooled_self_play']['mastered_pool_max_prob']
        self.reward_distance = run_config['pooled_self_play']['reward_distance']
        self.mastered_pool_sampler = self.get_pool_sampler(run_config['pooled_self_play']['mastered_pool_sampling'])
        self.competing_pool_sampler = self.get_pool_sampler(run_config['pooled_self_play']['competing_pool_sampling'])
        self.initial_opponents = run_config['pooled_self_play']['init_opponents']
        self.init_with_shared_weights = run_config['pooled_self_play']['init_with_shared_weights']

        # How many times a main policy needs to win against a competing policy to move the competing policy
        # to the mastered pool
        self.consecutive_training_wins_needed = run_config['pooled_self_play']['consecutive_training_wins_needed']
        # How many times a main policy needs to win against a mastered policy during evaluation for the
        # mastered policy to stay in the mastered pool.
        self.consecutive_eval_wins_needed = run_config['pooled_self_play']['consecutive_eval_wins_needed']
        self.episode_results = {}  # dict of {episode_id : reward dict}
        self.longterm_episode_results = {}  # dict of {episode_id : reward dict}, kept mostly for debugging
        # dict of {policy_id : {opponent_id : consecutive wins}}
        self.policy_consecutive_wins = {p: {} for p in self.main_policy_ids}
        # dict of {policy_id : {opponent_id : (mean reward, num_games)}}
        self.train_policy_mean_rewards = {p: {} for p in self.main_policy_ids}
        self.eval_policy_mean_rewards = {p: {} for p in self.main_policy_ids}

        # dict of {policy_id : {opponent_id : last reward}}
        # Minimum reward against an opponent in most recent batch of episodes.
        self.train_policy_min_rewards = {p: {} for p in self.main_policy_ids}
        self.eval_policy_min_rewards = {p: {} for p in self.main_policy_ids}

        # dict of {policy_id : {opponent_id : last reward}}
        # Amount of games played against each policy.
        self.train_episodes_played = {p: {} for p in self.main_policy_ids}
        self.eval_episodes_played = {p: {} for p in self.main_policy_ids}

        self.curriculum_iteration = 0
        # dict of {policy_id : set(frozen_opponent_policy_ids)}
        self.mastered_pool = {p: set() for p in self.main_policy_ids}
        # dict of {policy_id : set(frozen_opponent_policy_ids)}
        self.competing_pool = {p: set() for p in self.main_policy_ids}
        self.training_episodes = {}  # dict of {episode_id: (main_policy_id, opponent_policy_id)}

        self.evaluation_pairings = []  # List of [(main_policy_id, opponent_policy_id)]
        self.evaluation_episodes = {}  # dict of {episode_id: (main_policy_id, opponent_policy_id)}
        # For some reason Rllib oversamples episodes during evaluation so we need to keep track of "dud" episodes
        # that we don't want to measure.
        self.dud_evaluation_episodes = set()  # set of str ids.

    def get_pool_sampler(self, pool_sampling: str):
        if pool_sampling == 'uniform':
            return self.sample_uniformly
        elif pool_sampling == 'inverse last minimum train reward':
            return self.sample_inverse_to_recent_training_min_reward
        else:
            raise Exception(f'Unknown pool sampling config: {pool_sampling}')

    def sample_uniformly(self, main_policy_id: str, pool: set) -> str:
        return np.random.choice(list(pool))

    def sample_inverse_to_recent_training_min_reward(self, main_policy_id: str, pool: set) -> str:
        # If the policy hasn't played against each opponent at least once, we choose an unplayed opponent.
        unplayed_opponents = [o for o in pool if o not in self.train_policy_min_rewards[main_policy_id]]

        if len(unplayed_opponents) > 0:
            return unplayed_opponents[0]

        rewards = [self.train_policy_min_rewards[main_policy_id][opponent_id] for opponent_id in pool]
        policy_ids = list(pool)
        probs = softmax(-zscore(rewards))  # Perform some level of normalisation, z-score and then flip sign.

        # In some cases (e.g when we have one opponent or all policies have the same reward), the z-score will be NaN.
        # This is not invalid, but simply a consequence of statistics (e.g not enough samples or 0 STD dev).
        # In such cases we sample uniformly instead.
        if any([np.isnan(p) for p in probs]):
            return self.sample_uniformly(main_policy_id, pool)
        return np.random.choice(policy_ids, p=probs)

    def get_attr(self, param_name):
        return getattr(self, param_name)

    def increment_curriculum_iteration(self):
        self.curriculum_iteration += 1

    def update_episode_results(self, episode_id: str, episode_rewards: dict):
        if episode_id in self.dud_evaluation_episodes:
            self.dud_evaluation_episodes.remove(episode_id)
        else:
            self.episode_results[episode_id] = episode_rewards
            self.longterm_episode_results[episode_id] = episode_rewards

    def reset_episode_results(self):
        self.episode_results = {}

    def reset_consecutive_wins(self):
        self.policy_consecutive_wins = {p: {} for p in self.main_policy_ids}

    def update_consecutive_wins(self, new_policy_consecutive_wins: dict):
        self.policy_consecutive_wins = new_policy_consecutive_wins

    def update_stats(self, policy_min_rewards: dict, policy_mean_rewards: dict, episodes_played: dict, eval_mode: True):
        if eval_mode:
            self.eval_policy_mean_rewards = policy_mean_rewards
            self.eval_policy_min_rewards = policy_min_rewards
            self.eval_episodes_played = episodes_played
        else:
            self.train_policy_mean_rewards = policy_mean_rewards
            self.train_policy_min_rewards = policy_min_rewards
            self.train_episodes_played = episodes_played

    def move_opponent_to_mastered_pool(self, main_policy_id: str, opponent_id: str):
        self.competing_pool[main_policy_id].discard(opponent_id)
        self.mastered_pool[main_policy_id].add(opponent_id)

    def move_opponent_to_competing_pool(self, main_policy_id: str, opponent_id: str):
        self.mastered_pool[main_policy_id].discard(opponent_id)
        self.competing_pool[main_policy_id].add(opponent_id)

    # Add a frozen version of a main policy to all other agents' competing pools.
    def add_new_opponent_to_competing(self, main_policy_id: str, opponent_id: str):
        for p in self.competing_pool.keys():
            if p != main_policy_id:
                self.competing_pool[p].add(opponent_id)

    def _choose_training_opponent(self, policy_id: str) -> str:
        agent_m_pool = self.mastered_pool[policy_id]
        agent_c_pool = self.competing_pool[policy_id]

        total_pool_size = len(agent_m_pool) + len(agent_c_pool)

        # Set the probability for a worker to choose an opponent from each pool.
        # The probability of selecting from the pool of already mastered agents is upper-bounded
        # so the agent doesn't spend too much time playing opponents it can already beat.
        m_pool_prob = min(len(agent_m_pool) / total_pool_size, self.mastered_pool_max_prob)
        c_pool_prob = 1 - m_pool_prob

        pool_choice = np.random.choice(a=['competing', 'mastered'], p=[c_pool_prob, m_pool_prob])

        if pool_choice == 'competing':
            opponent_id = self.competing_pool_sampler(policy_id, agent_c_pool)
        elif pool_choice == 'mastered':
            opponent_id = self.mastered_pool_sampler(policy_id, agent_m_pool)
        else:
            raise Exception(f'Unhandled pool choice {pool_choice}')

        return opponent_id

    def get_training_matchup(self, episode_id: str, agent_id: str):
        if episode_id not in self.training_episodes:
            self.training_episodes[episode_id] = {}

            # Filter main policies where there are no more opponents to compete with.
            # Usually this means we're waiting on some other main policy to finish learning.
            training_main_policies = [m for m in self.main_policy_ids if len(self.competing_pool[m]) > 0]

            # Choose one main policy
            chosen_policy_id = np.random.choice(training_main_policies)
            chosen_agent_id = f'player_{get_trailing_number(chosen_policy_id)}'
            self.training_episodes[episode_id][chosen_agent_id] = chosen_policy_id

            # For the opponent, choose a frozen policy
            base_opponent_policy_id = [m for m in self.main_policy_ids if m != chosen_policy_id][0]
            opponent_agent_id = f'player_{get_trailing_number(base_opponent_policy_id)}'
            opponent_policy_id = self._choose_training_opponent(policy_id=chosen_policy_id)
            self.training_episodes[episode_id][opponent_agent_id] = opponent_policy_id

        return self.training_episodes[episode_id][agent_id]

    def generate_evaluation_matchups(self) -> int:
        # Note: we shouldn't need to merge these pools as all opponents should be in the mastered pool prior
        # to evaluation anyway. Doing it is as a safety measure.
        merged_pools = {m: self.mastered_pool[m].union(self.competing_pool[m]) for m in self.main_policy_ids}
        # Generates tuples of every matchup we want to evaluate.
        self.evaluation_pairings = flatten_list([[[m, opp]] * self.consecutive_eval_wins_needed
                                                 for m in self.main_policy_ids
                                                 for opp in merged_pools[m]])
        return len(self.evaluation_pairings)

    def get_evaluation_matchup(self, episode_id: str):
        if episode_id not in self.evaluation_episodes:
            # If Rllib oversamples, yield main ids and mark the episode as a dud.
            if len(self.evaluation_pairings) == 0:
                self.evaluation_episodes[episode_id] = self.main_policy_ids
                self.dud_evaluation_episodes.add(episode_id)
            else:
                self.evaluation_episodes[episode_id] = self.evaluation_pairings.pop(-1)
        return self.evaluation_episodes[episode_id]

    def reset_evaluation_states(self):
        self.evaluation_pairings = []
        self.evaluation_episodes = {}


class PoolingSelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.policy_manager = ray.get_actor('policy_manager')

    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        print('Initialising Pooling Self-play...')
        main_policy_ids = ray.get(self.policy_manager.get_attr.remote('main_policy_ids'))
        initial_opponents = ray.get(self.policy_manager.get_attr.remote('initial_opponents'))
        init_with_shared_weights = ray.get(self.policy_manager.get_attr.remote('init_with_shared_weights'))

        # Initialise against random policies
        if initial_opponents == 'random policy':
            temp_env = algorithm.env_creator(algorithm.config['env_config'])
            for id in main_policy_ids:
                new_pol_id = f"{id}_random"
                self.policy_manager.add_new_opponent_to_competing.remote(id, new_pol_id)
                algorithm.add_policy(policy_id=new_pol_id,
                                     policy_cls=RandomPolicy,
                                     observation_space=temp_env.observation_space[f'player_0'],
                                     action_space=temp_env.action_space[f'player_0'])

                def _set(worker):
                    worker.set_policy_mapping_fn(training_policy_mapping_fn)

                algorithm.workers.foreach_worker(_set)
        # Initialise against frozen opponent policies
        elif initial_opponents == 'frozen main policy':
            for id in main_policy_ids:
                clone_policy_and_add_to_competing(self.policy_manager, algorithm, id)
        else:
            raise Exception(f'Unknown training opponent initialisation {initial_opponents}')

        # Choose whether to initialise all policies to the same weights
        if init_with_shared_weights:
            first_policy_weights = algorithm.get_weights()[main_policy_ids[0]]
            weights_dict = {m_id: first_policy_weights for m_id in main_policy_ids}
            algorithm.set_weights(weights_dict)
            algorithm.workers.sync_weights(policies=main_policy_ids)

    def on_train_result(self, *, result: dict, algorithm: Algorithm, trainer=None, **kwargs):

        # After training update win_rates and pools.
        consecutive_training_wins_needed = ray.get(
            self.policy_manager.get_attr.remote('consecutive_training_wins_needed'))
        self._update_win_stats_and_pools(consecutive_training_wins_needed, eval_mode=False)
        competing_pool = ray.get(self.policy_manager.get_attr.remote('competing_pool'))

        # We can temporarily hack pool states, so we can debug evaluation
        # for m_id in competing_pool:
        #     for o_id in competing_pool[m_id]:
        #         self.policy_manager.move_opponent_to_mastered_pool.remote(m_id, o_id)
        # competing_pool = ray.get(self.policy_manager.get_attr.remote('competing_pool'))

        # If there are no more competing policies, run an evaluation
        if all([len(competing_pool[a]) == 0 for a in competing_pool]):
            print('Running evaluation...')
            consecutive_eval_wins_needed = ray.get(self.policy_manager.get_attr.remote('consecutive_eval_wins_needed'))
            self._evaluate_pools(algorithm, algorithm.evaluation_workers)
            self._update_win_stats_and_pools(consecutive_eval_wins_needed, eval_mode=True)

            # Get updated pool
            competing_pool = ray.get(self.policy_manager.get_attr.remote('competing_pool'))
            # If there are still no more competing agents, we can freeze the current policies and continue
            if all([len(competing_pool[a]) == 0 for a in competing_pool]):
                print('Passed evaluation! Creating new agents...')
                main_policy_ids = ray.get(self.policy_manager.get_attr.remote('main_policy_ids'))
                self.policy_manager.increment_curriculum_iteration.remote()
                for policy_id in main_policy_ids:
                    clone_policy_and_add_to_competing(self.policy_manager, algorithm, policy_id)

        ray.get(self.policy_manager.reset_evaluation_states.remote())
        metrics = calculate_metrics(self.policy_manager)

        # Add custom metrics
        for m in metrics:
            result[m] = metrics[m]

    def _update_win_stats_and_pools(self, win_rates_needed: int, eval_mode: bool):
        episode_results = ray.get(self.policy_manager.get_attr.remote('episode_results'))
        main_policy_ids = ray.get(self.policy_manager.get_attr.remote('main_policy_ids'))
        reward_distance = ray.get(self.policy_manager.get_attr.remote('reward_distance'))
        consecutive_wins = ray.get(self.policy_manager.get_attr.remote('policy_consecutive_wins'))
        if eval_mode:
            policy_mean_rewards = ray.get(self.policy_manager.get_attr.remote('eval_policy_mean_rewards'))
            policy_min_rewards = ray.get(self.policy_manager.get_attr.remote('eval_policy_min_rewards'))
            episodes_played = ray.get(self.policy_manager.get_attr.remote('eval_episodes_played'))
        else:
            policy_mean_rewards = ray.get(self.policy_manager.get_attr.remote('train_policy_mean_rewards'))
            policy_min_rewards = ray.get(self.policy_manager.get_attr.remote('train_policy_min_rewards'))
            episodes_played = ray.get(self.policy_manager.get_attr.remote('train_episodes_played'))

        successful_opponents = {p: set() for p in main_policy_ids}
        min_rewards = {p: {} for p in main_policy_ids}

        # Iterate over episode results and determine winners
        for rewards in episode_results.values():
            policy_rewards = {p_id: rewards[(_, p_id)] for _, p_id in rewards}
            policy_ids = list(policy_rewards.keys())
            main_policy = [i for i in policy_ids if i in main_policy_ids][0]
            opponent_policy = [i for i in policy_ids if i not in main_policy_ids][0]

            # if the opponent and main policies have never played before:
            if opponent_policy not in consecutive_wins[main_policy]:
                consecutive_wins[main_policy][opponent_policy] = 0
            if opponent_policy not in policy_mean_rewards[main_policy]:
                policy_mean_rewards[main_policy][opponent_policy] = (0, 0)
                policy_min_rewards[main_policy][opponent_policy] = 0

            # Main policy won by a sufficient margin
            if policy_rewards[main_policy] > policy_rewards[opponent_policy] + reward_distance:
                consecutive_wins[main_policy][opponent_policy] += 1
            # Main policy did not win
            else:
                consecutive_wins[main_policy][opponent_policy] = 0  # reset win streak
                successful_opponents[main_policy].add(opponent_policy)  # add opponent to winners

            # Calculate mean
            prev_mean, num_games = policy_mean_rewards[main_policy][opponent_policy]
            policy_mean_rewards[main_policy][opponent_policy] = (
                prev_mean + 1 / (num_games + 1) * (policy_rewards[main_policy] - prev_mean), num_games + 1)

            # Calculate min
            if opponent_policy not in min_rewards[main_policy]:
                min_rewards[main_policy][opponent_policy] = policy_rewards[main_policy]
            else:
                min_rewards[main_policy][opponent_policy] = min(min_rewards[main_policy][opponent_policy],
                                                                policy_rewards[main_policy])
            # Update policy play counts
            if opponent_policy not in episodes_played[main_policy]:
                episodes_played[main_policy][opponent_policy] = 0

            episodes_played[main_policy][opponent_policy] += 1

        # Update min rewards
        for m in main_policy_ids:
            for o in min_rewards[m]:
                policy_min_rewards[m][o] = min_rewards[m][o]

        # Review policy performances and shift around pool if needed
        for m in main_policy_ids:
            for o in consecutive_wins[m]:
                # Move any successfully mastered opponent policies
                if consecutive_wins[m][o] >= win_rates_needed:
                    ray.get(self.policy_manager.move_opponent_to_mastered_pool.remote(m, o))
                    successful_opponents[m].discard(o)
            # Move any successful opponents back to the competing pool.
            for o in successful_opponents[m]:
                ray.get(self.policy_manager.move_opponent_to_competing_pool.remote(m, o))

        # Update consecutive wins
        ray.get(self.policy_manager.update_consecutive_wins.remote(consecutive_wins))
        ray.get(self.policy_manager.update_stats.remote(policy_min_rewards,
                                                        policy_mean_rewards,
                                                        episodes_played,
                                                        eval_mode))
        ray.get(self.policy_manager.reset_episode_results.remote())

    # TODO observation filter will likely not be called on direct evaluation of policies. May need to implement this as an Observation wrapper
    # Yields main custom evaluation function
    def _evaluate_pools(self, algo: Algorithm, eval_workers: WorkerSet):

        # Update eval workers with most recent training state.
        eval_workers.sync_weights(from_worker=algo.workers.local_worker())

        # Generate eval games for workers to use
        eval_games_count = ray.get(self.policy_manager.generate_evaluation_matchups.remote())
        eval_workers_count = len(eval_workers.remote_workers()) + (1 if eval_workers.local_worker() else 0)

        # Set policy mapping function using evaluation for workers.
        update_func = lambda w: w.set_policy_mapping_fn(get_eval_opponent)
        for idx in range(eval_workers_count):
            apply_to_worker_i(eval_workers, update_func, idx)

        while eval_games_count > 0:
            # If there still are enough games to play, use all workers.
            if eval_workers_count >= eval_workers_count:
                eval_workers.foreach_worker(func=lambda w: w.sample())
                eval_games_count -= eval_workers_count
            # Use subset of workers to play rest.
            else:
                for i in range(eval_games_count):
                    apply_to_worker_i(eval_workers, lambda w: w.sample(), i)
                    eval_workers_count -= 1

    def on_episode_end(self, **kwargs):
        episode: Episode = kwargs['episode']
        rewards = dict(episode.agent_rewards)
        self.policy_manager.update_episode_results.remote(episode.episode_id, rewards)


def training_policy_mapping_fn(a_id: str, episode: Episode, worker, **kwargs):
    policy_manager = ray.get_actor('policy_manager')
    policy_id = ray.get(policy_manager.get_training_matchup.remote(episode.episode_id, a_id))
    return policy_id


# Freezes a current policy and adds it to the other agent's competing pool.
def clone_policy_and_add_to_competing(policy_manager: ActorHandle, algorithm: Algorithm, main_policy_id: str):
    curriculum_iteration = ray.get(policy_manager.get_attr.remote('curriculum_iteration'))
    new_pol_id = f"{main_policy_id}_v{curriculum_iteration}"
    policy_manager.add_new_opponent_to_competing.remote(main_policy_id, new_pol_id)

    temp_env = algorithm.env_creator(algorithm.config['env_config'])
    main_state = algorithm.get_policy(main_policy_id).get_state()
    algorithm.add_policy(
        policy_id=new_pol_id,
        policy_cls=type(algorithm.get_policy(main_policy_id)),
        policy_state=main_state,
        policy_mapping_fn=training_policy_mapping_fn,
        # TODO make these IDs non-hard-coded
        observation_space=temp_env.observation_space[f'player_0'],
        action_space=temp_env.action_space[f'player_0'],
    )
    # We need to sync the just copied local weights to all the
    # remote workers as well.
    algorithm.workers.sync_weights(policies=[new_pol_id])
    print(f'Created new policy copy {new_pol_id}')

    def _set(worker):
        worker.set_policy_mapping_fn(training_policy_mapping_fn)

    algorithm.workers.foreach_worker(_set)


# Returns a policy mapping function to match up exclusively to a particular opponent
def get_eval_opponent(a_id: str, episode: Episode, worker, **kwargs):
    policy_manager = ray.get_actor('policy_manager')
    matchup_policies = ray.get(policy_manager.get_evaluation_matchup.remote(episode.episode_id))

    agent_policy = f'policy{get_trailing_number(a_id)}'

    # If the agent matches to a main policy, play as that
    if agent_policy in matchup_policies:
        return agent_policy

    # If the agent matches to an opponent policy, play as that
    matching_opps = [opp for opp in matchup_policies if agent_policy in opp]
    if len(matching_opps) == 1:
        return matching_opps[0]

    raise Exception(f'Could not match agent {a_id} to policies {matchup_policies}')


# Applies one function per worker from a list.
def apply_to_workers(workers: WorkerSet, funcs: List[Callable[[RolloutWorker], Any]]):
    local_result = []
    offset = 0
    # Local worker: Index=0.
    if workers.local_worker() is not None:
        local_result = [funcs[0](workers.local_worker())]
        offset = 1
    # Remote workers: Index > 0.
    remote_results = ray.get(
        [w.apply.remote(funcs[i + offset]) for i, w in enumerate(workers.remote_workers())]
    )
    return local_result + remote_results


def apply_to_worker_i(workers: WorkerSet, func: Callable[[RolloutWorker], Any], index: int):
    # Local worker: Index=0.
    if workers.local_worker() is not None and index == 0:
        return func(workers.local_worker())

    # Remote workers: Index >= 0.
    remote_workers = list(workers.remote_workers())
    remote_result = ray.get(remote_workers[index].apply.remote(func))
    return remote_result


def calculate_metrics(policy_manager: ActorHandle):
    curriculum_iteration = ray.get(policy_manager.get_attr.remote('curriculum_iteration'))
    policy_consecutive_wins = ray.get(policy_manager.get_attr.remote('policy_consecutive_wins'))
    competing_pool = ray.get(policy_manager.get_attr.remote('competing_pool'))
    mastered_pool = ray.get(policy_manager.get_attr.remote('mastered_pool'))
    competing_pool_sizes = [len(competing_pool[m]) for m in competing_pool]
    mastered_pool_sizes = [len(mastered_pool[m]) for m in mastered_pool]

    def get_mean_rewards(pmr: dict):
        mean_rewards = {}
        for m in pmr:
            mean_rewards[m] = {opp: pmr[m][opp][0] for opp in pmr[m]}
        return mean_rewards

    eval_pmr = ray.get(policy_manager.get_attr.remote('eval_policy_mean_rewards'))
    eval_policy_mean_rewards = get_mean_rewards(eval_pmr)
    eval_policy_min_rewards = ray.get(policy_manager.get_attr.remote('eval_policy_min_rewards'))
    train_pmr = ray.get(policy_manager.get_attr.remote('train_policy_mean_rewards'))
    train_policy_mean_rewards = get_mean_rewards(train_pmr)
    train_policy_min_rewards = ray.get(policy_manager.get_attr.remote('train_policy_min_rewards'))
    train_episodes_played = ray.get(policy_manager.get_attr.remote('train_episodes_played'))
    eval_episodes_played = ray.get(policy_manager.get_attr.remote('eval_episodes_played'))

    return {'curriculum_iteration': curriculum_iteration,
            'policy_consecutive_wins': policy_consecutive_wins,
            'competing_pool_size': np.sum(competing_pool_sizes),
            'mastered_pool_size': np.sum(mastered_pool_sizes),
            'eval_policy_min_rewards': eval_policy_min_rewards,
            'eval_policy_mean_rewards': eval_policy_mean_rewards,
            'eval_episodes_played': eval_episodes_played,
            'train_policy_min_rewards': train_policy_min_rewards,
            'train_policy_mean_rewards': train_policy_mean_rewards,
            'train_episodes_played': train_episodes_played}
