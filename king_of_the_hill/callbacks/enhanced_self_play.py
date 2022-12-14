import ray
# from ray.rllib.agents import Trainer
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation import Episode
from king_of_the_hill.custom_metrics import MetricsCallbacks
from collections import OrderedDict
import numpy as np
import scipy as sp


# TODO count game stats (number of games played, number of games won, draw and lost)
@ray.remote
class PolicyScorer:
    def __init__(self, main_policy_id: str):
        self.main_policy = main_policy_id
        self.policy_metrics = OrderedDict()  # score, games played, won, draw and lost
        self.learning_rate = 0.05

    def add_policy(self, policy_id: str):
        self.policy_metrics[policy_id] = {'score': 0, 'games': 0, 'won': 0, 'draw': 0, 'lost': 0}

    # Dict of {(player_id, policy_id):reward}
    def update_metrics(self, rewards: dict):
        main_policy_rewards = [rewards[t] for t in rewards if t[1] == self.main_policy]
        assert len(
            main_policy_rewards) > 0, f"Expected to use the current policy '{self.main_policy}', but got {rewards.keys()}"
        main_reward = max(main_policy_rewards)

        for (_, policy_id), reward in rewards.items():
            if policy_id == self.main_policy:
                continue
            self.policy_metrics[policy_id]['games'] += 1

            if reward == main_reward:
                self.policy_metrics[policy_id]['draw'] += 1
            elif reward > main_reward:
                self.policy_metrics[policy_id]['won'] += 1
            else:
                self.policy_metrics[policy_id]['lost'] += 1
                past_policy_count = len(self.policy_metrics)
                probs = self.get_policy_probabilities()
                self.policy_metrics[policy_id]['score'] -= self.learning_rate / (past_policy_count * probs[policy_id])

    def get_policy_probabilities(self):
        if len(self.policy_metrics) == 0:
            return {}

        policy_scores = self.get_policy_scores()
        probs = sp.special.softmax(list(policy_scores.values()))
        probs_dict = OrderedDict()
        for name, p in zip(policy_scores, probs):
            probs_dict[name] = p
        return probs_dict

    def get_policy_metrics(self) -> dict:
        return self.policy_metrics

    def get_policy_scores(self):
        policy_scores = OrderedDict()
        for policy_id in self.policy_metrics:
            policy_scores[policy_id] = self.policy_metrics[policy_id]['score']

        return policy_scores


# TODO would this still work on resume?
# TODO will older policies need to be cached?
# TODO what about recolouring observation?
# This is an implementation of this paper's self-play: https://arxiv.org/pdf/1912.06680.pdf  -  page 59
class EnhancedSelfPlayCallback(MetricsCallbacks):
    def __init__(self):
        super().__init__()
        self.main_policy = 'av'
        self.policy_scorer = ray.get_actor('policy_scorer')

        self.save_policy_every = 10  # after how many training iterations to save a policy
        self.print_policies_every = 10
        self.match_past_opponent = 0.2  # probability of matching a past opponent
        self.episode_policies = {}  # maps episode ID to dict of player id and policy used

    # Initialise variables here because the RLLib trainer does not allow inputting arguments into the constructor.
    def _extra_init(self):
        pass

    def _generate_policy_mapping_fn(self):
        policy_probs = ray.get(self.policy_scorer.get_policy_probabilities.remote())
        policy_ids = list(policy_probs.keys())

        def select_policies(episode: Episode) -> dict:
            policy_mapping = {}
            player_ids = episode.worker.env.get_agent_ids()

            # Assign each player a policy
            for player in player_ids:
                # If we have no past policies to train against yet
                if len(policy_ids) == 0:
                    policy_mapping[player] = self.main_policy
                # Select either present or past policies with probability
                else:
                    choice = np.random.choice(a=[1, 2], p=[1 - self.match_past_opponent, self.match_past_opponent])
                    # Play against current agent
                    if choice == 1:
                        policy_mapping[player] = self.main_policy
                    # Play against past agent
                    elif choice == 2:
                        probs = list(policy_probs.values())
                        policy_mapping[player] = np.random.choice(a=policy_ids, p=probs)
                    else:
                        raise Exception(f'Unhandled case {choice}')

            # Ensure that at least one player is mapped to the trainable policy.
            mapped_main_policy = self.main_policy in list(policy_mapping.values())
            if not mapped_main_policy:
                player = np.random.choice(list(player_ids))
                policy_mapping[player] = self.main_policy

            return policy_mapping

        def policy_fn_mapper_wrapper(agent_id: str, episode: Episode, worker, **kwargs):
            episode_id = episode.episode_id
            if episode_id not in self.episode_policies:
                self.episode_policies[episode_id] = select_policies(episode)

            return self.episode_policies[episode_id][agent_id]

        return policy_fn_mapper_wrapper

    def on_train_result(self, *, result: dict, algorithm: Algorithm, **kwargs):
        policy_mapping_fn = self._generate_policy_mapping_fn()

        # Add a new policy to train against later
        if result['training_iteration'] % self.save_policy_every == 0:
            # Need to prefix it with trial id because of some weird caching stuff RLlib does when restoring an agent.
            new_pol_id = f"{algorithm.trial_id}_t_{result['training_iteration']}"
            self.policy_scorer.add_policy.remote(new_pol_id)
            policy_mapping_fn = self._generate_policy_mapping_fn()

            temp_env = algorithm.env_creator(algorithm.config['env_config'])
            new_policy = algorithm.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(algorithm.get_policy(self.main_policy)),
                policy_mapping_fn=policy_mapping_fn,
                observation_space=temp_env.observation_space[f'player_0'],
                action_space=temp_env.action_space[f'player_0'],
            )
            main_state = algorithm.get_policy(self.main_policy).get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights to all the
            # remote workers as well.
            algorithm.workers.sync_weights(policies=[new_pol_id])
            print(f'Created new policy copy {new_pol_id}')

        def _set(worker):
            worker.set_policy_mapping_fn(policy_mapping_fn)
            worker.set_is_policy_to_train([self.main_policy])

        algorithm.workers.foreach_worker(_set)

        # Add policy stats
        policy_metrics = ray.get(self.policy_scorer.get_policy_metrics.remote())
        result['policy_scores'] = {p: policy_metrics[p]['score'] for p in policy_metrics}
        result['policy_games'] = {p: policy_metrics[p]['games'] for p in policy_metrics}
        result['policy_wins'] = {p: policy_metrics[p]['won'] for p in policy_metrics}
        result['policy_draws'] = {p: policy_metrics[p]['draw'] for p in policy_metrics}
        result['policy_losses'] = {p: policy_metrics[p]['lost'] for p in policy_metrics}
        result['policy_probabilities'] = ray.get(self.policy_scorer.get_policy_probabilities.remote())

        if result['training_iteration'] % self.print_policies_every == 0:
            self._print_policies()

    def _print_policies(self):
        policy_scores = ray.get(self.policy_scorer.get_policy_scores.remote())
        print(f'There are currently {len(policy_scores)} past policies with scores:')
        for name, score in policy_scores.items():
            print(f'{name}: {score}')

    def on_episode_end(self, **kwargs):
        episode: Episode = kwargs['episode']
        rewards = episode.agent_rewards
        main_policy_rewards = [rewards[t] for t in rewards if t[1] == self.main_policy]
        assert len(
            main_policy_rewards) > 0, f"Expected to use the current policy '{self.main_policy}', but got {rewards.keys()}"

        only_used_main_policy = all([policy == self.main_policy for _, policy in rewards.keys()])
        if not only_used_main_policy:
            self.policy_scorer.update_metrics.remote(rewards)

        super().on_episode_end(**kwargs)
