from typing import List, Optional, Dict

from ray.rllib import BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID
from ray.tune import Callback


class MetricsCallback(Callback):
    def on_step_end(self, iteration: int, trials: List["Trial"], **info):
        """Called at the start of each tuning loop step.

        Arguments:
            iteration: Number of iterations of the tuning loop.
            trials: List of trials.
            **info: Kwargs dict for forward compatibility.
        """
        ##print(iteration)
        return


class MetricsCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        # TODO
        pass
