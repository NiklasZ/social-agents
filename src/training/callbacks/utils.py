from typing import List, Callable, Any

import ray
from ray.rllib import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet


def apply_to_worker_i(workers: WorkerSet, func: Callable[[RolloutWorker], Any], index: int):
    """
    Applies a function to a worker at a given index.
    @param workers:
    @param func:
    @param index:
    @return:
    """
    # Local worker: Index=0.
    if workers.local_worker() is not None and index == 0:
        return func(workers.local_worker())

    # Remote workers: Index >= 0.
    remote_workers = list(workers.remote_workers())
    remote_result = ray.get(remote_workers[index - 1].apply.remote(func))
    return remote_result
