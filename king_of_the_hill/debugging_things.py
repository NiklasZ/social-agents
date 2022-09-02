import json
from ray.tune.utils.serialization import TuneFunctionDecoder
# NOTE: this is a debugging investigation into why Ray will sometimes not use the most recent training iteration.
paths = sorted(['/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_16-00-08.json', '/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_16-05-00.json', '/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_16-12-41.json', '/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_16-10-42.json', '/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_15-58-00.json', '/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_15-49-41.json', '/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_14-46-59.json', '/tmp/ray_results/run-1661780977/experiment_state-2022-08-29_15-01-29.json'])

for p in paths:
    print(p
          )
    with open(p, "r") as f:
        runner_state = json.load(f, cls=TuneFunctionDecoder)
        last_checkpoint = json.loads(runner_state['checkpoints'][0])
        print(last_checkpoint['_last_result']['training_iteration'])