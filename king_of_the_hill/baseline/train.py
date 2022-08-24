# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs an example of a self-play training experiment."""
import shutil
from time import time
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import tune
from king_of_the_hill.baseline.arg_parser import get_run_args
from king_of_the_hill.baseline.generate_config import get_alg_config
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger, WandbLoggerCallback
from king_of_the_hill.baseline.wandb_logger_v2 import AnotherWandbLoggerCallback
import wandb
import os


def train(run_config: dict):
    # 6. Initialize ray, train and save

    # Local mode ensures the training is executed on a single process so a debugger can be attached
    ray.init(local_mode=run_config['debug_mode'])
    agent_algorithm = "PPO"
    folder = 'king_of_the_hill'
    folder_idx = __file__.index(folder) + len(folder)
    folder_path = __file__[:folder_idx]

    if run_config['command'] == 'resume':
        print(f"Resuming wandb run {run_config['run_id']}")
        api = wandb.Api()
        run_path = f"{run_config['entity_name']}/{run_config['wandb_project_name']}/{run_config['run_id']}"
        run = api.run(run_path)
        combined_config = run.config
        run_config = combined_config['run_config']
        config = get_alg_config(run_config)
        all_runs_dir = f"/home/niklasz/ray_results/{run_config['rllib_run_name']}"
        if not os.path.exists(all_runs_dir):
            os.makedirs(all_runs_dir)

        with wandb.restore('info.txt', run_path=run_path, root=all_runs_dir) as f:
            run_folder, run_timestamp, checkpoint_folder_name, tf_logging_file_name = f.read().split('\n')

        wandb.restore(f'basic-variant-state-{run_timestamp}.json', run_path=run_path, root=all_runs_dir)
        wandb.restore(f'experiment_state-{run_timestamp}.json', run_path=run_path, root=all_runs_dir)

        run_dir = f'{all_runs_dir}/{run_folder}'
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        wandb.restore(f'{tf_logging_file_name}', run_path=run_path, root=run_dir)
        wandb.restore(f'params.json', run_path=run_path, root=run_dir)
        wandb.restore(f'params.pkl', run_path=run_path, root=run_dir)
        wandb.restore(f'progress.csv', run_path=run_path, root=run_dir)
        wandb.restore(f'result.json', run_path=run_path, root=run_dir)

        wandb.restore(f'{checkpoint_folder_name}.zip', run_path=run_path, root=run_dir)
        shutil.unpack_archive(f'{run_dir}/{checkpoint_folder_name}.zip', extract_dir=run_dir)

        results = tune.run(agent_algorithm,
                           config=config,
                           resume="LOCAL",
                           name=run_config['rllib_run_name'],
                           callbacks=[AnotherWandbLoggerCallback(api_key_file=f'{folder_path}/wandb_api_key.txt',
                                                                 project='social_marl_koth',
                                                                 group=run_config['group_name'],
                                                                 name=run_config['rllib_run_name'],
                                                                 resume='must',
                                                                 save_checkpoints=True
                                                                 )])
    else:
        config = get_alg_config(run_config)
        results = tune.run(
            agent_algorithm,
            stop={'training_iteration': run_config['training_iterations']},
            config=config,
            metric='episode_reward_mean',
            mode='max',
            name=run_config['rllib_run_name'],
            checkpoint_at_end=True,
            checkpoint_freq=run_config['checkpoint_frequency'],
            callbacks=[
                AnotherWandbLoggerCallback(api_key_file=f'{folder_path}/wandb_api_key.txt',
                                           project='social_marl_koth',
                                           name=run_config['rllib_run_name'],
                                           group=run_config['group_name'],
                                           # Note for some reason RLLib still dumps in the config at the top level of the
                                           # dictionary. Not sure why, but these keys are still in there, so this is at
                                           # worst a bit redundant.
                                           config={'run_config': run_config, 'full_config': config},
                                           save_checkpoints=True
                                           )
            ]
        )

    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean'),
                                                      metric='episode_reward_mean')
    checkpoint_path, checkpoint_reward = checkpoints[0]
    print(f'Training complete. Best model found in {checkpoint_path} with reward {checkpoint_reward}')


if __name__ == "__main__":
    args = vars(get_run_args())
    train(args)
