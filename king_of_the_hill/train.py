import numpy as np
import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import tune
from king_of_the_hill.arg_parser import get_train_args
from king_of_the_hill.callbacks.enhanced_self_play import PolicyScorer
from king_of_the_hill.callbacks.pooling_self_play import PolicyPoolManager
from king_of_the_hill.callbacks.trial_state_logger import PrintResultCallback
from king_of_the_hill.generate_config import get_alg_config
from king_of_the_hill.models.models import get_cnn_lstm_constructor
from king_of_the_hill.wandb_api import get_api_key, download_train_state
from king_of_the_hill.wandb_logger_v2 import AnotherWandbLoggerCallback


# TODO make wandb optional
# TODO fix wandb logging on resume
def train(run_config: dict):
    # 6. Initialize ray, train and save

    # Local mode ensures the training is executed on a single process so a debugger can be attached
    num_gpus = int(np.ceil(run_config['num_gpus'] + run_config['num_gpus_per_worker'] * run_config['num_workers']))
    ray.init(local_mode=run_config['debug_mode'], num_gpus=num_gpus)
    agent_algorithm = 'PPO'
    rllib_path = run_config['rllib_path']
    api_key = get_api_key()
    manager = None

    # TODO figure out policy scorer on resume
    # TODO figure out policy manager on resume
    if run_config['command'] == 'resume':
        print(f"Resuming wandb run {run_config['run_id']}")
        config, run_config = download_train_state(rllib_path=rllib_path, api_key=api_key, run_config=run_config)

        results = tune.run(agent_algorithm,
                           config=config,
                           resume="LOCAL",
                           local_dir=rllib_path,
                           name=run_config['rllib_run_name'],
                           callbacks=[AnotherWandbLoggerCallback(api_key=api_key,
                                                                 project='social_marl_koth',
                                                                 group=run_config['group_name'],
                                                                 name=run_config['rllib_run_name'],
                                                                 resume='must',
                                                                 save_checkpoints=True
                                                                 )])
    else:
        config = get_alg_config(run_config)
        # Set up policy score tracking across workers
        if run_config['generalising_self_play']:
            manager = PolicyScorer.options(name='policy_scorer').remote('av')

        if run_config['pooled_self_play']:
            manager = PolicyPoolManager.options(name='policy_manager').remote(config, run_config)

        print(f"Training for {run_config['training_iterations']} iterations")

        results = tune.run(
            agent_algorithm,
            stop={'training_iteration': run_config['training_iterations']},
            config=config,
            metric='episode_reward_mean',
            local_dir=rllib_path,
            mode='max',
            name=run_config['rllib_run_name'],
            checkpoint_at_end=True,
            verbose=2 if run_config['debug_mode'] else 1,
            checkpoint_freq=run_config['checkpoint_frequency'],
            callbacks=[
                AnotherWandbLoggerCallback(api_key=api_key,
                                           project='social_marl_koth',
                                           name=run_config['rllib_run_name'],
                                           group=run_config['group_name'],
                                           # Note for some reason RLLib still dumps in the config at the top level of the
                                           # dictionary. Not sure why, but these keys are still in there, so this is at
                                           # worst a bit redundant.
                                           config={'run_config': run_config, 'full_config': config},
                                           save_checkpoints=True
                                           ),
                PrintResultCallback()
            ]
        )

    print('Training complete')

    # Clean up policy process
    if manager:
        ray.kill(manager)
    ray.shutdown()


if __name__ == "__main__":
    args = vars(get_train_args())
    train(args)
