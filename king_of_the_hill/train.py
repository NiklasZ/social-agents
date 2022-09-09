import ray
from ray.tune import tune
from king_of_the_hill.arg_parser import get_train_args
from king_of_the_hill.generate_config import get_alg_config
from king_of_the_hill.wandb_api import get_api_key, download_train_state
from king_of_the_hill.wandb_logger_v2 import AnotherWandbLoggerCallback


# TODO make wandb optional
# TODO fix wandb logging on resume
def train(run_config: dict):
    # 6. Initialize ray, train and save

    # Local mode ensures the training is executed on a single process so a debugger can be attached
    ray.init(local_mode=run_config['debug_mode'])
    agent_algorithm = "PPO"
    rllib_path = '/tmp/ray_results'
    api_key = get_api_key()

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
        results = tune.run(
            agent_algorithm,
            stop={'training_iteration': run_config['training_iterations']},
            config=config,
            metric='episode_reward_mean',
            local_dir=rllib_path,
            mode='max',
            name=run_config['rllib_run_name'],
            checkpoint_at_end=True,
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
                                           )
            ]
        )

    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean'),
                                                      metric='episode_reward_mean')

    if len(checkpoints) > 0:
        checkpoint_path, checkpoint_reward = checkpoints[0]
        print(f'Training complete. Best model found in {checkpoint_path} with reward {checkpoint_reward}')


if __name__ == "__main__":
    args = vars(get_train_args())
    train(args)
