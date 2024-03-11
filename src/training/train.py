from typing import Mapping
import numpy as np
import ray
from ray.tune import tune
from src.training.arg_parser import get_train_args
from src.training.callbacks.trial_state_logger import PrintResultCallback
from src.training.wandb_logger_v2 import AnotherWandbLoggerCallback
from src.utils.generate_config import get_alg_config
from training.callbacks.population_self_play import PopulationManager, has_at_least_one_policy_per_player


# TODO make wandb optional
def train(run_config: dict):
    # 6. Initialize ray, train and save

    # Local mode ensures the training is executed on a single process so a debugger can be attached
    num_gpus = int(np.ceil(run_config['num_gpus'] + run_config['num_gpus_per_worker'] * run_config['num_workers']))
    ray.init(local_mode=run_config['debug_mode'], num_gpus=num_gpus)
    agent_algorithm = 'PPO'
    rllib_path = run_config['rllib_path']
    manager = None

    if run_config['command'] == 'resume':
        raise Exception('Resume has not been maintained and needs to be re-implemented.')
        # print(f"Resuming wandb run {run_config['run_id']}")
        # config, run_config = download_train_state(rllib_path=rllib_path, api_key=api_key, run_config=run_config)
        #
        # results = tune.run(agent_algorithm,
        #                    config=config,
        #                    resume="LOCAL",
        #                    local_dir=rllib_path,
        #                    name=run_config['rllib_run_name'],
        #                    callbacks=[AnotherWandbLoggerCallback(api_key=api_key,
        #                                                          project='social_marl_koth',
        #                                                          group=run_config['group_name'],
        #                                                          name=run_config['rllib_run_name'],
        #                                                          resume='must',
        #                                                          save_checkpoints=True
        #                                                          )])
    else:
        config = get_alg_config(run_config)

        stopping_conditions = []
        stopping_func = lambda trial_id, result: any([condition(trial_id, result) for condition in stopping_conditions])
        stop_afer_k_iterations = lambda trial_id, result: result['training_iteration'] >= run_config[
            'training_iterations']
        stopping_conditions.append(stop_afer_k_iterations)

        # Set up policy score tracking across workers
        if run_config['population_self_play']:
            manager = PopulationManager.options(name='population_manager').remote(config, run_config)

            # Stop when we no longer have a training policy per player
            def stop_when_out_of_pops(trial_id: str, result: Mapping) -> bool:
                return not has_at_least_one_policy_per_player(list(result['curriculum_avg_rewards'].keys()),
                                                              len(run_config['agent_roles']))

            stopping_conditions.append(stop_when_out_of_pops)

        print(f"Training for {run_config['training_iterations']} iterations")

        results = tune.run(
            agent_algorithm,
            stop=stopping_func,  # {'training_iteration': run_config['training_iterations']},
            config=config,
            metric='episode_reward_mean',
            local_dir=rllib_path,
            mode='max',
            name=run_config['rllib_run_name'],
            checkpoint_at_end=True,
            verbose=2 if run_config['debug_mode'] else 1,
            checkpoint_freq=run_config['checkpoint_frequency'],
            callbacks=[
                AnotherWandbLoggerCallback(project='social_marl_koth',
                                           name=run_config['rllib_run_name'],
                                           group=run_config['group_name'],
                                           # Note for some reason RLLib still dumps in the config at the top level of the
                                           # dictionary. Not sure why, but these keys are still in there, so this is at
                                           # worst a bit redundant.
                                           config={'run_config': run_config, 'full_config': config},
                                           save_checkpoints=True,
                                           id=run_config['run_id'],
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
