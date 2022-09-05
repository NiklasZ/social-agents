import ray
from ray.rllib.agents.ppo import ppo
from king_of_the_hill.generate_config import wrapped_env_constructor, get_trailing_number
from gym.wrappers import RecordVideo
from datetime import datetime
from king_of_the_hill.wandb_api import get_api_key, download_evaluation_state


# TODO argparser
def evaluate():
    ray.init(local_mode=True)  # Run with local_mode=True to use debugger
    rllib_path = '/tmp/ray_results'
    checkpoint_folder = 'checkpoint_001000'
    checkpoint_name = 'checkpoint-1000'
    api_key = get_api_key()
    run_config = {
        'run_id': 'cf895_00000',
        'entity_name': 'ece-239-as',
        'wandb_project_name': 'social_marl_koth',
        'rllib_run_name': 'run-1662129890'
    }
    checkpoint_path, config, run_config = download_evaluation_state(rllib_path=rllib_path,
                                                                    run_config=run_config,
                                                                    api_key=api_key,
                                                                    checkpoint_folder=checkpoint_folder)
    # all_runs_dir = f"{rllib_path}/{run_config['rllib_run_name']}"
    # run_folder = find_run_folder(all_runs_dir)
    path = f"{checkpoint_path}/{checkpoint_name}"
    agent = ppo.PPOTrainer(config=config)
    agent.restore(path)
    test_env = wrapped_env_constructor(config["env_config"])
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y-%H-%M-%S")
    recordable_env = RecordVideo(test_env, 'videos', video_length=1000,
                                 name_prefix=f'{timestamp}-')  # 1000 steps is one episode
    done = False
    obs = recordable_env.reset()

    while not done:

        if run_config['multi_model']:
            actions = {}
            for policy in config['multiagent']['policies']:
                id = get_trailing_number(policy)
                player = f'player_{id}'
                action = agent.compute_single_action(obs[player], policy_id=policy)
                print(player, policy, action)
                actions[player] = action
            obs, reward, dones, _ = recordable_env.step(actions)
            done = all(dones.values())

        else:
            action = agent.compute_actions(obs, policy_id='av')
            print(action)
            obs, reward, dones, _ = recordable_env.step(action)
            done = all(dones.values())

    recordable_env.close()


if __name__ == "__main__":
    evaluate()
