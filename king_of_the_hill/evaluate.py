import ray
from ray.rllib.agents.ppo import ppo
from king_of_the_hill.generate_config import wrapped_env_constructor
from gym.wrappers import RecordVideo
from datetime import datetime

from king_of_the_hill.wandb_api import sync_from_wandb, get_api_key, find_run_folder


def evaluate():
    ray.init(local_mode=True)  # Run with local_mode=True to use debugger
    rllib_path = '/tmp/ray_results'
    checkpoint_folder = 'checkpoint_000290'
    checkpoint_name = 'checkpoint-290'
    api_key = get_api_key()
    run_config = {
        'run_id': '71a5c_00000',
        'entity_name': 'ece-239-as',
        'wandb_project_name': 'social_marl_koth',
        'rllib_run_name': 'run-1661950204'
    }
    config = sync_from_wandb(rllib_path=rllib_path, api_key=api_key, run_config=run_config,
                             sync_checkpoints=[checkpoint_folder])
    all_runs_dir = f"{rllib_path}/{run_config['rllib_run_name']}"
    run_folder = find_run_folder(all_runs_dir)
    path = f"{all_runs_dir}/{run_folder}/{checkpoint_folder}/{checkpoint_name}"
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
        action = agent.compute_actions(obs, policy_id='av')
        print(action)
        obs, reward, dones, _ = recordable_env.step(action)
        done = all(dones.values())

    recordable_env.close()


if __name__ == "__main__":
    evaluate()
