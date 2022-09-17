from typing import Dict

import ray
from ray.rllib.agents.ppo import ppo
from king_of_the_hill.arg_parser import get_evaluation_args
from king_of_the_hill.generate_config import get_trailing_number
from gym.wrappers import RecordVideo
from datetime import datetime
from king_of_the_hill.wandb_api import get_api_key, download_evaluation_state
from king_of_the_hill.models import get_model


def evaluate(run_config: Dict):
    ray.init(local_mode=True)  # Run with local_mode=True to use debugger
    rllib_path = '/tmp/ray_results'
    checkpoint_folder = f"checkpoint_{run_config['checkpoint_number']:06}"
    checkpoint_name = f"checkpoint-{run_config['checkpoint_number']}"
    api_key = get_api_key()
    checkpoint_path, config, run_config = download_evaluation_state(rllib_path=rllib_path,
                                                                    run_config=run_config,
                                                                    api_key=api_key,
                                                                    checkpoint_folder=checkpoint_folder)
    path = f"{checkpoint_path}/{checkpoint_name}"
    agent = ppo.PPOTrainer(config=config)
    agent.restore(path)
    model_description = get_model(run_config['model_name'])
    env_constructor = model_description['env_constructor']
    test_env = env_constructor(config["env_config"])
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y-%H-%M-%S")
    recordable_env = RecordVideo(test_env, f"videos/{run_config['rllib_run_name']}", video_length=1000,
                                 name_prefix=f'{checkpoint_name}-{timestamp}-')  # 1000 steps is one

    for episode in range(run_config['episodes']):
        print(f'==Episode {episode}==')
        done = False
        obs = recordable_env.reset()

        while not done:
            if run_config['multi_model']:
                actions = {}
                for policy in config['multiagent']['policies']:
                    id = get_trailing_number(policy)
                    player = f'player_{id}'
                    action = agent.compute_single_action(obs[player], policy_id=policy)
                    if run_config['verbose']:
                        print(player, policy, action)
                    actions[player] = action
                obs, reward, dones, _ = recordable_env.step(actions)
                done = all(dones.values())

            else:
                action = agent.compute_actions(obs, policy_id='av')
                if run_config['verbose']:
                    print(action)
                obs, reward, dones, _ = recordable_env.step(action)
                done = all(dones.values())

        recordable_env.close_video_recorder()


if __name__ == "__main__":
    args = vars(get_evaluation_args())
    evaluate(args)
