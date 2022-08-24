import ray
from ray.rllib.agents.ppo import ppo
from examples.rllib import utils
from king_of_the_hill.baseline.generate_config import get_alg_config, wrapped_env_constructor
from gym.wrappers import RecordVideo
from datetime import datetime
import tensorflow as tf

def evaluate():
    ray.init(local_mode=True)  # Run with local_mode=True to use debugger
    path = '/home/niklasz/ray_results/PPO/PPO_meltingpot_b1a7c_00000_0_2022-08-05_20-09-06/checkpoint_005150/checkpoint-5150'
    config = get_alg_config()
    agent = ppo.PPOTrainer(config=config)
    agent.restore(path)
    agent.get_policy('av').export_model('model')
    # This is a hacky way to save the keras model
    tf.saved_model.save(agent.get_policy('av').model.base_model, 'model')
    test_env = wrapped_env_constructor(config["env_config"])
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y-%H-%M-%S")
    recordable_env = RecordVideo(test_env, 'videos', video_length=1000, name_prefix=f'{timestamp}-') # 1000 steps is one episode
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

# pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl