# This script builds the docker image that can be run on AWS Sagemaker as a training job.

if [ "$(basename "$PWD")" != "social-agents" ]; then
    echo "Error: The current directory is not 'social-agents'. Please run this script from that directory"
    exit 1
fi

pwd

# To log more details run with flag --progress plain
docker build -t main -f aws/sagemaker/Dockerfile .

# To test it locally, try something like this:
# docker run --env WANDB_API_KEY=$WANDB_API_KEY --entrypoint python main src/training/train.py new --num-gpus 0 --num-workers 1 --agent-roles default default --env terro --num-gpus-per-worker 0 --map-name 15x15_minified --model-name baseline_cnn_lstm --episodes-per-training-iteration 1 --checkpoint-frequency 10 --model-sharing independent --training-iterations 1500 --population-self-play "{'population_per':2, 'min_games_against_new_opponent':1, 'recent_reward_window':2, 'curriculum_end_timesteps':5, 'sufficient_improved_population_ratio':0.25, 'policy_improvement_factor': 1.05}"