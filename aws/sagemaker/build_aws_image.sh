# This script builds the docker image that can be run on AWS Sagemaker as a training job.

if [ "$(basename "$PWD")" != "social-agents" ]; then
    echo "Error: The current directory is not 'social-agents'. Please run this script from that directory"
    exit 1
fi

pwd

# To log more details run with flag --progress plain
docker build -t main -f aws/sagemaker/Dockerfile .

# To test it locally, try something like this:
# docker run --env WANDB_API_KEY=$WANDB_API_KEY --entrypoint python main ./training/train.py new --num-gpus 0 --num-workers 4 --agent-roles predator prey --env papo --num-gpus-per-worker 0 --map-name 13x13_example --model-name baseline_cnn_lstm --episodes-per-training-iteration 4 --checkpoint-frequency 10 --multi-model --training-iterations 1500