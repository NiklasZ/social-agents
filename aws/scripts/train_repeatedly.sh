#!/bin/bash

# Usage: ./run_python_script.sh script_name.py k --arg1 value1 --arg2 value2 ...
# An example: ./king_of_the_hill/scripts/train_repeatedly.sh king_of_the_hill/train.py 3 new --num-gpus 1 --map-name 13x13_minified_v2 --num-workers 4 --model-name chunky_cnn_lstm --episodes-per-training-iteration 4 --checkpoint-frequency 5 --multi-model --pooled-self-play "{'evaluation_num_workers': 2, 'consecutive_training_wins_needed': 4, 'consecutive_eval_wins_needed':5, 'reward_distance': 100, 'mastered_pool_sampling':'inverse last minimum train reward','mastered_pool_max_prob':0.25}" --training-iterations 30
# Note that for the example, you will need to run export PYTHONPATH=$PWD first.

# Check if at least two arguments are provided (script name and k)
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 script_name.py k [--arg1 value1 --arg2 value2 ...]"
    exit 1
fi

# Get the script name and k value
script_name=$1
k=$2

# Remove the script name and k value from the arguments list
shift 2

# Run the Python script k times with the remaining arguments
for i in $(seq 1 "$k"); do
    echo "Running iteration $i"
    python "$script_name" "$@"
done