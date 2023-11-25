#!/usr/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/shawn/Documents/social-agents/src:/home/shawn/Documents/social-agents/src/deepmind"
export WANDB_API_KEY="c047e297f7902f8f688b677c5d434bf03340a6fc"

# python src/training/train.py new \
#   --num-workers 1 \
#   --agent-roles predator prey \
#   --env predator_and_prey_open \
#   --map-name 13x13_example \
#   --model-name baseline_cnn_lstm \
#   --episodes-per-training-iteration 4 \
#   --checkpoint-frequency 20 \
#   --multi-model \
#   --training-iterations 20 \
#   --num-gpus 1 

python src/training/train.py new \
  --num-workers 1 \
  --agent-roles default default \
  --env territory_open \
  --model-name baseline_cnn_lstm \
  --episodes-per-training-iteration 4 \
  --checkpoint-frequency 20 \
  --multi-model \
  --training-iterations 20 \
  --num-gpus 1 