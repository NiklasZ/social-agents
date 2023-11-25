#!/usr/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/shawn/Documents/social-agents/src:/home/shawn/Documents/social-agents/src/deepmind"
export WANDB_API_KEY="c047e297f7902f8f688b677c5d434bf03340a6fc"

# python src/evaluation/evaluate.py \
#   --episodes 1 \
#   --num-gpus 1 \
#   --players "{'policy_id':'policy0', 'run_id':'f541e_00000', 'checkpoint':20, 'role':'default'}" \
#             "{'policy_id':'policy1', 'run_id':'f541e_00000', 'checkpoint':20, 'role':'default'}" \
#   --meltingpot-bots "{'bot_name': 'territory__open__aggressor_0'}" \
#   --record-episodes \
#   --recording-fps 15

python src/evaluation/evaluate.py \
  --episodes 3 \
  --num-gpus 1 \
  --players "{'policy_id':'policy0', 'run_id':'f541e_00000', 'checkpoint':20, 'role':'default'}" \
            "{'policy_id':'policy1', 'run_id':'f541e_00000', 'checkpoint':20, 'role':'default'}" \
  --meltingpot-bots "{'bot_name': 'territory__open__aggressor_0'}" \
  --record-episodes \
  --recording-fps 15

