#python src/evaluation/evaluate.py \
#  --episodes 10 \
#  --num-gpus 1 \
#  --players "{'policy_id':'policy_0', 'run_id':'run-13e93f6cf3', 'checkpoint':1550, 'role':'predator'}" \
#            "{'policy_id':'policy_1', 'run_id':'run-13e93f6cf3', 'checkpoint':1550, 'role':'predator'}" \
#            "{'policy_id':'policy_2', 'run_id':'run-13e93f6cf3', 'checkpoint':1550, 'role':'prey'}" \
#            "{'policy_id':'policy_3', 'run_id':'run-13e93f6cf3', 'checkpoint':1550, 'role':'prey'}" \
#            "{'policy_id':'policy_4', 'run_id':'run-13e93f6cf3', 'checkpoint':1550, 'role':'prey'}" \
#            "{'policy_id':'policy_5', 'run_id':'run-13e93f6cf3', 'checkpoint':1550, 'role':'prey'}" \
#  --rllib-path /e/predator_prey/open_field_10x10/trial_1/ \
#  --entity-name qinli2021-ucla \
#  --record-episodes --record-observations


#python src/evaluation/evaluate_1v1.py \
#  --episodes 10 \
#  --num-gpus 0 \
#  --players "{'policy_id':'policy_0', 'run_id':'run-13fe8b91cf', 'checkpoint':1550, 'role':'predator'}" \
#            "{'policy_id':'policy_2', 'run_id':'run-13fe8b91cf', 'checkpoint':1550, 'role':'prey'}" \
#  --rllib-path /e/predator_prey/open_field_10x10/trial_1/ \
#  --entity-name qinli2021-ucla \
#  --record-episodes --record-observations

python src/evaluation/evaluate_1v1.py \
  --episodes 10 \
  --num-gpus 0 \
  --players "{'policy_id':'policy_0', 'run_id':'run-1436a49ed6', 'checkpoint':5000, 'role':'predator'}" \
            "{'policy_id':'policy_1', 'run_id':'run-1436a49ed6', 'checkpoint':5000, 'role':'prey'}" \
  --rllib-path /e/predator_prey/open_field_10x10/1v1_baseline_addobs_trial_1/ \
  --entity-name qinli2021-ucla \
  --record-episodes --record-observations


python src/evaluation/evaluate_1v1.py \
  --episodes 10 \
  --num-gpus 0 \
  --players "{'policy_id':'policy_0', 'run_id':'run-1436ac65f6', 'checkpoint':5000, 'role':'predator'}" \
            "{'policy_id':'policy_1', 'run_id':'run-1436ac65f6', 'checkpoint':5000, 'role':'prey'}" \
  --rllib-path /e/predator_prey/open_field_10x10/1v1_baseline_addobs_trial_2/ \
  --entity-name qinli2021-ucla \
  --record-episodes --record-observations


python src/evaluation/evaluate_1v1.py \
  --episodes 10 \
  --num-gpus 0 \
  --players "{'policy_id':'policy_0', 'run_id':'run-1436a381d8', 'checkpoint':5000, 'role':'predator'}" \
            "{'policy_id':'policy_1', 'run_id':'run-1436a381d8', 'checkpoint':5000, 'role':'predator'}" \
            "{'policy_id':'policy_2', 'run_id':'run-1436a381d8', 'checkpoint':5000, 'role':'prey'}" \
            "{'policy_id':'policy_3', 'run_id':'run-1436a381d8', 'checkpoint':5000, 'role':'prey'}" \
            "{'policy_id':'policy_4', 'run_id':'run-1436a381d8', 'checkpoint':5000, 'role':'prey'}" \
            "{'policy_id':'policy_5', 'run_id':'run-1436a381d8', 'checkpoint':5000, 'role':'prey'}" \

  --rllib-path /e/predator_prey/open_field_10x10/2v4_baseline_addobs/ \
  --entity-name qinli2021-ucla \
  --record-episodes --record-observations