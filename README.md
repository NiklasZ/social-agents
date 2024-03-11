# Social MARL

This a project to find correlations in social representations between artificial agents. For our environments we use
the [Melting Pot](https://github.com/google-deepmind/meltingpot) suite, whose installation is
under [src/deepmind](src/deepmind).

# Installation

1. Please install Python 3.10.x. This project has a lot of dependencies, so it is recommended to create a VENV for it.
2. Install Melting Pot's dependencies and let it download assets needed to use the environments. To do this, run:

```bash
(cd src/deepmind && pip install --editable .[dev])
```

3. Install the project dependencies. Note: we presently use `torch` with CUDA 12.1. You may need to install a 
different torch binary to support your CUDA version. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

# Running

To run the project you should ensure the following two working directories are added to your `PYTHONPATH`. Note, that
`${PWD}` is assumed to be the root of this project.

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src:${PWD}/src/deepmind"
```
You will also need to set up wandb to track your runs or use the [existing workspace](https://wandb.ai/ece-239-as/social_marl_koth).
To authenticate yourself, please set your wandb access key:
```bash
export WANDB_API_KEY=<your_api_key>
```

## Training

Training is controlled by [src/training/train.py](src/training/train.py) and has an arg-parser which you can query with:
```bash
python src/training/train.py new --help
```

Here is an example command to use to train a pair of agents in the `predator_and_prey` environment:
```bash
python src/training/train.py new \
  --num-workers 1 \
  --agent-roles predator prey \
  --env predator_and_prey_open \
  --map-name 13x13_example \
  --model-name baseline_cnn_lstm \
  --episodes-per-training-iteration 4 \
  --checkpoint-frequency 100 \
  --model-sharing independent \
  --training-iterations 1500
```
The above command trains the agents on the CPU only. To use the GPU, specify `--num-gpus 1` as well.

In order to train a population of agents together the additional `--population-self-play` must be specified. Here an
example for the `king_of_the_hill` environment:
```bash
python src/training/train.py new \
  --num-workers 1 \
  --agent-roles default default \
  --env king_of_the_hill \
  --map-name 13x13_minified_v2 \
  --model-name baseline_cnn_lstm \
  --episodes-per-training-iteration 8 \
  --checkpoint-frequency 200 \
  --model-sharing independent \
  --training-iterations 2500 \
  --population-self-play "{'population_per':8, 
                        'min_games_against_new_opponent':10, 
                        'recent_reward_window':30, 
                        'curriculum_end_timesteps':50, 
                        'sufficient_improved_population_ratio':0.5, 
                        'policy_improvement_factor': 1.1}"
```

## Evaluation

Once a run has at least one checkpoint, we can evaluate agents with [src/evaluation/evaluate.py](src/evaluation/evaluate.py), which can
play the learned policies of the agents for multiple games, record them and also track their metrics. The tool can be 
queried with:
nd has an arg-parser which you can query with:
```bash
python src/evaluation/evaluate.py --help
```

Here is an example command to use to evaluate a pair of agents in the environment:
```bash
python src/evaluation/evaluate.py \
  --episodes 3 \
  --num-gpus 1 \
  --players "{'policy_id':'policy_0_p1', 'run_id':'run-fd734f6e5', 'checkpoint':2000, 'role':'default'}" \
            "{'policy_id':'policy_1_p0', 'run_id':'run-fd734f6e5', 'checkpoint':2000, 'role':'default'}"
  ```

Here is an example command to use to evaluate trained agents with Meltingpot-provided agents
```bash
python src/evaluation/evaluate.py \
  --episodes 3 \
  --num-gpus 1 \
  --players "{'policy_id':'policy_0_p0', 'run_id':'run-fd32be957', 'checkpoint':2500, 'role':'default'}" \
            "{'policy_id':'policy_1_p1', 'run_id':'run-fd32be957', 'checkpoint':2500, 'role':'default'}" \
  --meltingpot-bots "{'bot_name': 'territory__open__aggressor_0'}" \
                    "{'bot_name': 'territory__open__aggressor_1'}" 
```

Where the above pulls the specified policies from wandb and has them play against each other on the environment 
they trained on. Other useful flags for evaluation include `--record-episodes` to save videos of the agents and
`--activations "{'layer_path':'lstm', 'activation_path':'1.0', 'alias':'lstm_hidden'}"`, which will save the 
activations of the specified layer. In the above example, the layer is the LSTM with the hidden state from the output.
All evaluations will be saved locally to `eval_games` with results in a `metrics.pkl` file.

## AWS

The project can be run on AWS EC2 services and Sagemaker. See [aws](aws) for some instructions.
