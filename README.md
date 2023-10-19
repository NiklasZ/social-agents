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

3. Install the project dependencies. Note: we presently use `torch` with CUDA 11.7. You may need to install a 
different torch binary to support your CUDA version. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

# Running

To run the project you should ensure the following two working directories are added to your `PYTHONPATH`. Note,
you need to fill in `<dir_to_project>` here.

```bash
export PYTHONPATH="${PYTHONPATH}:/<dir_to_project>/social-agents/src:/<dir_to_project>/social-agents/src/deepmind"
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
  --multi-model \
  --training-iterations 1500
```
The above command trains the agents on the CPU only. To use the GPU, specify `--num-gpus 1` as well.


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
  --policies "{'policy_id':'policy0', 'run_id':'7558f_00000', 'checkpoint':10}" \
            "{'policy_id':'policy1', 'run_id':'7558f_00000', 'checkpoint':10}" \
  --record-episodes \
  --recording-fps 15
  ```

Where the above pulls the specified policies from wandb and has them play against each other on the environment 
they trained on.

## AWS

Coming soon...