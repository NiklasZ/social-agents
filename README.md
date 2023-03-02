# Social MARL
This a project to find correlations in social representations between artificial agents.

# Installation
1. Please install Python >=3.9.
2. Install the project dependencies by running:
```
./king_of_the_hill/scripts/install_requirements.sh
```
3. Follow the installation instructions from Melting Pot's [README.md](MELTING_POT_README.md). Note: it's usually easiest to run:
```
./install.sh
```
which installs everything and runs Melting Pot's test suite.

3. Install the optional Rllib dependencies in [setup.py](setup.py). Via `pip`, this has the following command:
```
pip install -e .['rllib']
```
4. Install either `torch` or `tensorflow`. We currently use `torch~=1.11.0+cu113` (torch 1.11 with CUDA 11.3) 
although newer versions should also work.
6. Ensure you have access to the [wandb project](https://wandb.ai/ece-239-as/social_marl_koth) 
and create a file `king_of_the_hill/wandb_api_key.txt` and paste your API access key into it.

# Running
To run the project you should ensure the Python working directory is added to the PYTHONPATH:
```bash
export PYTHONPATH='<path_to_repository>/social-agents'
```

## Training
Training is controlled by [king_of_the_hill/train.py](king_of_the_hill/train.py) and has two modes:
- `new`, which starts a new training run
- `resume`, which continues an existing training run

The exact arguments possible for each mode can be obtained via `python king_of_the_hill/train.py new -h` and
`python king_of_the_hill/train.py resume -h` respectively. Below are example commands


### New command example
Here's an example training command:
```bash
python king_of_the_hill/train.py new --num-gpus 1 --map-name 13x13_minified_v2 --num-workers 8 --model-name basic_cnn_lstm --episodes-per-training-iteration 8 --checkpoint-frequency 50
```

This should create a run on `wandb` and save the run configuration and checkpoints there. 
Locally checkpoints are kept in `/tmp/ray_results` although that directory is emptied on every new machine start.
Lastly, the run should also generate a unique run id shaped like this one: `8c82d_00000`.

### Resume command example
Using the previously created run id the training can be resumed as follows:
```bash
python king_of_the_hill/train.py resume --run-id 8c82d_00000
```

This works reasonably well, although rllib will often choose to go back a few training iterations for stability. This 
can mess up `wandb` charts a bit as now multiple datapoints refer to the same training iteration, 
but it is not a huge concern.


## Evaluation
Once a run has at least one checkpoint, we can evaluate agents with `python king_of_the_hill/evaluate.py`, which can
play the learned policies of the agents for multiple games, record them and also track their metrics. Here is an example
evaluation command:
```bash
python king_of_the_hill/evaluate.py --run-id 8c82d_00000 --checkpoint-number 1 --episodes 3 --record-observations --recording-fps 5
```

Which pulls the checkpoint of the 1st training iteration and evaluates 3 episodes. Notably, videos and other collected metrics
will be put into [king_of_the_hill/evaluation](king_of_the_hill/evaluation). For more options, run `python king_of_the_hill/evaluate.py -h`

## AWS
This setup can also be run AWS or any similar cloud service. For further details see [AWS_SETUP_NOTES.md](AWS_SETUP_NOTES.md)