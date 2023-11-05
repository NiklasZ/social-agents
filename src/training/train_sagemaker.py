# This contains the extra parser needed to handle AWS Sagemaker parameters.
# Sagemaker stores its training parameters in a JSON file /opt/ml/input/config/hyperparameters.json
# that only contains string  values. We need to reformat the JSON into a commandline string to feed into the training
# parser.
# For details, see https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
import argparse
import json
import os
from typing import List

from training.arg_parser import get_train_args
from training.train import train
from utils.helpers import dict_to_argparser_args


def parse_aws_hyperparameters(parameters: dict = None) -> List[str]:
    if parameters is None:
        config_path = '/opt/ml/input/config/hyperparameters.json'
        assert os.path.exists(config_path), f'No hyperparameters.json found at {config_path}'
        with open(config_path) as f:
            parameters = json.load(f)

        print(f'Found hyperparameters.json: \n{parameters}')

    arg_list = ['new'] + dict_to_argparser_args(parameters)
    # Sagemaker parameter JSONs keeps every value as a string
    arg_list = [a for a in arg_list if a != 'True']  # we need to remove the 'True' string for flags.
    arg_list = [s for a in arg_list for s in a.split(' ')]  # we need to split any multi-argument command.
    return arg_list


if __name__ == "__main__":
    training_args = parse_aws_hyperparameters()
    args = vars(get_train_args(training_args))
    train(args)
