# This is a script that allows running a training job on AWS Sagemaker. After running, a job should appear in:
# https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs
# The below config dictionaries allow picking training parameters, resources and more.
# See https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html
# and https://medium.com/@smertatli/aws-sagemaker-is-one-of-the-most-advanced-machine-learning-services-in-the-cloud-world-46ff67d45c0
# for configuration guidance

from datetime import datetime
import boto3
import os
from src.training.arg_parser import get_train_args
from src.training.train_sagemaker import parse_aws_hyperparameters

# Training parameters
hyperparameters = {'num-gpus': 1,
                   'num-workers': 1,
                   'agent-roles': 'default default',
                   'env': 'koth',
                   'num-gpus-per-worker': 0,
                   'map-name': 'default',
                   'model-name': 'baseline_cnn_lstm',
                   'episodes-per-training-iteration': 1,
                   'checkpoint-frequency': 10,
                   'multi-model': True,
                   'training-iterations': 1500,
                   }

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

training_job_name = f"{hyperparameters['env']}-job-{timestamp}"

# Which docker image to use for training. Usually the "latest" is fine
algorithm_specification = {
    'TrainingImage': '931357261381.dkr.ecr.us-west-2.amazonaws.com/social-agents:latest',
    'TrainingInputMode': 'File',
    'ContainerEntrypoint': ['python', 'src/training/train_sagemaker.py'],
}

# Executing role for the job. This can be the same for all runs
role_name = 'arn:aws:iam::931357261381:role/service-role/AmazonSageMaker-ExecutionRole-20231031T180269'

# Where training outputs are saved. Required by AWS, but not used for much.
output_data_config = {
    'S3OutputPath': f's3://social-agents/output-f{timestamp}'
}

# What machine type to run.
resource_config = {
    'InstanceType': 'ml.g4dn.2xlarge',
    'InstanceCount': 1,
    'VolumeSizeInGB': 50
}

# Termination conditions for the job.
stop_condition = {
    'MaxRuntimeInSeconds': 60 * 60 * 24 * 5  # 5 days max runtime
}

environment_vars = {'WANDB_API_KEY': os.getenv('WANDB_API_KEY')}


# Note: Sagemaker hyperparameters must be string values
hyperparameters = {k: str(v) for k, v in hyperparameters.items()}

# Parse and validate the hyperparameters. Saves some time by checking for basic errors before launching a job
training_args = parse_aws_hyperparameters(hyperparameters)
_ = vars(get_train_args(training_args))

sagemaker_client = boto3.client(
    "sagemaker",
    region_name='us-west-2',
)

sagemaker_client.create_training_job(
    TrainingJobName=training_job_name,
    AlgorithmSpecification=algorithm_specification,
    RoleArn=role_name,
    OutputDataConfig=output_data_config,
    ResourceConfig=resource_config,
    StoppingCondition=stop_condition,
    Environment=environment_vars,
    HyperParameters=hyperparameters
)
