FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
LABEL authors="niklasz"

# This will output the python version if the "docker build" command is run with --progress plain.
# Make sure it matches the development version
RUN python --version

# Set the working directory and ensure python knows about it.
WORKDIR /social-agents
ENV PYTHONPATH="${PYTHONPATH}:/social-agents/src"
ENV PYTHONPATH="${PYTHONPATH}:/social-agents/src/deepmind"

# Install misc dependencies used by python dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy only requirements.txt first. We do this so we can cache the installation of dependencies and avoid long waits.
COPY requirements.txt /social-agents/requirements.txt

# Install python dependencies
RUN pip install -r /social-agents/requirements.txt

# Copy project files, skipping those in .dockerignore
COPY . /social-agents
