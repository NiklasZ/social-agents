# Introduction
This document describes how to install and run the project on an AWS EC2 virtual machine. It is written specifically
for the AWS **Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20231003** image. Note that AWS periodically updates and
replaces images (see [release notes](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-ubuntu-20-04/)),
so it may become unavailable after some time. In this case, it can still be found by using the AWS CLI using the command
```bash
aws ec2 describe-images --region us-west-2 --filters "Name=name,Values=Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20231003" --query 'Images[*].[ImageId]' --output text
# Yields ami-0c29f8386758cadf1
```
This AMI ID can then still be run via `aws ec2 run-instances   --image-id ami-0c29f8386758cadf1 ...`

# Image Setup
The AMI should be launched via EC2's web-interface or command-line. Afterwards, ssh into the server using the command:
```bash
ssh ubuntu@<server_address>.compute.amazonaws.com
 ```

Once logged in, you should verify that the following requirements are met:
1. Check that the major python version matches the project one via `python --version`.
2. Check that the CUDA version matches the project one via `nvcc --version`.

### If Python version does not match
First, install all the tools needed to compile the desired python version (in our case 3.10.9) and then create it.
1. `sudo apt update`
2. `sudo apt install build-essential zlib1g-dev libncurses5-dev libbz2-dev libffi-dev libreadline-dev libsqlite3-dev libssl-dev liblzma-dev unzip`
3. Install pyenv via `curl https://pyenv.run | bash`
4. Append following lines to `.bashrc`
```
# Custom additions
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
5. `source .bashrc`
6. `pyenv install -v 3.10.9` or whichever version is appropriate
7. `pyenv global 3.10.9`

### If CUDA version does not match
Usually it's not easy to reinstall CUDA for a virtual machine. There are some guides online, but they are not straightforward.
In this case it is usually better to use a different AMI.

### Install the project
1. Copy the project to the AWS server using the  utility script `copy_project_to.sh` (see file for usage).
2. Move to the directory of the project and follow the base [README.md](README.md)'s installation instructions.

