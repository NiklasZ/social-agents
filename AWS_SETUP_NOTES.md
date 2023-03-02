# Introduction
This project can also be configured to run on AWS. For this we use the **Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230222** 
Amazon image. To get things running, create a base AMI using the first section and then create a custom AMI based off of it.
This custom AMI can then be used directly to launch an EC2 server with everything preconfigured. 

# Custom image setup
This section covers how to create the custom AMI for future use. Note that this could also be easily converted into a
single executable script if needed.

### Install Python
First, install all the tools needed to compile the desired python version (in our case 3.9.6) and then create it.
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

5.`source .bashrc`
6. `pyenv install -v 3.9.6`
7. `pyenv global 3.9.6`

### Prepare virtual env
Now create a virtual python to install project dependencies later.
3. `mkdir marl`
4. `cd marl`
5. Copy project data over via: `./king_of_the_hill/scripts/copy_project_to.sh <user>@<url> <remote_dir>`. An example:
```
./king_of_the_hill/scripts/copy_project_to.sh ubuntu@ec2-34-220-207-223.us-west-2.compute.amazonaws.com /home/ubuntu/marl
```
6. `pyenv virtualenv marl`
7. `pyenv shell marl`

### Install project dependencies
This is pretty much identical to the base [README.md](README.md)'s instructions.
1. `./king_of_the_hill/scripts/install_requirements.sh`
2.  Install bazel
```
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
```
3. Install melting pot `./install.sh`
4. Install rllib: `pip install -e .['rllib']`
5. Install torch `pip3 install torch==1.13 torchvision==1.13 torchaudio==1.13`. Note that this is not necessarily the
torch version as we have installed locally, but this is constrained by the GPU used by AWS. Also, torch is well-designed,
so there is some leniency with versions here.

### Create the Image
Create the custom image in the AWS EC2 interface (see https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html)

# Running the Custom Image
1. Create an EC2 server based on the custom image and allocate according resources.
2. `ssh` into the server (i.e `ssh <user>@<url>`).
3. `cd marl/social_agents`
4. Start the virtual env `pyenv shell marl`
5. And now train as usual.