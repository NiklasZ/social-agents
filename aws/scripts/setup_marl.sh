# Contains the commands necessary to prepare environment to run the social MARL agents.
# Run on server with dependencies installed.

# Example usage:
#     niklasz@leia:~$ source ./social-marl/social-agents/king_of_the_hill/scripts/setup_marl.sh

# Activate python dependency environment
export PATH="$HOME/miniconda3/bin:$HOME/bin:$PATH"
source ~/miniconda3/bin/activate
conda activate marl

# Ensure python has the right working directory.
export PYTHONPATH='/home3/niklasz/social-marl/social-agents'