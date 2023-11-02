#!/bin/bash
# Contains the commands necessary to copy the project to an NECL or AWS server
# usage examples:
#       copy_project_to.sh niklasz@leia.ee.ucla.edu /home3/niklasz/social-marl
#       copy_project_to.sh ubuntu@ec2-52-12-61-128.us-west-2.compute.amazonaws.com /home/ubuntu/marl

if [ "$(basename "$PWD")" != "social-agents" ]; then
    echo "Error: The current directory is not 'social-agents'. Please run this script from that directory"
    exit 1
fi

(cd .. && ./social-agents/aws/ec2/zip_project.sh) # zip deployment files
scp /tmp/social-agents.zip $1:$2 # copy files to server
ssh $1 unzip -o $2/social-agents.zip -d $2 # unzip deployment files into previous deployment directory
