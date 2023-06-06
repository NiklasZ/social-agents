#!/bin/bash
# Contains the commands necessary to copy the project to leia or other NECL server
# usage examples:
#       copy_project_to.sh niklasz@leia.ee.ucla.edu /home3/niklasz/social-marl
#       copy_project_to.sh ubuntu@ec2-52-12-61-128.us-west-2.compute.amazonaws.com /home/ubuntu/marl
cd ~/Desktop/ucla/research_work # set directory from which to zip the project
./social-agents/king_of_the_hill/scripts/zip_project.sh # zip deployment files
scp ~/Desktop/ucla/research_work/social-agents.zip $1:$2 # copy files to server
ssh $1 unzip -o $2/social-agents.zip -d $2 # unzip deployment files into previous deployment directory


