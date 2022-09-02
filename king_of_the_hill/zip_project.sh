# Run from parent directory of repository root (i.e /<this_dir>/social-agents/king_of_the_hill)
# -x directories are skipped so as to keep the resulting ZIP as small as possible.
zip -r social-agents.zip social-agents -x \
social-agents/venv/**\* \
social-agents/build/**\* \
social-agents/lab2d/**\*  \
social-agents/examples/results/**\*  \
social-agents/meltingpot/assets/**\* \
social-agents/meltingpot/docs/**\* \
social-agents/.git/**\* \
social-agents/.pytest_cache/**\* \
social-agents/docs/**\* \
social-agents/wandb/**\* \
social-agents/king_of_the_hill/baseline/wandb/**\* \
social-agents/king_of_the_hill/ray_results/**\*