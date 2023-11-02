# Run from parent directory of repository root (i.e /<this_dir>/social-agents)

if [ ! -d "social-agents" ]; then
    echo "Error: social-agents directory does not exist."
    exit 1
fi

# -x directories are skipped so as to keep the resulting ZIP as small as possible.
zip -r -q /tmp/social-agents.zip social-agents -x \
social-agents/venv/**\* \
social-agents/wandb/**\* \
social-agents/doc/**\* \
social-agents/.git/**\* \
social-agents/.pytest_cache/**\* \
social-agents/.idea/**\* \
social-agents/eval_games/**\*