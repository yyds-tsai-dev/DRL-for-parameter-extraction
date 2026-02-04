#!/bin/bash

cat /etc/os-release
chsh -s /bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
source .venv/bin/activate
conda deactivate
uv run train_ppo_tune.py --reward_norm --n_iterations 200 || echo "Initializing script failed" \