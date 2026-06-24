#!/bin/bash

cat /etc/os-release
chsh -s /bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
source .venv/bin/activate
conda deactivate
uv run train_ppo.py --env eehemt --n_iterations 100 || echo "Initializing script failed" \
