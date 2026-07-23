#!/bin/bash

set -euo pipefail

source .venv/bin/activate
python train_ppo.py "$@"
