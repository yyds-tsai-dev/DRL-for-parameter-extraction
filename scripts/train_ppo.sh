#!/bin/bash

source .venv/bin/activate
python train_ppo_tune.py --n_iterations 400 || echo "Training script failed" \