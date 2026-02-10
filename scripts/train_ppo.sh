#!/bin/bash

source .venv/bin/activate
python train_ppo_tune.py --n_iterations 600 --random_init --reduce_obs_err_dim || echo "Training script failed" \