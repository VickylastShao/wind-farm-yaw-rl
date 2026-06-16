#!/bin/bash
# -*- coding: utf-8 -*-
# Task C13: Pure marginal reward control group (no SLSQP headroom)
#
# Trains Config-E PPO WITHOUT the SLSQP-regret reward, falling back to
# the baseline-aligned marginal reward.  All other Config-E features
# (j=3, deficit norm, positions, focused sampling, cosine LR, KL
# early-stop, AdamW, gamma=0.995, +/-10 bounds) are retained.
#
# This ablation isolates the contribution of the regret-reward signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export N_SEEDS=3
export N_ENVS=128
export TOTAL_STEPS=60000000
export N_STEPS=256
export BATCH_SIZE=4096
export N_EPOCHS=10
export MAX_EPISODE_STEPS=200
export ACT_BOUND=10.0
export J=3

# Config-E features
export USE_POSITIONS=1
export LR_DECAY=1
export LR_END=3e-5
export WEIGHT_DECAY=1e-4
export TARGET_KL=0.015
export WIND_MIXTURE="0.3,0.3,0.4"
export GAMMA=0.995

# KEY: disable regret reward
export USE_REGRET=""

export OUT_TAG="marginal_reward"

echo "=== Pure Marginal Reward Ablation ==="
echo "  USE_REGRET = (empty → marginal reward)"
echo "  N_SEEDS    = $N_SEEDS"
echo "  Config     = Config-E features without SLSQP headroom"
echo ""

python train_3x3_nnx_jaxenv.py
