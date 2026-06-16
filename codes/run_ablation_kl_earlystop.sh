#!/bin/bash
# -*- coding: utf-8 -*-
# Task C14: KL early-stop ablation
#
# Trains two Config-D variants to quantify KL early-stopping's contribution:
#   - kl_on:  TARGET_KL=0.015 (default, early-stop active)
#   - kl_off: TARGET_KL=100.0 (effectively disabled)
#
# Both variants share Config-D features: j=3, deficit norm, positions,
# regret reward, focused sampling, cosine LR, AdamW, gamma=0.995,
# +/-5 bounds (Config-D uses ±5, same as Config-A/B/C).
#
# Run both back-to-back on the same GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Shared config
export N_SEEDS=3
export N_ENVS=128
export TOTAL_STEPS=60000000
export N_STEPS=256
export BATCH_SIZE=4096
export N_EPOCHS=10
export MAX_EPISODE_STEPS=200
export ACT_BOUND=5.0
export J=3
export USE_POSITIONS=1
export USE_REGRET=1
export LR_DECAY=1
export LR_END=3e-5
export WEIGHT_DECAY=1e-4
export WIND_MIXTURE="0.3,0.3,0.4"
export GAMMA=0.995

echo "========================================="
echo "=== KL Early-Stop Ablation: KL ON ==="
echo "========================================="
export TARGET_KL=0.015
export OUT_TAG="kl_on"
python train_3x3_nnx_jaxenv.py

echo ""
echo "========================================="
echo "=== KL Early-Stop Ablation: KL OFF ==="
echo "========================================="
export TARGET_KL=100.0
export OUT_TAG="kl_off"
python train_3x3_nnx_jaxenv.py

echo ""
echo "=== KL ablation complete ==="
echo "  Compare: checkpoints_3x3_nnx_jaxenv/metrics_seed*_kl_on.json"
echo "           checkpoints_3x3_nnx_jaxenv/metrics_seed*_kl_off.json"
