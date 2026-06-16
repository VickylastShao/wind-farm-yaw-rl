#!/bin/bash
# Serial training of fatigue-aware DRL policies.
# Runs 3 configs sequentially (GPU cannot handle parallel JAX processes).

set -e
cd "$(dirname "$0")"
source /home/gpu/sz_workspace/JAX-WFCOYAW-RL/.venv/bin/activate

echo "=== Training rate_med (lambda_rate=5e-4) ==="
LAMBDA_RATE=5e-4 LAMBDA_MAG=0 OUT_TAG=rate_med N_SEEDS=3 TOTAL_STEPS=30000000 python train_3x3_nnx_jaxenv_penalty.py

echo "=== Training rate_high (lambda_rate=2e-3) ==="
LAMBDA_RATE=2e-3 LAMBDA_MAG=0 OUT_TAG=rate_high N_SEEDS=3 TOTAL_STEPS=30000000 python train_3x3_nnx_jaxenv_penalty.py

echo "=== Training rate_extreme (lambda_rate=1e-2) ==="
LAMBDA_RATE=1e-2 LAMBDA_MAG=0 OUT_TAG=rate_extreme N_SEEDS=3 TOTAL_STEPS=30000000 python train_3x3_nnx_jaxenv_penalty.py

echo "=== All training complete ==="
