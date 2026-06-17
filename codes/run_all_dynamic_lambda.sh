#!/bin/bash
set -e
cd /home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes
export USE_POSITIONS=1 USE_DEFICIT=1 J=3 USE_REGRET=1 WIND_MIXTURE="0.3,0.3,0.4"
export N_SEEDS=5 TOTAL_STEPS=60000000

for lr in 0 0.0001 0.0005 0.002 0.01; do
    tag="dyn_lambda$(echo $lr | sed 's/0\.0*//' | sed 's/\.//')_60M"
    [ "$lr" = "0" ] && tag="dyn_lambda0_60M"
    echo "=== Training λrate=$lr, tag=$tag, 5 seeds × 60M steps ==="
    export LAMBDA_RATE=$lr OUT_TAG=$tag
    python -u train_3x3_nnx_jaxenv_dynamic.py > "train_${tag}.log" 2>&1
    echo "Done: λrate=$lr ($(date))"
done
echo "ALL DONE ($(date))"
