#!/usr/bin/env bash
# Extrapolation sweep: train on 1..K bits, validate on K+1..8.
#
# Produces one run per (K, seed) pair.
# Results land in runs/extrap_train<K>_s<seed>_<timestamp>/
#
# Usage:
#   bash scripts/run_extrapolation_sweep.sh
#   STEPS=50000 bash scripts/run_extrapolation_sweep.sh   # faster smoke test

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-30000}
SEEDS=(0 1)
K_VALUES=(2 3 4 5 6)

total=$(( ${#K_VALUES[@]} * ${#SEEDS[@]} ))
i=0

echo "══════════════════════════════════════════════════════════════"
echo "  Extrapolation sweep  (${total} runs, ${STEPS} steps each)"
echo "══════════════════════════════════════════════════════════════"

for k in "${K_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        i=$(( i + 1 ))
        echo ""
        echo "  [$i/$total]  max_train_bits=$k  seed=$seed"
        echo "────────────────────────────────────────────────────"
        uv run python scripts/train_extrapolation.py \
            --max_train_bits "$k" \
            --seed "$seed" \
            --steps "$STEPS"
    done
done

echo ""
echo "Sweep complete. Analyse with:"
echo "  uv run python scripts/analyze_extrapolation.py"
