#!/usr/bin/env bash
# E1 — Gate benchmark: all 8 functions × 5 seeds.
# Seed 0 already done — this runs seeds 1-4 (32 runs).
# Run from project root: bash scripts/run_e1.sh

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=15000
GATES=(AND OR XOR NAND NOR XNOR half_adder majority3)

total=$(( ${#GATES[@]} * 4 ))
i=0

for gate in "${GATES[@]}"; do
    for seed in 1 2; do
        i=$(( i + 1 ))
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  [$i/$total]  gate=$gate  seed=$seed"
        echo "════════════════════════════════════════════════════"
        uv run python scripts/train_gate.py \
            --gate  "$gate" \
            --seed  "$seed" \
            --steps "$STEPS"
    done
done

echo ""
echo "E1 complete. Results in runs/E1_*"
