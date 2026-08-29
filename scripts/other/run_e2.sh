#!/usr/bin/env bash
# E2 — 4-bit adder, 3 seeds, 50k steps.
# Run from project root: bash scripts/other/run_e2.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS=50000

for seed in 0 1 2; do
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  4-bit adder  seed=$seed"
    echo "════════════════════════════════════════════════════"
    uv run python scripts/train/train_adder.py --bits 4 --seed "$seed" --steps "$STEPS"
done

echo ""
echo "E2 complete. Results in runs/E2_adder4_*"
