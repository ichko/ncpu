#!/usr/bin/env bash
# E3 — Two-column adder experiments (remaining only).
#   8-bit adder, cols2, seed 1, 50k steps
#   (4-bit cols2 s0 and 8-bit cols2 s0 already completed)
# Run from project root: bash scripts/run_e3.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo ""
echo "════════════════════════════════════════════════════"
echo "  8-bit adder  layout=cols2  seed=1"
echo "════════════════════════════════════════════════════"
uv run python scripts/train_adder.py --bits 8 --layout cols2 --seed 1 --steps 50000

echo ""
echo "E3 complete. Results in runs/E3_adder8_cols2_*"
