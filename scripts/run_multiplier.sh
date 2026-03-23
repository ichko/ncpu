#!/usr/bin/env bash
set -euo pipefail

STEPS=${STEPS:-50000}
DEVICE=${DEVICE:-cuda}

for seed in 0 1; do
    echo "=== 4-bit multiplier  seed=${seed} ==="
    uv run python scripts/train_multiplier.py \
        --seed "$seed" \
        --steps "$STEPS" \
        --device "$DEVICE"
done

echo "=== All multiplier runs complete ==="
