#!/usr/bin/env bash
# Gate ablations — kernel size and alive masking on representative gates.
#
#   E_KS: kernel ∈ {3, 5, 7}       × 4 gates × 2 seeds × 15k steps  (24 runs)
#   E_AM: alive_threshold ∈ {0, 0.1} × 4 gates × 2 seeds × 15k steps (16 runs)
#
# k=5 / threshold=0 runs are shared (tagged E1) — only 32 unique runs.
# Run from project root: bash scripts/run_gate_ablations.sh

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=15000
GATES=(XOR AND half_adder)

# ── Kernel size ablation ───────────────────────────────────────────────────────

total_ks=$(( ${#GATES[@]} * 3 * 2 ))
i=0

echo "══════════════════════════════════════════════════════════════"
echo "  Kernel size ablation  (${total_ks} runs)"
echo "══════════════════════════════════════════════════════════════"

for gate in "${GATES[@]}"; do
    for k in 3 5 7; do
        for seed in 0 1; do
            i=$(( i + 1 ))
            echo ""
            echo "  [KS $i/$total_ks]  gate=$gate  kernel=$k  seed=$seed"
            echo "────────────────────────────────────────────────────"
            uv run python scripts/train_gate.py \
                --gate "$gate" --kernel_size "$k" --seed "$seed" --steps "$STEPS"
        done
    done
done

# ── Alive masking ablation ─────────────────────────────────────────────────────
# k=5 / threshold=0 already covered above — only run threshold=0.1 here.

total_am=$(( ${#GATES[@]} * 2 ))
i=0

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Alive masking ablation  (${total_am} runs, threshold=0.1 only)"
echo "══════════════════════════════════════════════════════════════"

for gate in "${GATES[@]}"; do
    for seed in 0 1; do
        i=$(( i + 1 ))
        echo ""
        echo "  [AM $i/$total_am]  gate=$gate  alive_threshold=0.1  seed=$seed"
        echo "────────────────────────────────────────────────────"
        uv run python scripts/train_gate.py \
            --gate "$gate" --alive_threshold 0.1 --seed "$seed" --steps "$STEPS"
    done
done

echo ""
echo "Ablations complete."
echo "  Kernel size : runs/E_KS_*  (k=3,7)  and  runs/E1_*  (k=5)"
echo "  Alive mask  : runs/E_AM_*  (threshold=0.1)"
