#!/usr/bin/env bash
# Gate ablations — kernel size, alive masking, zero initialization.
#
#   E_KS: kernel ∈ {3, 5, 7}        × 3 gates × 2 seeds × 15k steps  (DONE)
#   E_AM: alive_threshold ∈ {0, 0.1} × 3 gates × 2 seeds × 15k steps  (DONE)
#   E_ZI: zero_init ∈ {False, True}  × 3 gates × 2 seeds × 15k steps  (ACTIVE)
#
# Run from project root: bash scripts/other/run_gate_ablations.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS=15000
GATES=(XOR AND half_adder)

# ── Kernel size ablation (DONE) ────────────────────────────────────────────────

# total_ks=$(( ${#GATES[@]} * 3 * 2 ))
# i=0
#
# echo "══════════════════════════════════════════════════════════════"
# echo "  Kernel size ablation  (${total_ks} runs)"
# echo "══════════════════════════════════════════════════════════════"
#
# for gate in "${GATES[@]}"; do
#     for k in 3 5 7; do
#         for seed in 0 1; do
#             i=$(( i + 1 ))
#             echo ""
#             echo "  [KS $i/$total_ks]  gate=$gate  kernel=$k  seed=$seed"
#             echo "────────────────────────────────────────────────────"
#             uv run python scripts/train/train_gate.py \
#                 --gate "$gate" --kernel_size "$k" --seed "$seed" --steps "$STEPS"
#         done
#     done
# done

# ── Alive masking ablation (DONE) ─────────────────────────────────────────────
# k=5 / threshold=0 already covered above — only run threshold=0.1 here.

# total_am=$(( ${#GATES[@]} * 2 ))
# i=0
#
# echo ""
# echo "══════════════════════════════════════════════════════════════"
# echo "  Alive masking ablation  (${total_am} runs, threshold=0.1 only)"
# echo "══════════════════════════════════════════════════════════════"
#
# for gate in "${GATES[@]}"; do
#     for seed in 0 1; do
#         i=$(( i + 1 ))
#         echo ""
#         echo "  [AM $i/$total_am]  gate=$gate  alive_threshold=0.1  seed=$seed"
#         echo "────────────────────────────────────────────────────"
#         uv run python scripts/train/train_gate.py \
#             --gate "$gate" --alive_threshold 0.1 --seed "$seed" --steps "$STEPS"
#     done
# done

# ── Zero initialization ablation (ACTIVE) ─────────────────────────────────────
# zero_init=False is the baseline (E1); only run zero_init=True here.

total_zi=$(( ${#GATES[@]} * 2 ))
i=0

echo "══════════════════════════════════════════════════════════════"
echo "  Zero init ablation  (${total_zi} runs, zero_init=True only)"
echo "══════════════════════════════════════════════════════════════"

for gate in "${GATES[@]}"; do
    for seed in 0 1; do
        i=$(( i + 1 ))
        echo ""
        echo "  [ZI $i/$total_zi]  gate=$gate  zero_init=True  seed=$seed"
        echo "────────────────────────────────────────────────────"
        uv run python scripts/train/train_gate.py \
            --gate "$gate" --zero_init --seed "$seed" --steps "$STEPS"
    done
done

echo ""
echo "Ablations complete."
echo "  Zero init : runs/E_ZI_*  (zero_init=True)  vs  runs/E1_*  (baseline)"
