#!/usr/bin/env bash
# Runs a few gate trainings across several seeds so you can study how the
# random seed affects training. Each run's results are saved into their own
# run directory postfixed with the seed (e.g. ..._s1, ..._s2) via
# train_gate.py --seed.
#
# Usage:
#   scripts/other/train_seed_sweep.sh
#
# Env overrides:
#   GATES="AND XOR"  scripts/other/train_seed_sweep.sh   # subset of gates
#   NOISE=0.4 FIRE=0.5 scripts/other/train_seed_sweep.sh # fixed noise
#   (noise & fire both 0 -> robust/no-noise, see the training script)
#   SEEDS="1 2 3"     scripts/other/train_seed_sweep.sh   # custom seeds
#   N_INPUTS=4        scripts/other/train_seed_sweep.sh   # 4-input gates
#   STEPS=30000       scripts/other/train_seed_sweep.sh
#   PARALLEL=2        scripts/other/train_seed_sweep.sh   # run 2 at once
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

GATES=(${GATES:-AND OR XOR NAND NOR XNOR})
SEEDS=(${SEEDS:-1 2 3})
NOISE="${NOISE:-0.0}"
FIRE="${FIRE:-0.0}"
STEPS="${STEPS:-20000}"
KERNEL_SIZE="${KERNEL_SIZE:-5}"
ALIVE_THRESHOLD="${ALIVE_THRESHOLD:-0.0}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
N_INPUTS="${N_INPUTS:-2}"
PARALLEL="${PARALLEL:-1}"

run_config() {
    local gate="$1" seed="$2"
    uv run python scripts/train/train_gate.py \
        --gate "$gate" \
        --gaussian-noise "$NOISE" \
        --gaussian-noise-fire-rate "$FIRE" \
        --kernel_size "$KERNEL_SIZE" \
        --alive_threshold "$ALIVE_THRESHOLD" \
        --steps "$STEPS" \
        --eval-every "$EVAL_EVERY" \
        --n-inputs "$N_INPUTS" \
        --seed "$seed"
}
export -f run_config
export NOISE FIRE STEPS KERNEL_SIZE ALIVE_THRESHOLD EVAL_EVERY N_INPUTS

printf 'Seed sweep: noise=%s fire=%s steps=%s n_inputs=%s gates=[%s] seeds=[%s] (%d gates x %d seeds)\n' \
    "$NOISE" "$FIRE" "$STEPS" "$N_INPUTS" "${GATES[*]}" "${SEEDS[*]}" "${#GATES[@]}" "${#SEEDS[@]}"
printf 'Run dirs include seed postfix (e.g. ..._s1, ..._s2) and n_inputs (e.g. _ni4).\n'

# "gate seed" lines
lines=()
for gate in "${GATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        lines+=("$gate $seed")
    done
done

if [ "$PARALLEL" -gt 1 ]; then
    printf '%s\n' "${lines[@]}" | xargs -P "$PARALLEL" -n 2 bash -c 'run_config "$1" "$2"' _
else
    for line in "${lines[@]}"; do
        read -r gate seed <<<"$line"
        printf '\n===== gate=%s seed=%s =====\n' "$gate" "$seed"
        run_config "$gate" "$seed"
    done
fi

# Summarize final eval bits per seed for quick comparison.
printf '\n'
printf '%-8s %-6s %-s\n' "gate" "seed" "dir"
for line in "${lines[@]}"; do
    read -r gate seed <<<"$line"
    tag="robust"
    if [ "$(python3 -c "print(1 if $NOISE > 0 or $FIRE > 0 else 0)")" = "1" ]; then
        tag="n$(python3 -c "print(int($NOISE*100))")fr$(python3 -c "print(int($FIRE*100))")"
    fi
    dir="$(ls -td runs/E1_${gate}*_ni${N_INPUTS}_${tag}_s${seed}_* 2>/dev/null | head -1 || true)"
    printf '%-8s %-6s %-s\n' "$gate" "$seed" "${dir:-N/A}"
done
