#!/usr/bin/env bash
# Sweeps bit-adder trainings across several bit widths and seeds so you can
# study how the seed (and bit width) affects training. Wraps the existing
# train_adder.py (which already supports --bits/--seed/--steps/--layout).
# Each run lands in its own seeded run dir, e.g. runs/E2_adder4_s1_<ts>/.
#
# Usage:
#   scripts/other/train_adder_seed_sweep.sh
#
# Env overrides:
#   BITS="4 8"        scripts/other/train_adder_seed_sweep.sh  # widths to train
#   SEEDS="1 2 3"     scripts/other/train_adder_seed_sweep.sh  # custom seeds
#   STEPS=50000       scripts/other/train_adder_seed_sweep.sh
#   LAYOUT=cols2      scripts/other/train_adder_seed_sweep.sh  # cols1/cols2
#   DEVICE=cpu        scripts/other/train_adder_seed_sweep.sh
#   PARALLEL=2        scripts/other/train_adder_seed_sweep.sh  # run 2 at once
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

BITS=(${BITS:-4 8})
SEEDS=(${SEEDS:-1 2 3})
STEPS="${STEPS:-50000}"
LAYOUT="${LAYOUT:-}"
DEVICE="${DEVICE:-cuda}"
PARALLEL="${PARALLEL:-1}"

run_config() {
    local bits="$1" seed="$2"
    local layout_args=()
    [[ -z "$LAYOUT" ]] || layout_args+=(--layout "$LAYOUT")
    uv run python scripts/train/train_adder.py \
        --bits  "$bits" \
        --seed  "$seed" \
        --steps "$STEPS" \
        --device "$DEVICE" \
        "${layout_args[@]}"
}
export -f run_config
export STEPS LAYOUT DEVICE

printf 'Adder seed sweep: steps=%s device=%s layout=%s bits=[%s] seeds=[%s] (%d bits x %d seeds)\n' \
    "$STEPS" "$DEVICE" "${LAYOUT:-any}" "${BITS[*]}" "${SEEDS[*]}" "${#BITS[@]}" "${#SEEDS[@]}"
printf 'Run dirs include the seed (e.g. runs/E2_adder4_s1_<ts>/).\n'

# "bits seed" lines
lines=()
for bits in "${BITS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        lines+=("$bits $seed")
    done
done

if [ "$PARALLEL" -gt 1 ]; then
    printf '%s\n' "${lines[@]}" | xargs -P "$PARALLEL" -n 2 bash -c 'run_config "$1" "$2"' _
else
    for line in "${lines[@]}"; do
        read -r bits seed <<<"$line"
        printf '\n===== bits=%s seed=%s =====\n' "$bits" "$seed"
        run_config "$bits" "$seed"
    done
fi

# Print the run dirs matching each bits x seed for quick reference.
printf '\n'
printf '%-5s %-6s %-s\n' "bits" "seed" "dir"
for line in "${lines[@]}"; do
    read -r bits seed <<<"$line"
    exp="E2"; [[ "$bits" == "8" ]] && exp="E3"
    dir="$(ls -td runs/${exp}_adder${bits}*_s${seed}_* 2>/dev/null | head -1 || true)"
    printf '%-5s %-6s %-s\n' "$bits" "$seed" "${dir:-N/A}"
done
