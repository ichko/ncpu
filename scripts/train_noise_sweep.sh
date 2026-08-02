#!/usr/bin/env bash
# Replicates the old per-noise-level training scripts
# (train_gates_noise_{20,40,60,80,100}.py) using the unified
# train_gates_noise_robust.py with fixed-noise configs.
#
<<<<<<< HEAD
# Each config trains a single gate only (--gate is always passed), and the
# whole noise sweep is run for every gate in GATES.
#
# Old behaviour being reproduced:
#   - noise 0.2/0.4/0.6/0.8/1.0 -> 30k steps
#   - fire rates {0.2, 0.4, 0.6, 0.8, 1.0} per noise level
#   - kernel size 7, alive_threshold 0.1, eval every 1000 steps
#
# Usage:
#   scripts/train_noise_sweep.sh                # all gates, one at a time
#   GATES="AND XOR" scripts/train_noise_sweep.sh # a subset of gates
#   GATE=XOR scripts/train_noise_sweep.sh       # just one gate
=======
# Old behaviour being reproduced:
#   - noise 0.2/0.4/0.6/0.8 -> 20k steps, noise 1.0 -> 30k steps
#   - fire rates {0.2, 0.4, 0.6, 0.8, 1.0} per noise level
#   - kernel size 7, alive_threshold 0.1, eval every 1000 steps
#   - all four gates via the one-hot code (no --gate)
#
# Usage:
#   scripts/train_noise_sweep.sh                # sequential, as before
#   GATE=XOR scripts/train_noise_sweep.sh       # restrict to one gate
>>>>>>> 12c5021 (updating noise scripts)
#   PARALLEL=2 scripts/train_noise_sweep.sh     # run up to 2 configs at once
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

KERNEL_SIZE="${KERNEL_SIZE:-7}"
ALIVE_THRESHOLD="${ALIVE_THRESHOLD:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
GATE="${GATE:-}"
PARALLEL="${PARALLEL:-1}"

# noise_level:steps pairs from the original scripts
declare -A CONFIGS=(
    [0.2]=30000
    [0.4]=30000
    [0.6]=30000
    [0.8]=30000
    [1.0]=30000
)
FIRE_RATES=(0.2 0.4 0.6 0.8 1.0)

<<<<<<< HEAD
# Gates to sweep; GATE=XOR overrides to a single gate.
if [ -n "$GATE" ]; then
    GATES=("$GATE")
else
    GATES=(${GATES:-AND OR NOR XOR NAND})
fi

run_config() {
    local gate="$1" noise="$2" fire_rate="$3" steps="$4"
    uv run python scripts/train_gates_noise_robust.py \
        --gate "$gate" \
=======
run_config() {
    local noise="$1" fire_rate="$2" steps="$3"
    local gate_args=()
    [ -n "$GATE" ] && gate_args=(--gate "$GATE")
    uv run python scripts/train_gates_noise_robust.py \
        "${gate_args[@]}" \
>>>>>>> 12c5021 (updating noise scripts)
        --gaussian-noise "$noise" \
        --gaussian-noise-fire-rate "$fire_rate" \
        --kernel-size "$KERNEL_SIZE" \
        --alive-threshold "$ALIVE_THRESHOLD" \
        --steps "$steps" \
        --eval-every "$EVAL_EVERY"
}
export -f run_config
<<<<<<< HEAD
export KERNEL_SIZE ALIVE_THRESHOLD EVAL_EVERY

printf 'Noise-gate sweep: kernel=%s alive=%s eval_every=%s gates=[%s] (%d gates x %d noise configs)\n' \
    "$KERNEL_SIZE" "$ALIVE_THRESHOLD" "$EVAL_EVERY" "${GATES[*]}" "${#GATES[@]}" \
    "$((${#CONFIGS[@]} * ${#FIRE_RATES[@]}))"

for gate in "${GATES[@]}"; do
    printf '\n==================== gate=%s ====================\n' "$gate"

    # "noise fire_rate steps" lines
    lines=()
    for n in 0.2 0.4 0.6 0.8 1.0; do
        for fr in "${FIRE_RATES[@]}"; do
            lines+=("$n $fr ${CONFIGS[$n]}")
        done
    done

    if [ "$PARALLEL" -gt 1 ]; then
        printf '%s\n' "${lines[@]}" | xargs -P "$PARALLEL" -n 3 bash -c 'run_config "$1" "$2" "$3" "$4"' _ "$gate"
    else
        for line in "${lines[@]}"; do
            read -r n fr steps <<<"$line"
            printf '\n===== gate=%s noise=%s fire_rate=%s =====\n' "$gate" "$n" "$fr"
            run_config "$gate" "$n" "$fr" "$steps"
        done
    fi
done
=======
export KERNEL_SIZE ALIVE_THRESHOLD EVAL_EVERY GATE

# "noise fire_rate steps" lines
lines=()
for n in 0.2 0.4 0.6 0.8 1.0; do
    for fr in "${FIRE_RATES[@]}"; do
        lines+=("$n $fr ${CONFIGS[$n]}")
    done
done

printf 'Replicating old noise-gate sweep: kernel=%s alive=%s eval_every=%s gate=%s (%d configs)\n' \
    "$KERNEL_SIZE" "$ALIVE_THRESHOLD" "$EVAL_EVERY" "${GATE:-all}" "${#lines[@]}"

if [ "$PARALLEL" -gt 1 ]; then
    printf '%s\n' "${lines[@]}" | xargs -P "$PARALLEL" -n 3 bash -c 'run_config "$1" "$2" "$3"' _
else
    for line in "${lines[@]}"; do
        read -r n fr steps <<<"$line"
        printf '\n===== noise=%s fire_rate=%s =====\n' "$n" "$fr"
        run_config "$n" "$fr" "$steps"
    done
fi
>>>>>>> 12c5021 (updating noise scripts)
