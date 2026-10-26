#!/usr/bin/env bash
# Deterministic fire (every cell applies 0.5 * delta every step: the same mean update
# as fire rate 0.5, no randomness) from the hold run's step 20k; 30k steps, then the
# 1000-cycle test on its best and last checkpoints.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261025_210406_ab_hold_s0/nca_keep_20k.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 3e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --tape-hold 1 --steps 30001 --init-from $BASE"
uv run python scripts/train_tape.py --tag ab3_meanfire $COMMON --mean-fire 1 > runs/ab3_meanfire.log 2>&1
d=$(ls -d runs/TAPE_*_ab3_meanfire_s0 | tail -1)
for c in best last; do
    uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_${c}.pt --lengths 8,32,128 \
        --cycles 1000 --batch 64 --out runs/ab3_long_meanfire_${c}.json > runs/ab3_long_meanfire_${c}.log 2>&1
done
echo meanfire done
