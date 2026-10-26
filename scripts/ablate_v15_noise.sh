#!/usr/bin/env bash
# Tape hold vs tape hold + per-step denoising noise, from the hold run's step 20k.
# Two 30k-step runs in parallel, then the 1000-cycle test on both.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261025_210406_ab_hold_s0/nca_keep_20k.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 3e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --tape-hold 1 --steps 30001 --init-from $BASE"
uv run python scripts/train_tape.py --tag ab2_hold $COMMON > runs/ab2_hold.log 2>&1 &
uv run python scripts/train_tape.py --tag ab2_stepnoise $COMMON --step-noise 0.02 > runs/ab2_stepnoise.log 2>&1 &
wait
LONG="--lengths 8,32,128 --cycles 1000 --batch 64"
for t in hold stepnoise; do
    d=$(ls -d runs/TAPE_*_ab2_${t}_s0 | tail -1)
    uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_last.pt $LONG \
        --out runs/ab2_long_${t}.json > runs/ab2_long_${t}.log 2>&1
done
echo ablation done
