#!/usr/bin/env bash
# Mean fire + EMA of the weights (decay 0.999): training is the same as the mean-fire
# run, but evals and checkpoints use the averaged weights, which should damp the
# checkpoint-to-checkpoint swings. 30k steps, then the 1000-cycle test on best and last.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261025_210406_ab_hold_s0/nca_keep_20k.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 3e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --tape-hold 1 --steps 30001 --init-from $BASE"
uv run python scripts/train_tape.py --tag ab4_meanfire_ema $COMMON --mean-fire 1 --ema 0.999 > runs/ab4_meanfire_ema.log 2>&1
d=$(ls -d runs/TAPE_*_ab4_meanfire_ema_s0 | tail -1)
for c in best last; do
    uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_${c}.pt --lengths 8,32,128 \
        --cycles 1000 --batch 64 --out runs/ab4_long_meanfire_ema_${c}.json > runs/ab4_long_meanfire_ema_${c}.log 2>&1
done
echo meanfire ema done
