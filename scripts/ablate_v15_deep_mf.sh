#!/usr/bin/env bash
# Deep pool on the deterministic machine: from the mean-fire + EMA best (step 5k), pool
# states are advanced 10-60 cycles every 20 steps, so training sees the faded tapes of
# long runs (the pool was ~6 cycles deep). 30k steps, then the 1000-cycle test.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261027_033352_ab4_meanfire_ema_s0/nca_keep_best5k.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 3e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --tape-hold 1 --steps 30001 --init-from $BASE --mean-fire 1 --ema 0.999"
uv run python scripts/train_tape.py --tag ab6_deep_mf $COMMON --deep-every 20 --deep-advance 10,60 \
    --pool-draw balanced --pool-evict balanced > runs/ab6_deep_mf.log 2>&1
d=$(ls -d runs/TAPE_*_ab6_deep_mf_s0 | tail -1)
for c in best last; do
    uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_${c}.pt --lengths 8,32,128 \
        --cycles 1000 --batch 64 --out runs/ab6_long_deep_mf_${c}.json > runs/ab6_long_deep_mf_${c}.log 2>&1
done
echo ab6 done
