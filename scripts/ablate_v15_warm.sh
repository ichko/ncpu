#!/usr/bin/env bash
# Realistic register states: half of the fresh training samples get registers from
# running the program in software for 0-200 cycles (long runs collapse towards zeros;
# uniform random registers never show that). From the ab5 best (mean fire + EMA), 30k
# steps; then the 1000-cycle test on short programs (the honest ones: long programs end
# all-zero) for the new run and, as the baseline, its starting checkpoint.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261027_044159_ab5_mf_ema_lr1e5_s0/nca_keep_best12k.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 3e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --tape-hold 1 --steps 30001 --init-from $BASE --mean-fire 1 --ema 0.999"
uv run python scripts/train_tape.py --tag ab7_warm $COMMON --warm-regs 0.5 > runs/ab7_warm.log 2>&1
d=$(ls -d runs/TAPE_*_ab7_warm_s0 | tail -1)
LONG="--lengths 4,8,12,32 --cycles 1000 --batch 64"
for c in best last; do
    uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_${c}.pt $LONG \
        --out runs/ab7_long_warm_${c}.json > runs/ab7_long_warm_${c}.log 2>&1
done
uv run python scripts/long_eval_tape.py --run runs/TAPE_20261027_044159_ab5_mf_ema_lr1e5_s0 --ckpt nca_keep_best12k.pt $LONG \
    --out runs/ab7_long_base.json > runs/ab7_long_base.log 2>&1
echo ab7 done
