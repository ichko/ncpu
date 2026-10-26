#!/usr/bin/env bash
# Repair training (damaged bands, clean targets) vs no stochastic fire (fire rate 1),
# both from the hold run's step 20k. Two 30k-step runs in parallel, then the
# 1000-cycle test on each run's best and last checkpoints.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261025_210406_ab_hold_s0/nca_keep_20k.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 3e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --tape-hold 1 --steps 30001 --init-from $BASE"
uv run python scripts/train_tape.py --tag ab3_repair $COMMON --repair 0.5 > runs/ab3_repair.log 2>&1 &
uv run python scripts/train_tape.py --tag ab3_fire1 $COMMON --fire-rate 1.0 > runs/ab3_fire1.log 2>&1 &
wait
LONG="--lengths 8,32,128 --cycles 1000 --batch 64"
for t in repair fire1; do
    d=$(ls -d runs/TAPE_*_ab3_${t}_s0 | tail -1)
    for c in best last; do
        uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_${c}.pt $LONG \
            --out runs/ab3_long_${t}_${c}.json > runs/ab3_long_${t}_${c}.log 2>&1
    done
done
echo ablation done
