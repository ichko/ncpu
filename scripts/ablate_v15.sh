#!/usr/bin/env bash
# One-change-at-a-time ablation from the v15 step-70k checkpoint (lr 3e-5 run).
# Three 30k-step runs in parallel, then the 1000-cycle test on each and on the start.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261023_142335_v15lr3e5_s0/nca_best.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 3e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --steps 30001 --init-from $BASE"
uv run python scripts/train_tape.py --tag ab_control $COMMON > runs/ab_control.log 2>&1 &
uv run python scripts/train_tape.py --tag ab_deep $COMMON --deep-every 20 --deep-advance 10,60 \
    --pool-draw balanced --pool-evict balanced > runs/ab_deep.log 2>&1 &
uv run python scripts/train_tape.py --tag ab_hold $COMMON --tape-hold 1 > runs/ab_hold.log 2>&1 &
wait
LONG="--lengths 8,32,128 --cycles 1000 --batch 64"
uv run python scripts/long_eval_tape.py --run "$(dirname $BASE)" --ckpt nca_best.pt $LONG \
    --out runs/ab_long_start.json > runs/ab_long_start.log 2>&1
for t in control deep hold; do
    d=$(ls -d runs/TAPE_*_ab_${t}_s0 | tail -1)
    uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_last.pt $LONG \
        --out runs/ab_long_${t}.json > runs/ab_long_${t}.log 2>&1
done
echo ablation done
