#!/usr/bin/env bash
# Continue the mean-fire run's best (step 17.5k) with EMA 0.999 and lr 1e-5, then
# the 1000-cycle test on best and last; then the decay study of the starting model
# (deterministic, so 128 programs x 1 seed).
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261027_001021_ab3_meanfire_s0/nca_keep_best17k.pt
COMMON="--channels 24 --hidden 384 --load-weight 3 --tape-weight 3 --lr 1e-5 --pool-gate 2
        --visible-noise 0.05 --pool-fp16 1 --eval-batch 128 --eval-cycles 1,2,4,8,16,32,64,128
        --tape-hold 1 --steps 30001 --init-from $BASE"
uv run python scripts/train_tape.py --tag ab5_mf_ema_lr1e5 $COMMON --mean-fire 1 --ema 0.999 > runs/ab5_mf_ema_lr1e5.log 2>&1
d=$(ls -d runs/TAPE_*_ab5_mf_ema_lr1e5_s0 | tail -1)
for c in best last; do
    uv run python scripts/long_eval_tape.py --run "$d" --ckpt nca_${c}.pt --lengths 8,32,128 \
        --cycles 1000 --batch 64 --out runs/ab5_long_mf_ema_lr1e5_${c}.json > runs/ab5_long_mf_ema_lr1e5_${c}.log 2>&1
done
uv run python scripts/decay_tape.py --run runs/TAPE_20261027_001021_ab3_meanfire_s0 --ckpt nca_keep_best17k.pt --n 32 --programs 128 --seeds 1 \
    --cycles 1000 --out v15_decay_meanfire > runs/v15_decay_meanfire.log 2>&1
echo ab5 done
