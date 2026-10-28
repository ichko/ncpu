#!/usr/bin/env bash
# Local programs only (a, b, d within 3 consecutive registers: what a local compiler emits),
# long narrow rings + short wide memories under one pixel cap: 8 regs x 256 instrs ...
# 128 regs x 8 instrs. Long rings failed after ~45 cycles (bands), the compiler needs
# ~100 registers. From mix2 best (32.5k). Half the fresh samples are windows of COMPILED programs
# (ncc random programs) on their real memory states: zeros, flags and small ints, which uniform
# random registers never show (mix2 computed 0 - 0 = -32 on an all-zero memory).
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261028_154832_mix2_s0/nca_best.pt
uv run python scripts/train_tape.py --tag ncc --channels 24 --hidden 384 --load-weight 3 \
    --tape-weight 3 --lr 3e-5 --pool-gate 2 --visible-noise 0.05 --pool-fp16 1 --tape-hold 1 --seamless 1 \
    --mean-fire 1 --exec-steps 48 --ema 0.999 --init-from $BASE --steps 100001 \
    --word-bits 6 --local-span 2 --regs 8,128 --regs-set 8,16,32,64,96,128 --instrs 8,256 \
    --instrs-set 8,16,32,64,128,256 --max-area 50000 --ncc-frac 0.5 --ncc-progs 400 \
    --cycles-range 1,6 --batch-ref-regs 4 --batch-ref-instrs 12 --pool-size 512 --pool-gb 8 \
    --eval-instrs 64 --eval-regs 32 --eval-batch 8 --eval-cycles 1,2,4,8,16,32,64,128,256 > runs/ncc.log 2>&1
echo ncc done
