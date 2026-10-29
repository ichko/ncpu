#!/usr/bin/env bash
# clock: the NCA kept its own time and it drifted -- band moves finish later every cycle (half-done step
# 4 -> 8-13 by cycle 30), regions desync, 35-40% of the band motion lands in execute, the stamp leaves
# early. Supervised clock: the last two channels are painted once at phase 0 and scored every step
# against cos/sin of the phase; the NCA keeps time itself, nothing writes the state after painting.
# From ncc 12.5k; no top-k, no deep advance.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261029_003444_ncc_s0/nca_keep_12k5.pt
uv run python scripts/train_tape.py --tag clock --channels 24 --hidden 384 --load-weight 3 \
    --tape-weight 3 --lr 3e-5 --pool-gate 2 --visible-noise 0.05 --pool-fp16 1 --tape-hold 1 --seamless 1 \
    --mean-fire 1 --exec-steps 48 --ema 0.999 --init-from $BASE --steps 100001 \
    --word-bits 6 --local-span 2 --regs 8,128 --regs-set 8,16,32,64,96,128 --instrs 8,256 \
    --instrs-set 8,16,32,64,128,256 --max-area 50000 --ncc-frac 0.85 --ncc-progs 400 \
    --cycles-range 1,6 --batch-ref-regs 4 --batch-ref-instrs 12 --pool-size 512 --pool-gb 8 \
    --clock 1 \
    --eval-instrs 64 --eval-regs 32 --eval-batch 16 --eval-cycles 1,2,4,8,16,32,64,128,256 > runs/clock.log 2>&1
echo clock done
