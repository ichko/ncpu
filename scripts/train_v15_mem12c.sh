#!/usr/bin/env bash
# Bigger memory for Game of Life: fine-tune the mean-fire + EMA best (4 regs x 4 bits)
# stage 2 of the width curriculum: from the 4..8-register 6-bit best to 6..12 registers (8 -> 16 learned slowly).
# Batches shrink with width so each step holds about the same pixels.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261027_143449_mem8c_s0/nca_keep_best37k.pt
uv run python scripts/train_tape.py --tag mem12c --channels 24 --hidden 384 --load-weight 3 \
    --tape-weight 3 --lr 3e-5 --pool-gate 2 --visible-noise 0.05 --pool-fp16 1 --tape-hold 1 \
    --mean-fire 1 --ema 0.999 --init-from $BASE --steps 40001 \
    --word-bits 6 --regs-set 6,8,10,12 --batch-ref-regs 4 --pool-size 512 \
    --eval-regs 12 --eval-batch 64 --eval-cycles 1,2,4,8,16,32,64,128 > runs/mem12c.log 2>&1
echo mem12c done
