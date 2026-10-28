#!/usr/bin/env bash
# Bigger memory for Game of Life: fine-tune the mean-fire + EMA best (4 regs x 4 bits)
# stage 4: 12..24 registers, 48 exec steps (24 regs: torus column distance up to 96 px;
# 32 steps x 3 px solved 16 regs). From the mem16e best.
# Batches shrink with width so each step holds about the same pixels.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261028_024107_mem16e_s0/nca_keep_best.pt
uv run python scripts/train_tape.py --tag mem24 --channels 24 --hidden 384 --load-weight 3 \
    --tape-weight 3 --lr 3e-5 --pool-gate 2 --visible-noise 0.05 --pool-fp16 1 --tape-hold 1 \
    --mean-fire 1 --exec-steps 48 --ema 0.999 --init-from $BASE --steps 40001 \
    --word-bits 6 --regs-set 12,16,20,24 --batch-ref-regs 4 --pool-size 512 \
    --eval-regs 24 --eval-batch 64 --eval-cycles 1,2,4,8,16,32,64,128 > runs/mem24.log 2>&1
echo mem24 done
