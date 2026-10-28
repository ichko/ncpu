#!/usr/bin/env bash
# Bigger memory for Game of Life: fine-tune the mean-fire + EMA best (4 regs x 4 bits)
# stage 5: 24..44 registers (Game of Life needs 43), 80 exec steps (44 regs: torus column
# distance up to 176 px; 48 steps handled 96 px). From the mem24 last checkpoint (40k).
# Batches shrink with width so each step holds about the same pixels.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261028_062316_mem24_s0/nca_keep_40k.pt
uv run python scripts/train_tape.py --tag mem44 --channels 24 --hidden 384 --load-weight 3 \
    --tape-weight 3 --lr 3e-5 --pool-gate 2 --visible-noise 0.05 --pool-fp16 1 --tape-hold 1 \
    --mean-fire 1 --exec-steps 80 --ema 0.999 --init-from $BASE --steps 40001 \
    --word-bits 6 --regs-set 24,32,40,44 --batch-ref-regs 4 --pool-size 512 \
    --eval-regs 44 --eval-batch 64 --eval-cycles 1,2,4,8,16,32,64,128 > runs/mem44.log 2>&1
echo mem44 done
