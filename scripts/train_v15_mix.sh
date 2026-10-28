#!/usr/bin/env bash
# One run that varies everything per step: program length 3..24, registers 4..44 (Game of
# Life needs 43), cycles per rollout 1..6 (batch scaled so the graph stays the size of 3).
# Seamless memory (wrap gap = register gap); 48 exec steps reach 144 px, and the farthest
# column of a seamless 44-register row is 132 px away. From the mem24 40k checkpoint.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261028_062316_mem24_s0/nca_keep_40k.pt
uv run python scripts/train_tape.py --tag mix --channels 24 --hidden 384 --load-weight 3 \
    --tape-weight 3 --lr 3e-5 --pool-gate 2 --visible-noise 0.05 --pool-fp16 1 --tape-hold 1 --seamless 1 \
    --mean-fire 1 --exec-steps 48 --ema 0.999 --init-from $BASE --steps 60001 \
    --word-bits 6 --regs 4,44 --instrs 3,24 --cycles-range 1,6 --batch-ref-regs 4 --pool-size 512 \
    --eval-regs 32 --eval-batch 64 --eval-cycles 1,2,4,8,16,32,64,128 > runs/mix.log 2>&1
echo mix done
