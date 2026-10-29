#!/usr/bin/env bash
# desync: regions of the picture drift apart in closed loop (band moves finish at different steps per
# region, spread up to 14 steps by cycle 30) and training never showed a desynced state. Half the samples
# start with a random horizontal strip 1-4 steps ahead (all channels from the state run on); targets
# unchanged, so the NCA must pull the strip back in step. Time-free (lag) did nothing: the slip is
# sub-cycle. From ncc 12.5k.
set -u
cd "$(dirname "$0")/.."
BASE=runs/TAPE_20261029_003444_ncc_s0/nca_keep_12k5.pt
uv run python scripts/train_tape.py --tag desync --channels 24 --hidden 384 --load-weight 3 \
    --tape-weight 3 --lr 3e-5 --pool-gate 2 --visible-noise 0.05 --pool-fp16 1 --tape-hold 1 --seamless 1 \
    --mean-fire 1 --exec-steps 48 --ema 0.999 --init-from $BASE --steps 100001 \
    --word-bits 6 --local-span 2 --regs 8,128 --regs-set 8,16,32,64,96,128 --instrs 8,256 \
    --instrs-set 8,16,32,64,128,256 --max-area 50000 --ncc-frac 0.85 --ncc-progs 400 \
    --cycles-range 1,6 --batch-ref-regs 4 --batch-ref-instrs 12 --pool-size 512 --pool-gb 8 \
    --desync 0.5 --desync-k 4 \
    --eval-instrs 64 --eval-regs 32 --eval-batch 16 --eval-cycles 1,2,4,8,16,32,64,128,256 > runs/desync.log 2>&1
echo desync done
