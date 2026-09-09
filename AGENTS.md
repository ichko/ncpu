# Project Conventions

## Train scripts must support the same CLI args across all experiment scripts

Every `scripts/train/train_*.py` must accept these arguments:
- `--seed`, `--steps`, `--device`, `--resume`
- `--eval-every` (int, default 1000)
- `--save-rollout-npz` (flag, default False)

**Rule**: Before creating a new SLURM script that calls a train script, verify the train script accepts ALL flags the SLURM script passes. The `slurm_alu_seed_sweep.sh` previously failed because it passed `--eval-every` and `--save-rollout-npz` to `train_alu.py` which didn't define them.

## SLURM scripts
- Use per-task venvs with `$SLURM_ARRAY_TASK_ID` in the path to avoid races
- All scripts follow the pattern: `uv venv` → `source activate` → `uv sync --active` → `uv run python ...`
- Default seeds: `SEEDS="1 2 3 4 5"`, default steps vary by experiment

## Adding new train scripts
When adding a new `train_*.py`:
1. Add `--eval-every` and `--save-rollout-npz` args to the ArgumentParser
2. Implement eval loop using `trainer.evaluate()` with a fixed eval batch
3. Implement NPZ rollout saving with `np.savez_compressed()`
4. Add eval curves to the plotting section
