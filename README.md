# NCPU — spatial computation in neural cellular automata

**Project page: [ncpu.pages.dev](https://ncpu.pages.dev/)**

**NCPU** asks one question. Can a Neural Cellular Automaton (NCA) learn exact
digital computation? An NCA is a single small update rule. Every cell on a grid
runs the same rule, and each cell sees only its neighbours. We draw the inputs
and outputs as circles on a 2D grid. We train the rule end-to-end on the pixel
target alone. The rule must move information across the grid and produce the
right answer. Nothing in the design knows about logic, wires, or bit order.

## What it does

- **Boolean gates.** One rule per function reaches 100% exact-match accuracy,
  across all seeds. The set is AND, OR, XOR, NAND, NOR, XNOR, a two-output
  half-adder, and three-input majority.
- **Multi-bit addition.** The same recipe also does 4-bit and 8-bit binary
  addition (16 input bits, 9-bit sum).
- **Carry propagation emerges.** On the longest carry chain, the output bits
  settle in order, least-significant first. The carry climbs the output column
  step by step, like a ripple-carry wave.
- **Length generalisation.** We train the rule only on sums of up to 3 bits. It
  then adds 8-bit operands it never saw, at 98% accuracy or better. A
  bottom-aligned encoding gives the rule a position-invariant carry.
- **Ablations.** Learning needs two things. The perception radius must be large
  enough to bridge neighbouring bits. The updates must be stochastic and start
  from a non-trivial state.
- **ALU (work in progress).** One rule for an 8-operation ALU learns the result
  byte well (about 99.7% per bit). It does not yet hold the one-bit status flags
  reliably (carry-out and branch-taken). This is an open problem, not a solved
  result.

## Setup

We use [uv](https://docs.astral.sh/uv/) to manage the project. uv installs the
package as an editable module named `ncpu`.

```bash
uv sync                      # install dependencies
uv run pytest tests/         # run the test suite
uv run nbstripout --install --attributes .gitattributes   # run once after clone
```

Format and lint:

```bash
uv run black src/
uv run autoflake --in-place --remove-all-unused-imports -r src/
uv run flake8 src/
```

## Usage

Standalone scripts in `scripts/` drive the training runs. Each script builds a
dataset (`ncpu.dataset`), an NCA (`ncpu.nca.NeuralCA`, or a variant such as
`ncpu.temporal_nca` or `ncpu.gated_nca`), and a training loop. It writes
checkpoints and configs under `runs/<name>/`. For example:

```bash
uv run python scripts/train_alu2_temporal.py --help
```

`ncpu.config` exports predefined task configurations (grid size, channels,
learning rate, and more) as constants. Examples are `TINY_AND_TRAINING_CONFIG`
and `BIG_4BIT_ADDER_TRAINING_CONFIG`. The analysis and figure scripts
(`analyze_*.py`, `inspect_*.py`) reproduce the contents of `results/`. The
`notebooks/` folder holds interactive exploration.

## Repository layout

| Path | What |
|------|------|
| `src/ncpu/` | package: `config`, `dataset`, `nca`, `trainer`, `runner`, … |
| `scripts/`  | training, analysis, and visualisation scripts |
| `results/`  | per-experiment metrics, figures, and analysis notes (E1–E5, ablations, ALU) |
| `runs/`     | training run directories with checkpoints and configs |
| `notebooks/`| interactive exploration |

See `CLAUDE.md` for a fuller description of the architecture and conventions.

## Roadmap

The goal is robust, reconfigurable logic on an NCA canvas. The steps:

1. Add read-only layers that mark where the inputs and outputs are. This helps
   the rule route information across distance.
2. Add a read-only selector, so one model can perform several gates.
3. Add a clock signal, and train the NCA to transfer in N-step units.
4. Add continuous Gaussian noise on each step, to mimic environmental damage
   such as radiation.
5. Train one model that withstands a range of noise levels and runs all gates.
6. Compose the gates into a larger, section-activated board that runs a simple
   digital circuit.

## References

1. Béna, Faldor, Goodman & Cully (2025), *A Path to Universal Neural Cellular
   Automata*, GECCO Companion — <https://arxiv.org/abs/2505.13058>
2. Miotti, Niklasson, Randazzo & Mordvintsev (2025), *Differentiable Logic
   Cellular Automata: From Game of Life to Pattern Generation*, ALIFE —
   <https://google-research.github.io/self-organising-systems/difflogic-ca/>
3. Mordvintsev, Randazzo, Niklasson & Levin (2020), *Growing Neural Cellular
   Automata*, Distill — <https://distill.pub/2020/growing-ca/>
4. Papadopoulos & Guichard (2025), *MaCE: General Mass Conserving Dynamics for
   Cellular Automata* — <https://arxiv.org/abs/2507.12306>
