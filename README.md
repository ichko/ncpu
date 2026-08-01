# NCPU — spatial computation in neural cellular automata

**Project page: [ncpu.pages.dev](https://ncpu.pages.dev/)**

**NCPU** explores whether a Neural Cellular Automaton (NCA) — a single small
update rule applied identically to every cell of a grid, seeing only its
neighbours — can learn to perform exact digital computation. Inputs and outputs
are drawn as circles on a 2D grid, and the rule is trained end-to-end (only on
pixel targets) to move information across space and produce the right answer.
Nothing in the architecture knows about logic, wires, or bit order.

## What it does

- **Boolean gates.** One rule per function reaches 100% exact-match accuracy on
  AND, OR, XOR, NAND, NOR, XNOR, a two-output half-adder, and three-input
  majority (across all seeds).
- **Multi-bit addition.** The same recipe scales to 4-bit and 8-bit binary
  addition (16 inputs → 9-bit sum).
- **Emergent carry propagation.** On a maximal carry chain, the output bits
  settle in order — least-significant first — as a *ripple-carry wave* that
  climbs the output column over successive NCA steps. See the rollout figures
  in `docs/alife2026_lba/figs/`.
- **Length generalisation.** Trained only on additions of up to 3 bits, the
  rule adds 8-bit operands it never saw at ≥98% accuracy (bottom-aligned
  encoding → a position-invariant carry rule).
- **Ablations.** Learning needs a perception radius large enough to bridge
  neighbouring bits and stochastic updates from a non-trivial initial state.
- **ALU (work in progress).** A single rule for an 8-operation ALU learns the
  result byte well (~99.7% per-bit) but does not yet reliably hold the one-bit
  status flags (carry-out, branch-taken); this is an open problem, not a solved
  result.

## Paper

A 2-page Late Breaking Abstract for ALIFE 2026 lives in
[`docs/alife2026_lba/`](docs/alife2026_lba/) (`main.pdf` + LaTeX source).

## Setup

We use [uv](https://docs.astral.sh/uv/) for project management; the package is
installed as an editable module named `ncpu`.

```bash
uv sync                      # install dependencies
uv run pytest tests/         # run the test suite
uv run nbstripout --install --attributes .gitattributes   # run once after clone
```

Formatting / linting:

```bash
uv run black src/
uv run autoflake --in-place --remove-all-unused-imports -r src/
uv run flake8 src/
```

## Usage

Training runs are driven by standalone scripts in `scripts/`. Each builds a
dataset (`ncpu.dataset`), an NCA (`ncpu.nca.NeuralCA`, or a variant such as
`ncpu.temporal_nca`/`ncpu.gated_nca`), and a training loop; checkpoints and
configs are written under `runs/<name>/`. For example:

```bash
uv run python scripts/train_alu2_temporal.py --help
```

Predefined task configurations (grid size, channels, learning rate, …) are
exported from `ncpu.config` as constants such as `TINY_AND_TRAINING_CONFIG` and
`BIG_4BIT_ADDER_TRAINING_CONFIG`. Analysis and figure scripts (`analyze_*.py`,
`inspect_*.py`) reproduce the contents of `results/`. Interactive exploration
is in `notebooks/`.

## Repository layout

| Path | What |
|------|------|
| `src/ncpu/` | package: `config`, `dataset`, `nca`, `trainer`, `runner`, … |
| `scripts/`  | training, analysis, and visualisation scripts |
| `results/`  | per-experiment metrics, figures, and analysis notes (E1–E5, ablations, ALU) |
| `runs/`     | training run directories with checkpoints and configs |
| `docs/`     | write-ups, paper assets, and the ALIFE 2026 abstract |
| `notebooks/`| interactive exploration |

See `CLAUDE.md` for a fuller description of the architecture and conventions.

## Roadmap

Toward robust, reconfigurable logic on an NCA "canvas":

1. Read-only layers that mark where inputs and outputs are, so the rule learns
   to route information across distance.
2. A read-only selector that lets one model perform several gates.
3. A clock signal, training the NCA to transfer in N-step units.
4. Continuous Gaussian noise per step to mimic environmental damage (radiation).
5. A single model that withstands varying noise levels and runs all gates.
6. Composition into a larger, section-activated board that executes a simple
   digital circuit.

## References

1. Béna & Faldor (2025), *A Path to Universal Neural Cellular Automata* — <https://arxiv.org/abs/2505.13058>
2. Miotti et al. (2025), *Differentiable Logic Cellular Automata* — <https://google-research.github.io/self-organising-systems/difflogic-ca/>
3. Mordvintsev et al. (2020), *Growing Neural Cellular Automata* — <https://distill.pub/2020/growing-ca/>
4. *MaCE: General Mass-Conserving Dynamics for Cellular Automata* (2025) — <https://arxiv.org/abs/2507.12306>
