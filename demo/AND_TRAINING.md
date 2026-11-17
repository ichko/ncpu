# Native AND sliding-window demo

Train a two-stage curriculum:

```sh
uv run python scripts/train/train_and.py --device cuda --seed 1 \
  --steps 10000 --stage1-steps 2000 --eval-every 100 --save-rollout-npz \
  --rollout-steps-min 64 --rollout-steps-max 96 --hop-min 2 --hop-max 24
```

New runs sample an inclusive 64–96 total NCA update budget for every rollout
(`--rollout-steps-min 64 --rollout-steps-max 96`). Stage 1 applies the whole
budget to a stationary 32×32 AND tile. Each batch contains all
four truth-table inputs equally. Stage 2 embeds that tile in a padded board and
moves a 32×32 window over an 80×80 board in a bounded random walk (default
displacement ±24 pixels, six windows). Each move samples a 2–24-pixel hop,
moving along one axis in either direction. Both horizontal and vertical
movement are included in every rollout. Only valid directions are sampled,
so large hops are not silently clamped into smaller moves.
The sampled update budget is split evenly across those
six windows, with any remainder assigned to the earliest windows. Four extra
settling updates follow the rollout and are not included in that budget.
Legacy checkpoints retain their saved fixed 48-update curriculum on resume.
Writes are persistent and
gradients flow through overlapping windows. This curriculum teaches local motion;
large-board, long-sweep generalization still needs evaluation.

Channel 3 carries a read-only, gate-local diagonal position ramp by default:
`(x + y) / 31 - 1`, increasing from −1 at the upper-left to +1 at the
lower-right. It is initialized on the board and moves with the gate, not
regenerated inside each window. Both stages and browser inference use this
same encoding. A diagonal scalar is a positional cue, not a unique 2D
coordinate. Disable it for an ablation with `--no-position-gradient`.

Each group of four AND examples includes one additional blank-input example:
no input dots, selector zero, and working channel zero (display midpoint 0.5).
The position ramp remains present, matching empty browser tiles. These are
not a new gate type. An additional whole-frame blank MSE (weight 1.0,
`--blank-weight`) supervises both pre-settling and post-settling blank outputs.
Evaluations report blank MSE and maximum absolute drift separately from AND
truth-table accuracy. `--batch-size 4` therefore processes five examples when
blank training is enabled; disable with `--no-blank-examples`.

Context padding defaults to 3 pixels (`--context-padding 3`): each update reads
a 38×38 patch from the persistent board, runs the local rule, and commits only
the central 32×32 window. The halo is refreshed from board state every update.
The browser uses the same operation, with a configurable Context padding slider.
This reduces artificial crop-edge discontinuities; true board edges still use
zero padding. Legacy checkpoints resume with their saved padding, or zero if
no context-padding setting was recorded.

The browser's Scanning head selector switches between uniform updates and a
Gaussian-weighted delta. Gaussian mode uses `next = clip(state + mask * delta)`;
it never multiplies the existing state by the mask. The Gaussian peaks at 1
on the four central pixels of the even-sized window. Width σ defaults to 8
pixels and is adjustable. To train with that head, pass
`--update-head gaussian --mask-sigma 8`; uniform remains the training default.
Switching an already-trained model's head is an inference experiment, and
matching the training head provides the directly trained behavior.

The compound loss is output MSE + 0.2 input-preservation MSE + 0.05 background
MSE + 0.1 settling-stability MSE + 0.0001 hidden-state energy. Output supervision
is applied both before and after four additional settling updates. Terms are
normalized by their supervised regions, including batch dimension. Every weight
is configurable with `--*-weight` flags.

Periodic `trainer.evaluate()` uses a fixed truth-table batch with repeatable
stochastic rollouts, testing stationary and displaced operation. Run outputs
include `config.json`, `log.jsonl`, `curves.png`, checkpoints, and optional
`rollout_latest.npz`. Resume with `--resume runs/and_s1_<timestamp>` or
`--resume last`; the checkpoint restores optimizer, curriculum and RNG state.

Launch the checkpoint:

```sh
uv run python web_demo.py --checkpoint runs/and_s1_<timestamp> \
  --canvas 512 --hop 2 --inner-steps 8
```

Open the URL printed by the launcher. The model is exported once; JavaScript
runs inference locally in WebGL2. Python serves only static files. The board
holds persistent state, windows overlap in serpentine order, and updates are
spread across animation frames. `Updates / frame` controls GPU workload and
`Steps / window` controls the dwell time before the next hop. Play/pause with
Space; step with S. Stage-1 checkpoints can be viewed with `--hop 32
--inner-steps 48`.

The browser engine supports the current 16-channel, one-hidden-layer,
zero-padded architecture. It uses four RGBA state textures and two small
ping-pong window buffers, so update cost depends on window size rather than
board size. Actual frame rate depends on the GPU and model dimensions.
