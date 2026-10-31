#!/usr/bin/env python3
"""v16 probe -- a QUEUE tape with no clock: a band moves only into an empty slot.

The picture is a torus of S slots (1 gap row + 2 band rows each), W px wide. A slot
holds a band (channel 0 = random +-1 bits per column) or is empty (0). The discrete
machine is rule 184: a band moves down one slot when the slot below is empty. No
step count says when; each band waits on its neighbour.

Training is timing-free. Every slot's own history is fixed by the rule whatever the
timing (bands never overtake, a slot always empties between two bands), so every
slot keeps its own pointer into that history:

    each frame, slot k matches either its current value or its next one; the closer
    one is scored, and reaching the next one advances the pointer (never back).

So regions may run at different speeds. The only demand on time is a minimum speed:
by step t each slot must have reached where the synchronous machine is after
t // --tau-max moves. Only channel 0 is scored; hidden channels never are.

Eval runs long and decodes the picture: a run is intact when the bands, read round
the ring, are the painted ones in the painted order, and it reports moves made.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.nca import NeuralCA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=200_000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="queue")
ap.add_argument("--slots", default="6,24", help="ring length: min,max (drawn per step)")
ap.add_argument("--width", type=int, default=8)
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--fire-rate", type=float, default=0.5)
ap.add_argument("--rollout", type=int, default=96, help="steps per optimiser step (one graph)")
ap.add_argument("--tau-max", type=int, default=16, help="slowest allowed move, steps")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--grad-clip", type=float, default=1.0)
ap.add_argument("--pool-size", type=int, default=2048, help="per ring length")
ap.add_argument("--pool-frac", type=float, default=0.5)
ap.add_argument("--boundary-noise", type=float, default=0.02)
ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--eval-every", type=int, default=2500)
ap.add_argument("--eval-batch", type=int, default=32)
ap.add_argument("--eval-slots", type=int, default=16)
ap.add_argument("--eval-steps", default="500,2000,8000")
ap.add_argument("--ckpt-every", type=int, default=10000)
args = ap.parse_args()

DEV = args.device
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
S_RANGE = tuple(int(v) for v in args.slots.split(","))
EVAL_STEPS = [int(v) for v in args.eval_steps.split(",")]
W, P, C = args.width, 3, args.channels


# ── the discrete machine ────────────────────────────────────────────────────
# A config is a list of S entries: None (empty) or a band (tuple of W bits).

def random_config(S, rng):
    n = rng.randint(1, S - 1)                         # at least one hole
    occ = [None] * S
    for i in rng.sample(range(S), n):
        occ[i] = tuple(rng.choice((-1.0, 1.0)) for _ in range(W))
    return occ


def step184(occ):
    S = len(occ)
    new = list(occ)
    for i in range(S):
        if occ[i] is not None and occ[(i + 1) % S] is None:
            new[i], new[(i + 1) % S] = None, occ[i]
    return new


def slot_pixels(v):
    """3 x W pixels of one slot: gap row, then the band twice (or zeros)."""
    a = np.zeros((P, W), np.float32)
    if v is not None:
        a[1:] = np.asarray(v, np.float32)[None]
    return a


def paint(occ):
    return np.concatenate([slot_pixels(v) for v in occ], 0)       # (S*P, W)


def histories(occ, T):
    """Per slot: its values in order (consecutive repeats merged), for every synchronous
    time m <= T the index it has reached by then, and for every band value the index
    in the NEXT slot's history where that band arrives (-1 for empty values): a band
    may only leave once it has arrived below, so nothing can just vanish."""
    S = len(occ)
    hist = [[slot_pixels(occ[k])] for k in range(S)]
    dep = [[-1] for _ in range(S)]
    reach = np.zeros((T + 1, S), np.int64)
    cur = occ
    for m in range(1, T + 1):
        nxt = step184(cur)
        for k in range(S):
            if nxt[k] != cur[k]:
                hist[k].append(slot_pixels(nxt[k]))
                dep[k].append(-1)
            reach[m, k] = len(hist[k]) - 1
        for k in range(S):
            if cur[k] is not None and nxt[k] is None:         # band left k: it arrived at k + 1
                dep[k][-2] = reach[m, (k + 1) % S]
        cur = nxt
    L = max(len(h) for h in hist) + 1
    V = np.zeros((S, L, P * W), np.float32)
    D = np.full((S, L), -1, np.int64)
    n = np.zeros(S, np.int64)
    for k, h in enumerate(hist):
        V[k, :len(h)] = np.stack(h).reshape(len(h), -1)
        V[k, len(h):] = V[k, len(h) - 1]                         # pad with the last value
        D[k, :len(h)] = dep[k]
        n[k] = len(h)
    return V, D, n, reach


def decode(img, strict=0.3):
    """img (S*P, W) -> config, or None when some slot is not cleanly a band or empty."""
    S = img.shape[0] // P
    occ = []
    for k in range(S):
        s = img[k * P:(k + 1) * P]
        if np.abs(s[0]).max() > strict:
            return None
        b = s[1:]
        if np.abs(b).mean() < 0.5:
            if np.abs(b).max() > strict:
                return None
            occ.append(None)
        else:
            if np.abs(b).min() < strict or (np.sign(b[0]) != np.sign(b[1])).any():
                return None
            occ.append(tuple(float(x) for x in np.sign(b[0])))
    return occ


def bands_in_order(occ):
    return [v for v in occ if v is not None]


def same_ring(a, b):
    """Same bands in the same cyclic order."""
    if len(a) != len(b):
        return False
    if not a:
        return True
    return any(a[r:] + a[:r] == b for r in range(len(a)))


# ── model ───────────────────────────────────────────────────────────────────

nca = NeuralCA(channels=C, hidden_channels=args.hidden, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="circular", read_only_dims=[],
               state_bound=10.0).to(DEV)


class MeanFire(torch.nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, delta):
        return delta * self.rate


nca.stochastic_update = MeanFire(args.fire_rate)
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)

root = Path(__file__).resolve().parent.parent
run_dir = root / "runs" / f"QUEUE_{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}_s{args.seed}"
run_dir.mkdir(parents=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
print("run", run_dir.name, flush=True)


def start_state(occs):
    x = np.zeros((len(occs), C, len(occs[0]) * P, W), np.float32)
    for i, occ in enumerate(occs):
        x[i, 0] = paint(occ)
    return torch.from_numpy(x).to(DEV)


# ── training ────────────────────────────────────────────────────────────────

pools = {}


def batch(S):
    pool = pools.get(S, [])
    nb = args.batch_size
    npool = min(len(pool), int(nb * args.pool_frac))
    picks = rng.sample(pool, npool) if npool else []
    occs = [p[1] for p in picks] + [random_config(S, rng) for _ in range(nb - npool)]
    x = start_state(occs)
    if picks:
        x[:npool] = torch.stack([p[0] for p in picks]).to(DEV)
    depth = [p[2] for p in picks] + [0] * (nb - npool)
    return x, occs, depth


def align_loss(seq, occs):
    """Per-slot timing-free loss over every frame of the rollout."""
    B, T1 = seq.shape[:2]
    T = T1 - 1
    S = len(occs[0])
    hs = [histories(o, T) for o in occs]
    L = max(h[0].shape[1] for h in hs)
    V = np.zeros((B, S, L, P * W), np.float32)
    D = np.full((B, S, L), -1, np.int64)
    n = np.zeros((B, S), np.int64)
    req = np.zeros((B, T + 1, S), np.int64)
    for i, (v, dd, nn_, r) in enumerate(hs):
        V[i, :, :v.shape[1]] = v
        V[i, :, v.shape[1]:] = v[:, -1:]
        D[i, :, :dd.shape[1]] = dd
        n[i] = nn_
        req[i] = r[np.arange(T + 1) // args.tau_max]
    V = torch.from_numpy(V).to(DEV)
    D = torch.from_numpy(D).to(DEV)
    n = torch.from_numpy(n).to(DEV)
    req = torch.from_numpy(req).to(DEV)
    j = torch.zeros(B, S, dtype=torch.long, device=DEV)
    bi = torch.arange(B, device=DEV)[:, None]
    si = torch.arange(S, device=DEV)[None]
    tot = 0.0
    forced = 0.0
    for t in range(1, T + 1):
        f = seq[:, t, 0].reshape(B, S, P * W)
        jr = torch.maximum(j, req[:, t])
        forced += float((jr > j).float().mean())
        j = jr
        j1 = torch.minimum(j + 1, n - 1)
        d0 = ((f - V[bi, si, j]) ** 2).mean(-1)
        d1 = ((f - V[bi, si, j1]) ** 2).mean(-1)
        dep = D[bi, si, j]
        arrived = (dep < 0) | (j.roll(-1, dims=1) >= dep)
        adv = (d1 < d0).detach() & (j1 > j) & arrived
        tot = tot + torch.where(adv, d1, d0).mean()
        j = torch.where(adv, j1, j)
    progress = float((j.float() / (n - 1).clamp(min=1).float()).mean())
    return tot / T, forced / T, progress


def pool_push(S, frames, occs, depth):
    """Push the latest frame (of the last 16) that decodes to the same ring."""
    pool = pools.setdefault(S, [])
    ch0 = frames[:, :, 0].cpu().numpy()
    for i, occ in enumerate(occs):
        for t in range(ch0.shape[1] - 1, -1, -1):
            d = decode(ch0[i, t])
            if d is not None:
                if same_ring(bands_in_order(d), bands_in_order(occ)):
                    pool.append((frames[i, t].cpu(), d, depth[i] + 1))
                break
    del pool[:-args.pool_size]


@torch.no_grad()
def evaluate():
    S, B = args.eval_slots, args.eval_batch
    erng = random.Random(1234)
    occs = [random_config(S, erng) for _ in range(B)]
    x = start_state(occs)
    want = [bands_in_order(o) for o in occs]
    prev = (x[:, 0].reshape(B, S, P * W)[:, :, W:].abs().mean(-1) > 0.5)
    events = torch.zeros(B, device=DEV)
    out, t = {}, 0
    for horizon in EVAL_STEPS:
        while t < horizon:
            k = min(4, horizon - t)
            x = nca(x, k)[:, -1]
            t += k
            occ = (x[:, 0].reshape(B, S, P * W)[:, :, W:].abs().mean(-1) > 0.5)
            events += (occ != prev).float().sum(1)
            prev = occ
        # intact if any of the next 16 frames decodes to the painted ring
        tail = nca(x, 16)[:, :, 0].cpu().numpy()
        ok = 0
        for i in range(B):
            for f in tail[i]:
                d = decode(f)
                if d is not None and same_ring(bands_in_order(d), want[i]):
                    ok += 1
                    break
        out[horizon] = (round(ok / B, 3), round(float(events.mean()) / 2 * 1000 / t, 1))
    return out


log = open(run_dir / "log.jsonl", "a")
t0 = time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        ev = evaluate()
        msg = {"step": step, "eval": {str(k): v for k, v in ev.items()}}
        print(step, "eval (intact, moves/1000 steps):", ev, flush=True)
        log.write(json.dumps(msg) + "\n"); log.flush()
    if step % args.ckpt_every == 0 and step:
        torch.save({"nca": nca.state_dict(), "args": vars(args)}, run_dir / f"nca_{step}.pt")
        torch.save({"nca": nca.state_dict(), "args": vars(args)}, run_dir / "nca_latest.pt")
    S = rng.randint(*S_RANGE)
    x, occs, depth = batch(S)
    if args.boundary_noise:
        x = x.clone()
        x[:, 1:] += torch.randn_like(x[:, 1:]) * args.boundary_noise
    seq = nca(x, args.rollout)
    loss, forced, prog = align_loss(seq, occs)
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.grad_clip)
    opt.step()
    pool_push(S, seq[:, -16:].detach(), occs, depth)
    if step % args.log_every == 0:
        d = [p[2] for p in pools.get(S, [])]
        msg = {"step": step, "loss": round(float(loss), 6), "forced": round(forced, 4),
               "S": S, "gn": round(float(gn), 3), "pool": len(d),
               "pool_depth_max": max(d) if d else 0, "sec": round(time.time() - t0)}
        print(msg, flush=True)
        log.write(json.dumps(msg) + "\n"); log.flush()
