"""Orbit-counter NCA encoding — the smallest task that needs a real loop.

A dim closed racetrack is drawn on the grid.  A bright comet must run around it
forever, and a tally register below the track must gain one count every time the
comet passes a fixed tap point::

    comet position   p(t) = (p0 + speed * t)  mod  L        (L = track perimeter)
    counter value    c(t) = c0 + (number of tap crossings up to step t)

Nothing about this is solvable by a fixed unroll: the trip count between two
counter states depends on where the comet started, and the answer at step t
depends on how many laps have happened, which is unbounded.  The comet's
position *is* the program counter, so no clock or halt channel is needed — the
loop phase is visible on screen.

Encoding conventions (same spirit as ``subleq.py``)
--------------------------------------------------
* Screens are float32 in [-1, 1]; background is neutral grey 0.0.
* The track is a permanent dim corridor at ``TRACK``; the comet is a bright
  disc at ``TOKEN`` with a quadratically fading tail behind it.  The tail is
  what makes the direction of travel readable from a single frame — without it
  a symmetric ring gives the NCA no way to know which way to go.
* The tally register is a row of discs below the track: ``MARK0`` = empty,
  ``MARK1`` = counted.  ``counter="unary"`` fills them left to right and
  saturates; ``counter="binary"`` treats them as bits, MSB-left.
* Channel 0 carries the mutable state; a frozen copy of the *geometry only*
  (track + empty register, no comet) is implanted in a read-only channel so the
  NCA always has a map of the racetrack.

Counter latency
---------------
The trigger has to travel from the tap to the register, and a binary increment
then has to ripple a carry across the marks.  That takes more steps than one
pixel of comet travel, so the counter is *not* supervised for ``settle`` frames
after each crossing — the returned ``cmask`` masks out that grace window.  The
comet is supervised on every frame.
"""

import numpy as np
import torch

BACKGROUND = 0.0
TRACK = -0.6
TOKEN = +1.0
MARK0 = -1.0
MARK1 = +1.0


class OrbitLayout:
    """Geometry + rendering for the orbit-counter task.

    Args:
        W, H:       grid size
        pad:        margin between the grid edge and the outermost ink
        r:          comet / register-mark radius
        track_pad:  corridor radius is ``r + track_pad`` (margin around the comet)
        marks:      number of register discs (unary: max count; binary: bits)
        mark_step:  horizontal spacing between register discs
        counter:    "unary" (saturating tally) or "binary" (wrapping bit field)
        tail:       comet tail length in pixels
        speed:      comet speed in pixels per NCA step.  A 3x3 kernel moves
                    information at most 1 px/step, so speed must be <= 1 — and
                    speed == 1 leaves zero slack (a perfect soliton), so the
                    default 0.5 gives the rule two steps per pixel of travel.
        settle:     frames of counter grace after a crossing (None → derived)
        tap:        where the counter trigger sits — "bottom" | "top" | "left"
                    | "right" | "tl"
    """

    def __init__(
        self,
        W=44,
        H=44,
        pad=4,
        r=2,
        track_pad=1,
        marks=5,
        mark_step=7,
        counter="unary",
        tail=6,
        speed=0.5,
        settle=None,
        tap="bottom",
    ):
        if counter not in ("unary", "binary"):
            raise ValueError(f"counter must be unary|binary, got {counter!r}")
        if not 0.0 < speed <= 1.0:
            raise ValueError(
                f"speed={speed} must be in (0, 1]; a 3x3 kernel cannot propagate faster "
                "than 1 cell/step"
            )

        self.W, self.H, self.pad, self.r = int(W), int(H), int(pad), int(r)
        self.track_r = self.r + int(track_pad)
        self.marks, self.mark_step = int(marks), int(mark_step)
        self.counter = counter
        self.tail = int(tail)
        self.speed = float(speed)

        # ── register row: centred, one r below the bottom margin ──────────────
        self.y_reg = self.H - 1 - self.pad - self.r
        span = (self.marks - 1) * self.mark_step
        x_reg0 = (self.W - span) // 2
        self.mark_centers = [(x_reg0 + i * self.mark_step, self.y_reg)
                             for i in range(self.marks)]
        if x_reg0 - self.r < 0:
            raise ValueError("register row does not fit — reduce marks or mark_step")

        # ── track rectangle: sits above the register row ─────────────────────
        self.x0 = self.pad + self.track_r
        self.x1 = self.W - 1 - self.pad - self.track_r
        self.y0 = self.pad + self.track_r
        self.y1 = self.y_reg - (3 * self.r + self.track_r + 2)
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        if self.w <= 2 * self.track_r or self.h <= 2 * self.track_r:
            raise ValueError("track rectangle degenerate — grid too small")
        self.L = 2 * (self.w + self.h)          # perimeter, px == steps per lap at speed 1
        self.lap_steps = self.L / self.speed

        # ── tap point (arc position where the counter is triggered) ───────────
        taps = {
            "top": self.w / 2,
            "right": self.w + self.h / 2,
            "bottom": self.w + self.h + self.w / 2,
            "left": 2 * self.w + self.h + self.h / 2,
            "tl": 0.0,
        }
        if tap not in taps:
            raise ValueError(f"tap must be one of {sorted(taps)}, got {tap!r}")
        self.tap_name = tap
        self.tap_s = float(taps[tap])

        # ── counter latency grace window ──────────────────────────────────────
        # The trigger must reach the register (vertical travel), then reach the
        # mark that changes (worst-case lateral travel), and for a binary
        # counter a carry may then ripple the whole width.  Information moves at
        # most 1 px/step, so the grace window is that distance in frames.
        if settle is None:
            tap_x, tap_y = self.point_at(self.tap_s)
            reach = abs(tap_y - self.y_reg) + 2 * self.r
            lateral = max(abs(cx - tap_x) for cx, _ in self.mark_centers)
            ripple = 0 if self.counter == "unary" else (self.marks - 1) * self.mark_step
            settle = reach + lateral + ripple + 4
        self.settle = int(settle)

        self.n_states = self.marks + 1 if self.counter == "unary" else (1 << self.marks)

        # ── rasterised, cached pieces ────────────────────────────────────────
        self._track_bool = self._rasterise_track()
        self.bg = self._geometry_screen()                 # (H, W)
        self.token_delta = self._token_frames()           # (L, H, W)
        self.head_bool = self._head_frames()              # (L, H, W) uint8
        self.reg_delta = self._reg_frames()                # (n_states, H, W)
        self._device_cache = {}

    # ── geometry ─────────────────────────────────────────────────────────────

    def point_at(self, s):
        """Clockwise from the top-left corner; returns a float (x, y)."""
        s = float(s) % self.L
        if s < self.w:
            return (self.x0 + s, float(self.y0))
        s -= self.w
        if s < self.h:
            return (float(self.x1), self.y0 + s)
        s -= self.h
        if s < self.w:
            return (self.x1 - s, float(self.y1))
        s -= self.w
        return (float(self.x0), self.y1 - s)

    def _disc(self, cx, cy, radius):
        ys, xs = np.ogrid[: self.H, : self.W]
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2

    def _rasterise_track(self):
        m = np.zeros((self.H, self.W), dtype=bool)
        for s in np.arange(0.0, self.L, 0.5):
            cx, cy = self.point_at(s)
            m |= self._disc(cx, cy, self.track_r)
        return m

    def _geometry_screen(self):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        s[self._track_bool] = TRACK
        for cx, cy in self.mark_centers:
            s[self._disc(cx, cy, self.r)] = MARK0
        return s

    def _token_frames(self):
        """Comet ink as a *delta on top of* TRACK, one frame per integer arc px."""
        out = np.zeros((self.L, self.H, self.W), dtype=np.float32)
        for s in range(self.L):
            f = out[s]
            for k in range(self.tail, 0, -1):          # oldest first, newest wins
                frac = 1.0 - k / (self.tail + 1.0)
                v = TRACK + (TOKEN - TRACK) * frac ** 2
                cx, cy = self.point_at(s - k)
                f[self._disc(cx, cy, self.r)] = v - TRACK
            cx, cy = self.point_at(s)
            f[self._disc(cx, cy, self.r)] = TOKEN - TRACK
        return out

    def _head_frames(self):
        """Head-only disc masks — used to decode the comet position from a frame."""
        out = np.zeros((self.L, self.H, self.W), dtype=np.float32)
        for s in range(self.L):
            cx, cy = self.point_at(s)
            out[s][self._disc(cx, cy, self.r)] = 1.0
        return out

    def _mark_bits(self, value):
        if self.counter == "unary":
            return [1 if i < value else 0 for i in range(self.marks)]
        return [(value >> (self.marks - 1 - i)) & 1 for i in range(self.marks)]

    def _reg_frames(self):
        """Register ink as a delta on top of MARK0, one frame per counter state."""
        out = np.zeros((self.n_states, self.H, self.W), dtype=np.float32)
        for v in range(self.n_states):
            for bit, (cx, cy) in zip(self._mark_bits(v), self.mark_centers):
                if bit:
                    out[v][self._disc(cx, cy, self.r)] = MARK1 - MARK0
        return out

    # ── masks ────────────────────────────────────────────────────────────────

    def track_mask(self):
        return torch.from_numpy(self._track_bool.astype(np.float32))

    def mark_masks(self):
        """(marks, H, W) — one disc per register mark."""
        return torch.from_numpy(
            np.stack([self._disc(cx, cy, self.r).astype(np.float32)
                      for cx, cy in self.mark_centers])
        )

    def register_mask(self):
        return self.mark_masks().sum(0).clamp(0, 1)

    def comet_masks(self, dilate=1):
        """(L, H, W) — the comet head disc at each arc position, dilated by `dilate`.

        Indexed by the per-frame arc position, this gives a loss weight on *where
        the comet is right now* instead of on the whole corridor.  Without it the
        corridor mask is ~99% empty track, so MSE is minimised by erasing the
        comet and painting the flat corridor baseline everywhere — which is
        exactly what the first run did.
        """
        out = np.zeros((self.L, self.H, self.W), dtype=np.float32)
        for s in range(self.L):
            cx, cy = self.point_at(s)
            out[s][self._disc(cx, cy, self.r + dilate)] = 1.0
        return torch.from_numpy(out)

    # ── batch sampling ───────────────────────────────────────────────────────

    def _to(self, device):
        """Cache the big lookup tables on the target device."""
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = dict(
                bg=torch.from_numpy(self.bg).to(device),
                token=torch.from_numpy(self.token_delta).to(device),
                reg=torch.from_numpy(self.reg_delta).to(device),
                head=torch.from_numpy(self.head_bool).to(device),
            )
        return self._device_cache[key]

    def schedule(self, p0, c0, steps):
        """Ground-truth arc index, counter state and supervision mask.

        Args:
            p0: (B,) integer starting arc positions
            c0: (B,) integer starting counter values
            steps: number of NCA steps (frames returned is steps + 1)

        Returns:
            arc:   (B, T) int64 — integer arc position of the comet head
            val:   (B, T) int64 — counter state index
            cmask: (B, T) float32 — 1 where the counter is supervised
            true_arc: (B, T) float64 — exact (fractional) arc position
        """
        p0 = np.asarray(p0).reshape(-1).astype(np.float64)
        c0 = np.asarray(c0).reshape(-1).astype(np.int64)
        t = np.arange(steps + 1, dtype=np.float64)[None, :]

        travelled = self.speed * t                                # (1, T)
        true_arc = (p0[:, None] + travelled) % self.L
        arc = np.floor(true_arc).astype(np.int64) % self.L

        # first crossing after d0 px of travel, then every L px
        d0 = (self.tap_s - p0) % self.L
        d0 = np.where(d0 == 0.0, float(self.L), d0)               # start-on-tap doesn't count
        n_cross = np.floor((travelled - d0[:, None]) / self.L).astype(np.int64) + 1
        n_cross = np.maximum(n_cross, 0)

        val = c0[:, None] + n_cross
        if self.counter == "unary":
            val = np.minimum(val, self.marks)
        else:
            val = val % (1 << self.marks)

        # grace window: `settle` frames after each crossing are unsupervised
        t_last = (d0[:, None] + (n_cross - 1) * self.L) / self.speed
        since = t - t_last
        cmask = ((n_cross < 1) | (since >= self.settle)).astype(np.float32)

        return arc, val, cmask, true_arc

    def sample_batch(self, batch_size, steps, device=None, rng=np.random):
        """Returns (inp, geom, tgt, cmask, meta).

        inp:   (B, H, W)      seed frame (comet at p0, counter at c0)
        geom:  (B, H, W)      static map — track + empty register, no comet
        tgt:   (B, T, H, W)   target frame per rollout step, T = steps + 1
        cmask: (B, T)         1 where the counter is supervised
        meta:  dict with p0, c0, arc (ground-truth position per frame)
        """
        c0_high = self.marks if self.counter == "unary" else (1 << self.marks)
        p0 = rng.randint(0, self.L, size=batch_size)
        c0 = rng.randint(0, max(1, c0_high), size=batch_size)

        arc, val, cmask, true_arc = self.schedule(p0, c0, steps)

        tab = self._to(device if device is not None else "cpu")
        arc_t = torch.from_numpy(arc).to(tab["bg"].device)
        val_t = torch.from_numpy(val).to(tab["bg"].device)

        tgt = tab["bg"] + tab["token"][arc_t] + tab["reg"][val_t]   # (B, T, H, W)
        inp = tgt[:, 0].clone()
        geom = tab["bg"].expand(batch_size, self.H, self.W)

        meta = dict(
            p0=p0, c0=c0,
            arc=arc_t, val=val_t,
            true_arc=torch.from_numpy(true_arc).to(tab["bg"].device),
        )
        return inp, geom, tgt, torch.from_numpy(cmask).to(tab["bg"].device), meta

    # ── read-out ─────────────────────────────────────────────────────────────

    def decode_arc(self, frames, device=None):
        """Comet arc position per frame, by best match against the head discs.

        frames: (..., H, W) → returns int64 (...) arc positions.
        """
        tab = self._to(frames.device if device is None else device)
        head = tab["head"].reshape(self.L, -1)                    # (L, HW)
        flat = frames.reshape(-1, self.H * self.W)                # (N, HW)
        scores = (flat @ head.t()) / head.sum(1)                  # (N, L)
        return scores.argmax(1).reshape(frames.shape[:-2])

    def arc_error(self, pred, true):
        """Circular distance in px between two arc positions."""
        d = (pred.to(torch.float32) - true.to(torch.float32)).abs() % self.L
        return torch.minimum(d, self.L - d)

    def describe(self):
        return (
            f"OrbitLayout {self.W}x{self.H}  track {self.w}x{self.h} @({self.x0},{self.y0})  "
            f"L={self.L}px  lap={self.lap_steps:.0f} steps  speed={self.speed}\n"
            f"  counter={self.counter} marks={self.marks} step={self.mark_step} "
            f"states={self.n_states}  tap={self.tap_name}@{self.tap_s:.0f}px  settle={self.settle}"
        )
