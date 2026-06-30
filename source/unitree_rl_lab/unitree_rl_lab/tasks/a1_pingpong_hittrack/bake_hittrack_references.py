"""Bake real deployment recordings into 100 Hz HitTrack reference streams.

Consumes the real-robot recording schema (deployment side; the 0629 logs sample at ~300 Hz, but
any rate works -- everything is resampled onto the 100 Hz grid). Per row, comma-separated CSV with
a header line (whitespace-separated ``.txt`` without a header is also accepted)::

    col  0      : t          (unified timestamp, s; resets per file/serve)
    col  1      : serve_id    (int)
    col  2..7   : mocap       x, y, z, vx, vy, vz   (raw mocap -- POSITION is reliable, VELOCITY is
                                                     noisy/spiky on dropouts, so it is reference-only)
    col  8..13  : kf          x, y, z, vx, vy, vz   (KF current filtered estimate -- smooth)
    col 14..18  : kf_pred     y, z, vx, vy, vz      (KF prediction at the hit plane x=-1.37)
    col 19      : tau          (KF's own time-to-hit estimate; unused -- we recompute tau from the
                                grid relative to the true crossing)
    col 20      : valid        (bool "True"/"False"; KF-converged flag, optional)

and produces, per serve, the streams the HitTrack reset loads:

  * ``noisy_ball_stream[S, T, 5]`` -- ``kf_pred (y,z,vx,vy,vz)`` resampled onto the 100 Hz grid.
  * ``clean_ball_state[S, 6]``     -- ``(x=hit_plane, y, z, vx, vy, vz)`` at the KF trajectory's
                                      crossing of ``x = hit_plane_x``. POSITION + VELOCITY both come
                                      from the KF estimate (not raw mocap): mocap velocity is too
                                      noisy and produces 100s-of-m/s spikes on dropouts.
  * ``tau_true[S, T]``             -- ``t_cross - grid_t``.
  * ``valid_len[S]``               -- number of valid grid steps per serve.
  * ``reachable[S]``               -- loose workspace gate on the crossing ``(y, z)``.

Per-serve windowing (so each baked stream drops into a fixed-horizon episode and the hit lands
inside it):

  * start  = the FIRST KF-valid sample (natural lock-on; its ball-x varies serve to serve, which is
             a free observation-horizon randomization that also matches real deployment).
  * GATE   = the start is kept only if its ball-x is still beyond ``start_x_gate`` (incoming side),
             so the policy always sees the ball while it is far enough out. Serves whose tracking
             only locked on after the ball passed the gate are dropped.
  * crossing detection runs on the SMOOTH ``kf_x`` (not mocap_x), so a single-frame mocap teleport
             cannot fabricate a spurious early crossing.
  * VELOCITY gate: drop a serve whose crossing speed is outside ``[v_min, v_max]`` (catches the few
             recordings corrupted at lock-on).

Lateral (y) data-augmentation is NOT done here -- it is applied continuously at env reset
(``reference_commands.reset_reference_command``) so the npz stays small and the augmentation is
infinite rather than a fixed discrete set.

Pure numpy, no isaaclab import -- unit-testable standalone (see tests/test_hittrack_baking.py).
Output: ``hittrack_references.npz``.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# Column layout of the recording (see module docstring).
_COL_T = 0
_COL_SERVE = 1
_COL_MOCAP = slice(2, 8)  # x, y, z, vx, vy, vz
_COL_KF = slice(8, 14)  # x, y, z, vx, vy, vz
_COL_KFPRED = slice(14, 19)  # y, z, vx, vy, vz  (5 cols; tau at col 19 is recomputed, not used)
_COL_VALID = 20
_MIN_COLS = 19  # need at least t..kf_pred(vz); valid is optional
_TRUE_TOKENS = ("1", "1.0", "true", "t", "yes")


def load_recording(path: str, *, z_offset: float = 0.0) -> dict:
    """Parse one recording file (CSV-with-header or whitespace ``.txt``) into per-column arrays.

    Returns a dict with ``t[N]``, ``serve_id[N]``, ``mocap[N,6]``, ``kf[N,6]``,
    ``kf_pred[N,5]`` (``y,z,vx,vy,vz``), ``valid[N]`` (bool). ``z_offset`` is added to every
    height channel -- ``mocap.z``, ``kf.z`` AND ``kf_pred.z`` -- so clean (from kf) and noisy
    (from kf_pred) share one real->sim height frame (the shift ``create_serve_states.py`` applies).
    """
    rows = []
    valids = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",") if "," in s else s.split()
            if len(parts) < _MIN_COLS:
                continue
            try:
                row = [float(p) for p in parts[: _COL_KFPRED.stop]]  # first 19 numeric cols
            except ValueError:
                continue  # header line / non-numeric row
            if not np.isfinite(row).all():
                continue
            valid = True
            if len(parts) > _COL_VALID:
                valid = parts[_COL_VALID].strip().lower() in _TRUE_TOKENS
            rows.append(row)
            valids.append(valid)
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    arr = np.asarray(rows, dtype=float)
    mocap = arr[:, _COL_MOCAP].copy()
    kf = arr[:, _COL_KF].copy()
    kf_pred = arr[:, _COL_KFPRED].copy()
    mocap[:, 2] += z_offset
    kf[:, 2] += z_offset
    kf_pred[:, 1] += z_offset  # kf_pred is (y, z, vx, vy, vz) -> z is index 1
    return {
        "t": arr[:, _COL_T].copy(),
        "serve_id": arr[:, _COL_SERVE].astype(np.int64),
        "mocap": mocap,
        "kf": kf,
        "kf_pred": kf_pred,
        "valid": np.asarray(valids, dtype=bool),
    }


def true_crossing(mocap_traj: np.ndarray, t: np.ndarray, hit_plane_x: float):
    """Interpolated ball state ``[x,y,z,vx,vy,vz]`` and time at the first crossing of
    ``x = hit_plane_x`` in the continuous trajectory. Returns ``(state6, t_cross)`` or
    ``None`` if the plane is never crossed (direction-agnostic straddle test)."""
    x = mocap_traj[:, 0]
    d = x - hit_plane_x
    for i in range(len(x) - 1):
        if d[i] == 0.0:
            return mocap_traj[i].copy(), float(t[i])
        if d[i] * d[i + 1] < 0.0:
            alpha = d[i] / (d[i] - d[i + 1])  # in (0, 1)
            state = mocap_traj[i] + alpha * (mocap_traj[i + 1] - mocap_traj[i])
            t_cross = t[i] + alpha * (t[i + 1] - t[i])
            return state, float(t_cross)
    return None


def kf_downward_crossing(kf_traj: np.ndarray, t: np.ndarray, hit_plane_x: float, start: int = 0):
    """First DOWNWARD crossing (x decreasing through ``hit_plane_x``) at or after index ``start``.

    Returns ``(state6, t_cross, idx)`` -- the interpolated KF ball state, the crossing time, and the
    pre-crossing sample index -- or ``None``. Restricting to downward crossings + starting at the
    lock-on index ignores the receding tail of long recordings (ball bouncing back past the robot).
    """
    x = kf_traj[:, 0]
    d = x - hit_plane_x
    for i in range(start, len(x) - 1):
        if d[i] > 0.0 and d[i + 1] <= 0.0:
            alpha = d[i] / (d[i] - d[i + 1])  # in (0, 1]
            state = kf_traj[i] + alpha * (kf_traj[i + 1] - kf_traj[i])
            t_cross = t[i] + alpha * (t[i + 1] - t[i])
            return state, float(t_cross), i
    return None


def resample_to_grid(t: np.ndarray, values: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """Per-column linear resample of ``values[N,D]`` sampled at times ``t[N]`` onto ``grid_t[M]``.
    Returns ``[M, D]`` (``np.interp`` clamps outside the source range)."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    out = np.empty((len(grid_t), values.shape[1]), dtype=float)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(grid_t, t, values[:, j])
    return out


def _reachable(cy: float, cz: float, reach_y_range, reach_z_range) -> bool:
    return bool(reach_y_range[0] <= cy <= reach_y_range[1] and reach_z_range[0] <= cz <= reach_z_range[1])


def bake(
    records: list[dict],
    *,
    hit_plane_x: float,
    step_dt: float,
    reach_y_range,
    reach_z_range,
    start_x_gate: float = 0.6,
    v_min: float = 1.0,
    v_max: float = 6.0,
) -> dict:
    """Build padded per-serve reference streams from parsed recordings.

    Serves are grouped by ``serve_id`` within each record. A serve is dropped when: it has <2 valid
    samples; its lock-on ball-x is already <= ``start_x_gate`` (tracked too late); ``kf_x`` never
    crosses the hit plane after lock-on; or its crossing speed is outside ``[v_min, v_max]``.
    """
    per_serve_noisy: list[np.ndarray] = []
    per_serve_clean: list[np.ndarray] = []
    per_serve_tau: list[np.ndarray] = []
    per_serve_len: list[int] = []
    per_serve_reach: list[bool] = []
    n_gate = n_cross = n_vel = n_short = 0

    for rec in records:
        t = rec["t"]
        serve_id = rec["serve_id"]
        kf = rec["kf"]
        kf_pred = rec["kf_pred"]
        valid = rec.get("valid")
        for sid in np.unique(serve_id):
            m = serve_id == sid
            if int(m.sum()) < 2:
                n_short += 1
                continue
            ts = t[m]
            order = np.argsort(ts)
            ts = ts[order]
            kf_s = kf[m][order]
            kfp_s = kf_pred[m][order]
            vmask = (valid[m][order] if valid is not None else np.ones(len(ts), dtype=bool))

            vi = np.where(vmask)[0]
            if len(vi) < 2:
                n_short += 1
                continue
            i0 = int(vi[0])  # lock-on = first KF-valid sample (natural, randomized horizon start)

            # start gate: keep only serves picked up while the ball is still beyond the gate plane
            if kf_s[i0, 0] <= start_x_gate:
                n_gate += 1
                continue

            cr = kf_downward_crossing(kf_s, ts, hit_plane_x, start=i0)
            if cr is None:
                n_cross += 1
                continue
            state6, t_cross, _ = cr

            speed = float(np.linalg.norm(state6[3:6]))
            if not (v_min <= speed <= v_max):
                n_vel += 1
                continue

            clean = np.array(
                [hit_plane_x, state6[1], state6[2], state6[3], state6[4], state6[5]], dtype=np.float32
            )

            # 100 Hz grid from lock-on (t0) up to (and including) the crossing.
            t0 = float(ts[i0])
            n_grid = max(int(np.floor((t_cross - t0) / step_dt)) + 1, 1)
            grid_t = t0 + step_dt * np.arange(n_grid)
            noisy = resample_to_grid(ts[i0:], kfp_s[i0:, 0:5], grid_t).astype(np.float32)  # (y,z,vx,vy,vz)
            tau = (t_cross - grid_t).astype(np.float32)

            per_serve_noisy.append(noisy)
            per_serve_clean.append(clean)
            per_serve_tau.append(tau)
            per_serve_len.append(n_grid)
            per_serve_reach.append(_reachable(float(clean[1]), float(clean[2]), reach_y_range, reach_z_range))

    if not per_serve_clean:
        raise SystemExit("no serves passed the gates; check data / hit_plane_x / start_x_gate / columns")

    s = len(per_serve_clean)
    t_max = max(per_serve_len)
    noisy_stream = np.zeros((s, t_max, 5), dtype=np.float32)
    tau_true = np.zeros((s, t_max), dtype=np.float32)
    for i in range(s):
        ln = per_serve_len[i]
        noisy_stream[i, :ln] = per_serve_noisy[i]
        tau_true[i, :ln] = per_serve_tau[i]
    print(
        f"[bake] kept={s}  dropped: gate(x<={start_x_gate})={n_gate} "
        f"no-crossing={n_cross} vel-outside[{v_min},{v_max}]={n_vel} too-short={n_short}"
    )
    return {
        "noisy_ball_stream": noisy_stream,
        "clean_ball_state": np.stack(per_serve_clean).astype(np.float32),
        "tau_true": tau_true,
        "valid_len": np.asarray(per_serve_len, dtype=np.int64),
        "reachable": np.asarray(per_serve_reach, dtype=bool),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", nargs="+", required=True, help="dir(s) of recordings (*.csv or *.txt)")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hittrack_references.npz"),
    )
    parser.add_argument("--hit-plane-x", type=float, default=-1.37)
    parser.add_argument("--step-dt", type=float, default=0.01)
    # 0629-frame real->sim height shift (table-surface ~0.03 in data -> sim table top 0.76). The
    # 0602 calibration used 0.714; ~0.73 lands the crossing z near the calibrated ~1.0-1.08 band.
    parser.add_argument("--z-offset", type=float, default=0.73)
    parser.add_argument("--start-x-gate", type=float, default=0.6, help="lock-on ball-x must exceed this")
    parser.add_argument("--v-min", type=float, default=1.0, help="drop serves slower than this at crossing")
    parser.add_argument("--v-max", type=float, default=6.0, help="drop serves faster than this at crossing")
    parser.add_argument("--reach-y-range", type=float, nargs=2, default=(-0.6, 0.6))
    parser.add_argument("--reach-z-range", type=float, nargs=2, default=(0.7, 1.5))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = []
    for d in args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))
        files.extend(sorted(glob.glob(os.path.join(d, "*.txt"))))
    if not files:
        raise SystemExit(f"no *.csv or *.txt recordings found in {args.data_dir}")
    records = []
    for path in files:
        try:
            records.append(load_recording(path, z_offset=args.z_offset))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path}: {exc}")
    baked = bake(
        records,
        hit_plane_x=args.hit_plane_x,
        step_dt=args.step_dt,
        reach_y_range=tuple(args.reach_y_range),
        reach_z_range=tuple(args.reach_z_range),
        start_x_gate=args.start_x_gate,
        v_min=args.v_min,
        v_max=args.v_max,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **baked)
    print(
        f"[save] {args.out} serves={baked['clean_ball_state'].shape[0]} "
        f"T={baked['noisy_ball_stream'].shape[1]} reachable={int(baked['reachable'].sum())}"
    )


if __name__ == "__main__":
    main()
