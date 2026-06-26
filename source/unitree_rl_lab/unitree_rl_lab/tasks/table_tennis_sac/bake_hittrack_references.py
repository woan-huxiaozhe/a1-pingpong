"""Bake real 120 Hz deployment recordings into 100 Hz HitTrack reference streams.

Consumes the real-robot recording schema (deployment side, 120 Hz) described in the HitTrack
design doc §3.1 -- per row, whitespace-separated::

    col  0      : t          (unified timestamp, s)
    col  1      : serve_id    (int)
    col  2..7   : mocap       x, y, z, vx, vy, vz   (ground-truth ball state)
    col  8..13  : kf          x, y, z, vx, vy, vz   (KF current filtered estimate)
    col 14..19  : kf_pred     y, z, vx, vy, vz, tau (KF prediction at the hit plane x=-1.37,
                                                      which is fixed and not recorded, + tau)

and produces, per serve, the streams the HitTrack reset loads:

  * ``noisy_ball_stream[S, T, 5]`` -- ``kf_pred (y,z,vx,vy,vz)`` resampled to the 100 Hz grid.
  * ``clean_ball_state[S, 6]``     -- ``(x=hit_plane, y, z, vx, vy, vz)`` solved from the mocap
                                      trajectory's true crossing of ``x = hit_plane_x``.
  * ``tau_true[S, T]``             -- ``t_cross - grid_t``.
  * ``valid_len[S]``               -- number of valid grid steps per serve.
  * ``reachable[S]``               -- loose workspace gate on the true crossing ``(y, z)``.

No planner is run here (the reference planner runs at env reset / deployment, sharing one code
path); this script only emits planner-agnostic ball states. Output: ``hittrack_references.npz``.

Pure numpy, no isaaclab import -- unit-testable standalone (see tests/test_hittrack_baking.py).

GATING NOTE: running this on real serves is blocked on the user recording the new-format 120 Hz
logs (mocap + deployment-KF + KF-pred). The transforms and unit tests do NOT depend on that data;
synthetic curriculum (1)/(2) train without it.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# Column layout of the 120 Hz recording (see module docstring / design §3.1).
_COL_T = 0
_COL_SERVE = 1
_COL_MOCAP = slice(2, 8)  # x, y, z, vx, vy, vz
_COL_KF = slice(8, 14)  # x, y, z, vx, vy, vz
_COL_KFPRED = slice(14, 20)  # y, z, vx, vy, vz, tau
_MIN_COLS = 20


def load_recording(path: str, *, z_offset: float = 0.0) -> dict:
    """Parse one whitespace-separated recording file into per-column arrays.

    Returns a dict with ``t[N]``, ``serve_id[N]``, ``mocap[N,6]``, ``kf[N,6]``,
    ``kf_pred[N,6]`` (``y,z,vx,vy,vz,tau``). ``z_offset`` is added to the mocap/kf z (the same
    real->sim height shift ``create_serve_states.py`` applies)."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < _MIN_COLS:
                continue
            try:
                row = [float(p) for p in parts[:_MIN_COLS]]
            except ValueError:
                continue
            if np.isfinite(row).all():
                rows.append(row)
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    arr = np.asarray(rows, dtype=float)
    mocap = arr[:, _COL_MOCAP].copy()
    kf = arr[:, _COL_KF].copy()
    mocap[:, 2] += z_offset
    kf[:, 2] += z_offset
    return {
        "t": arr[:, _COL_T].copy(),
        "serve_id": arr[:, _COL_SERVE].astype(np.int64),
        "mocap": mocap,
        "kf": kf,
        "kf_pred": arr[:, _COL_KFPRED].copy(),
    }


def true_crossing(mocap_traj: np.ndarray, t: np.ndarray, hit_plane_x: float):
    """Interpolated ball state ``[x,y,z,vx,vy,vz]`` and time at the first crossing of
    ``x = hit_plane_x`` in the continuous mocap trajectory. Returns ``(state6, t_cross)`` or
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


def bake(records: list[dict], *, hit_plane_x: float, step_dt: float, reach_y_range, reach_z_range) -> dict:
    """Build padded per-serve reference streams from parsed recordings.

    ``records`` is a list of ``load_recording`` dicts. Serves are grouped by ``serve_id`` within
    each record; a serve is dropped if its mocap never crosses the hit plane."""
    per_serve_noisy = []
    per_serve_clean = []
    per_serve_tau = []
    per_serve_len = []
    per_serve_reach = []

    for rec in records:
        t = rec["t"]
        serve_id = rec["serve_id"]
        mocap = rec["mocap"]
        kf_pred = rec["kf_pred"]
        for sid in np.unique(serve_id):
            m = serve_id == sid
            if int(m.sum()) < 2:
                continue
            ts = t[m]
            order = np.argsort(ts)
            ts = ts[order]
            mocap_s = mocap[m][order]
            kfp_s = kf_pred[m][order]

            crossing = true_crossing(mocap_s, ts, hit_plane_x)
            if crossing is None:
                continue
            state6, t_cross = crossing
            clean = np.array([hit_plane_x, state6[1], state6[2], state6[3], state6[4], state6[5]], dtype=np.float32)

            # 100 Hz grid from the first KF-pred sample up to (and including) the true crossing.
            t0 = float(ts[0])
            n_grid = max(int(np.floor((t_cross - t0) / step_dt)) + 1, 1)
            grid_t = t0 + step_dt * np.arange(n_grid)
            noisy = resample_to_grid(ts, kfp_s[:, 0:5], grid_t).astype(np.float32)  # (y,z,vx,vy,vz)
            tau = (t_cross - grid_t).astype(np.float32)

            per_serve_noisy.append(noisy)
            per_serve_clean.append(clean)
            per_serve_tau.append(tau)
            per_serve_len.append(n_grid)
            per_serve_reach.append(_reachable(float(clean[1]), float(clean[2]), reach_y_range, reach_z_range))

    if not per_serve_clean:
        raise SystemExit("no serves crossed the hit plane; check data / hit_plane_x / columns")

    s = len(per_serve_clean)
    t_max = max(per_serve_len)
    noisy_stream = np.zeros((s, t_max, 5), dtype=np.float32)
    tau_true = np.zeros((s, t_max), dtype=np.float32)
    for i in range(s):
        ln = per_serve_len[i]
        noisy_stream[i, :ln] = per_serve_noisy[i]
        tau_true[i, :ln] = per_serve_tau[i]
    return {
        "noisy_ball_stream": noisy_stream,
        "clean_ball_state": np.stack(per_serve_clean).astype(np.float32),
        "tau_true": tau_true,
        "valid_len": np.asarray(per_serve_len, dtype=np.int64),
        "reachable": np.asarray(per_serve_reach, dtype=bool),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", nargs="+", required=True, help="dir(s) of *.txt recordings")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hittrack_references.npz"),
    )
    parser.add_argument("--hit-plane-x", type=float, default=-1.37)
    parser.add_argument("--step-dt", type=float, default=0.01)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--reach-y-range", type=float, nargs=2, default=(-0.6, 0.6))
    parser.add_argument("--reach-z-range", type=float, nargs=2, default=(0.7, 1.5))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = []
    for d in args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(d, "*.txt"))))
    if not files:
        raise SystemExit(f"no .txt recordings found in {args.data_dir}")
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
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **baked)
    print(
        f"[save] {args.out} serves={baked['clean_ball_state'].shape[0]} "
        f"T={baked['noisy_ball_stream'].shape[1]} reachable={int(baked['reachable'].sum())}"
    )


if __name__ == "__main__":
    main()
