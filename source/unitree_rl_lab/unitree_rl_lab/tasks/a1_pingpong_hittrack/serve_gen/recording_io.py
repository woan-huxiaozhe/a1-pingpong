"""Load real deployment recordings (``kalman_trajectory_*.csv``) into per-serve arrays.

Schema (header line, comma-separated), the same the C++ prediction node logged and that
``bake_hittrack_references.py`` consumes::

    t, serve_id,
    mocap_x, mocap_y, mocap_z, mocap_vx, mocap_vy, mocap_vz,
    KF_x, KF_y, KF_z, KF_vx, KF_vy, KF_vz,
    KF_pred_y, KF_pred_z, KF_pred_vx, KF_pred_vy, KF_pred_vz,
    tau, valid

Positions/velocities are in METERS / m·s⁻¹ in the recording frame (table surface ~0.028 m; the
bake step later adds a +0.73 m z-offset to reach the sim table top 0.76 m). One file is a whole
session containing many serves keyed by ``serve_id``; we group + time-sort per serve.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

_COLS = [
    "t", "serve_id",
    "mocap_x", "mocap_y", "mocap_z", "mocap_vx", "mocap_vy", "mocap_vz",
    "KF_x", "KF_y", "KF_z", "KF_vx", "KF_vy", "KF_vz",
    "KF_pred_y", "KF_pred_z", "KF_pred_vx", "KF_pred_vy", "KF_pred_vz",
    "tau",
]  # 20 numeric cols; col 20 ("valid") is a True/False token handled separately
_TRUE = ("1", "1.0", "true", "t", "yes")


@dataclass
class Serve:
    serve_id: int
    t: np.ndarray          # [N] seconds (relative to file start)
    mocap: np.ndarray      # [N,6] x,y,z,vx,vy,vz (m, m/s)
    kf: np.ndarray         # [N,6] x,y,z,vx,vy,vz (m, m/s) -- smooth filter estimate
    kf_pred: np.ndarray    # [N,5] y,z,vx,vy,vz at the deploy hit plane (m, m/s)
    valid: np.ndarray      # [N] bool (KF-converged flag)


def _parse_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(num[N,20], valid[N] bool)`` for one recording file."""
    num_rows: list[list[float]] = []
    valid_rows: list[bool] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            if len(parts) < 20:
                continue
            try:
                row = [float(p) for p in parts[:20]]
            except ValueError:
                continue  # header line / non-numeric
            if not np.isfinite(row).all():
                continue
            v = parts[20].strip().lower() in _TRUE if len(parts) > 20 else True
            num_rows.append(row)
            valid_rows.append(v)
    if not num_rows:
        raise ValueError(f"no numeric rows parsed from {path}")
    return np.asarray(num_rows, dtype=float), np.asarray(valid_rows, dtype=bool)


def load_file(path: str) -> list[Serve]:
    """Parse one recording file into a list of time-sorted :class:`Serve` per ``serve_id``."""
    num, valid = _parse_file(path)
    idx = {c: i for i, c in enumerate(_COLS)}
    serves: list[Serve] = []
    for sid in np.unique(num[:, idx["serve_id"]]):
        m = num[:, idx["serve_id"]] == sid
        sub = num[m]
        vsub = valid[m]
        order = np.argsort(sub[:, idx["t"]])
        sub, vsub = sub[order], vsub[order]
        serves.append(
            Serve(
                serve_id=int(sid),
                t=sub[:, idx["t"]].copy(),
                mocap=sub[:, idx["mocap_x"]: idx["mocap_vz"] + 1].copy(),
                kf=sub[:, idx["KF_x"]: idx["KF_vz"] + 1].copy(),
                kf_pred=sub[:, idx["KF_pred_y"]: idx["KF_pred_vz"] + 1].copy(),
                valid=vsub.copy(),
            )
        )
    return serves


def load_dir(data_dir: str) -> list[Serve]:
    """Load + concatenate all serves from every ``*.csv`` in ``data_dir``."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no *.csv recordings in {data_dir}")
    out: list[Serve] = []
    for f in files:
        try:
            out.extend(load_file(f))
        except ValueError:
            continue
    return out
