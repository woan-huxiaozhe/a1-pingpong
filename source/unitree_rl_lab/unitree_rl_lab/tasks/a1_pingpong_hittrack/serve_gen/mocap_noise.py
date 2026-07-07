"""Mocap measurement-noise model, calibrated from real (mocap - KF) position residuals.

The vendored KF observes POSITION only, so the sole thing the generator must add to a clean
physics trajectory before feeding the KF is realistic position measurement noise. We calibrate its
per-axis magnitude from the real recordings: over KF-valid frames, the residual (mocap_xyz -
KF_xyz) is the measurement's deviation from the smooth estimate -- i.e. the mocap noise the KF was
tuned against (``measurement_std=5 mm`` is the KF's assumed value; we measure the empirical one).

Kept white + per-axis for v1 (matches the KF's diagonal R). Dropouts/glitches are handled by the
KF/gate already and are left out until the synthetic-vs-real KF_pred curve says they matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .recording_io import Serve
except ImportError:
    from recording_io import Serve

MM = 1000.0


@dataclass
class NoiseModel:
    std_xyz_mm: np.ndarray  # [3] per-axis white gaussian position-noise std (mm)

    def apply(self, clean_xyz_mm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add white per-axis gaussian noise to a clean position stream ``[N,3]`` (mm)."""
        noise = rng.normal(0.0, 1.0, size=clean_xyz_mm.shape) * self.std_xyz_mm[None, :]
        return clean_xyz_mm + noise


def calibrate(serves: list[Serve], min_kf_x: float = -1.44, max_kf_x: float = 1.6) -> NoiseModel:
    """Per-axis std of (mocap - KF) over valid, in-flight frames (m -> mm)."""
    res = []
    for s in serves:
        m = s.valid & (s.kf[:, 0] >= min_kf_x) & (s.kf[:, 0] <= max_kf_x)
        if m.sum() == 0:
            continue
        res.append((s.mocap[m, 0:3] - s.kf[m, 0:3]))
    r = np.concatenate(res, axis=0) * MM  # mm
    # robust std via MAD (mocap has occasional large glitches that a plain std would chase)
    med = np.median(r, axis=0)
    mad = np.median(np.abs(r - med[None, :]), axis=0)
    std_robust = 1.4826 * mad
    return NoiseModel(std_xyz_mm=std_robust)


def _report(data_dir: str) -> None:
    try:
        from .recording_io import load_dir
    except ImportError:
        from recording_io import load_dir
    serves = load_dir(data_dir)
    nm = calibrate(serves)
    # also plain std for contrast
    res = []
    for s in serves:
        m = s.valid & (s.kf[:, 0] >= -1.44) & (s.kf[:, 0] <= 1.6)
        if m.sum():
            res.append(s.mocap[m, 0:3] - s.kf[m, 0:3])
    r = np.concatenate(res, axis=0) * MM
    print("=== mocap measurement-noise calibration (mocap - KF), mm ===")
    print(f"  robust std (MAD) x/y/z = {nm.std_xyz_mm[0]:.2f} / {nm.std_xyz_mm[1]:.2f} / {nm.std_xyz_mm[2]:.2f}")
    print(f"  plain  std        x/y/z = {r[:,0].std():.2f} / {r[:,1].std():.2f} / {r[:,2].std():.2f}")
    print(f"  KF assumed measurement_std = 5.00 mm")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/PPO-pingpong/data/0629_RL_traj")
    _report(ap.parse_args().data_dir)
