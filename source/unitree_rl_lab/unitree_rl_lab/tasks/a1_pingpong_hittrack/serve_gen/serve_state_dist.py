"""Sampler over the fitted serve latents, for drawing new physically-faithful serves.

The 9-D latent is [pos0(mm,3), v0(mm/s,3), air_drag_coeff, bounce_alpha_z, bounce_alpha_xy].

We do NOT fit a single global Gaussian: sampling from one ellipsoid draws points from the
low-density space *between* the real serve clusters and from the Gaussian tails, producing
physically-incoherent (pos0, v0, drag/bounce) combos that roll to trajectories which cross the plane
too low / descending (an off-manifold drift confirmed against both a Gaussian fit and an independent
real dataset). Instead we treat the S fitted serves as the centers of a **kernel mixture (KDE)**:
each draw picks one real serve and adds a small per-dimension jitter. Sampling density then follows
the real data's S modes, every sample is a near-neighbour of a serve that already reproduces its real
trajectory to ~2 cm, and the crossing-state drift disappears. ``bandwidth_scale`` scales the jitter
(0 = replay the S serves exactly; ~1 = densify each into a continuous local neighbourhood; larger
widens coverage at the cost of drifting off-manifold again). ``interp_frac`` optionally moves a
fraction of draws along the segment to a nearest real neighbour (SMOTE-style hole-filling inside the
real convex hull); default 0 = pure per-center jitter. ``tau_lockon`` is NOT sampled -- flight length
is defined by rolling the physics until the ball crosses the plane. Samples are clipped so drag/bounce
stay physically valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# physical clip bounds for [pos0(3), v0(3), drag, alpha_z, alpha_xy]
_LO = np.array([-np.inf] * 6 + [1e-5, 0.5, 0.3])
_HI = np.array([np.inf] * 6 + [5e-4, 1.0, 1.0])


@dataclass
class ServeDist:
    X: np.ndarray                     # [S,9] fitted serve latents = the mixture centers
    h: np.ndarray                     # [9] per-dim base jitter std (bandwidth)
    nn: np.ndarray                    # [S] index of each center's nearest neighbour (standardized space)
    rmse_cm: np.ndarray | None = None  # [S] per-serve fit quality (info only)

    def sample(self, n: int, bandwidth_scale: float = 1.0, rng: np.random.Generator | None = None,
               interp_frac: float = 0.0):
        """Draw n serves from the kernel mixture over the S real fitted serves.

        Returns pos0_mm[n,3], v0_mm_s[n,3], drag[n], alpha_z[n], alpha_xy[n].
        """
        rng = rng or np.random.default_rng(0)
        S = self.X.shape[0]
        idx = rng.integers(0, S, size=n)
        base = self.X[idx].copy()  # [n,9] real anchors
        if interp_frac > 0.0:
            # move a fraction of draws toward their center's nearest neighbour (stay inside the
            # convex hull of two consistent real serves -> fills gaps without extrapolating)
            m = rng.random(n) < interp_frac
            if m.any():
                w = rng.random(int(m.sum()))[:, None]
                base[m] = (1.0 - w) * base[m] + w * self.X[self.nn[idx[m]]]
        jitter = rng.standard_normal(base.shape) * (self.h * float(bandwidth_scale))
        s = np.clip(base + jitter, _LO, _HI)
        return s[:, 0:3], s[:, 3:6], s[:, 6], s[:, 7], s[:, 8]


def fit_dist(fit_npz_path: str, bandwidth_frac: float = 0.2) -> ServeDist:
    """Load the fitted serves and build the KDE mixture. ``bandwidth_frac`` sets the base jitter as a
    fraction of each dimension's spread (0.2 = 1/5 of the per-dim std -> tight around each real serve).
    """
    d = np.load(fit_npz_path, allow_pickle=True)
    X = np.column_stack([d["pos0_mm"], d["v0_mm_s"], d["drag"], d["alpha_z"], d["alpha_xy"]])  # [S,9]
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    h = bandwidth_frac * std
    # nearest neighbour per center in standardized space (for optional interpolation)
    Z = (X - X.mean(axis=0)) / std
    dmat = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.argmin(axis=1)
    rmse = d["fit_rmse_cm"] if "fit_rmse_cm" in d.files else None
    return ServeDist(X=X, h=h, nn=nn, rmse_cm=rmse)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="fitted_serves.npz")
    ap.add_argument("--bandwidth-frac", type=float, default=0.2)
    args = ap.parse_args()
    dist = fit_dist(args.fit, args.bandwidth_frac)
    lbl = ["pos0_x", "pos0_y", "pos0_z", "v0_x", "v0_y", "v0_z", "drag", "alpha_z", "alpha_xy"]
    print(f"=== serve KDE mixture: {dist.X.shape[0]} centers (mean +/- std ; base jitter h) ===")
    mean = dist.X.mean(axis=0)
    std = dist.X.std(axis=0)
    for i, name in enumerate(lbl):
        print(f"  {name:9s} {mean[i]:+.4g} +/- {std[i]:.4g}   h={dist.h[i]:.4g}")
