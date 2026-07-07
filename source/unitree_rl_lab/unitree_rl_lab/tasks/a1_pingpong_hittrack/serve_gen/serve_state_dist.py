"""Joint distribution over the fitted serve latents, for sampling new physically-faithful serves.

For the drag/bounce model the 9-D latent is [pos0(mm,3), v0(mm/s,3), air_drag_coeff, bounce_alpha_z,
bounce_alpha_xy]. Fit a multivariate Gaussian (light shrinkage); ``envelope_scale`` inflates the
covariance (1.0 = same-distribution; >1 widens coverage). Samples are clipped to physical bounds so
drag/bounce stay valid. ``tau_lockon`` is NOT sampled -- the flight length is defined by rolling the
physics until the ball crosses the plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# physical clip bounds for [pos0(3), v0(3), drag, alpha_z, alpha_xy]
_LO = np.array([-np.inf] * 6 + [1e-5, 0.5, 0.3])
_HI = np.array([np.inf] * 6 + [5e-4, 1.0, 1.0])


@dataclass
class ServeDist:
    mean: np.ndarray   # [9]
    cov: np.ndarray    # [9,9]

    def sample(self, n: int, envelope_scale: float = 1.0, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        cov = self.cov * float(envelope_scale) ** 2
        s = rng.multivariate_normal(self.mean, cov, size=n)
        s = np.clip(s, _LO, _HI)
        # returns pos0_mm[n,3], v0_mm_s[n,3], drag[n], alpha_z[n], alpha_xy[n]
        return s[:, 0:3], s[:, 3:6], s[:, 6], s[:, 7], s[:, 8]


def fit_dist(fit_npz_path: str, shrinkage: float = 0.08) -> ServeDist:
    d = np.load(fit_npz_path, allow_pickle=True)
    X = np.column_stack([d["pos0_mm"], d["v0_mm_s"], d["drag"], d["alpha_z"], d["alpha_xy"]])  # [S,9]
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    return ServeDist(mean=mean, cov=cov)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="fitted_serves.npz")
    args = ap.parse_args()
    dist = fit_dist(args.fit)
    lbl = ["pos0_x", "pos0_y", "pos0_z", "v0_x", "v0_y", "v0_z", "drag", "alpha_z", "alpha_xy"]
    sd = np.sqrt(np.diag(dist.cov))
    print("=== serve latent distribution (mean +/- std) ===")
    for i, name in enumerate(lbl):
        print(f"  {name:9s} {dist.mean[i]:+.4g} +/- {sd[i]:.4g}")
