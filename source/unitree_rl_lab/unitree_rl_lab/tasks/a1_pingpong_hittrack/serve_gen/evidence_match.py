"""EVIDENCE that generated trajectories match real ones. Compares REAL (0629 recordings) vs GENERATED
(synth_serves) on trajectory-level geometry -- not just the crossing point -- and saves an overlay
figure. All on the KF-smoothed trajectory (apples-to-apples: both are KF tracks).

Features per serve (window = lock-on -> crossing of x=-1.37):
  flight_s        : lock-on to crossing time
  cross_speed     : |v| at the plane crossing (m/s)
  descent_deg     : descent angle at crossing = atan2(-vz,-vx) (deg)
  bounce_x        : x of the trajectory's lowest point (the table bounce, m)
  bounce_z        : that lowest height (m)
  cross_z         : height at the plane (m)

Outputs: side-by-side distribution stats + 2-sample KS test per feature, a Mahalanobis
"in-distribution" check (are real crossing-states typical draws of the generator?), and an overlay
PNG (z-vs-x and y-vs-x).

Usage: python evidence_match.py --real /data/PPO-pingpong/data/0629_RL_traj \
    --synth /data/PPO-pingpong/data/synth_serves --out evidence_overlay.png
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .ball_physics import first_downward_crossing
    from .recording_io import load_dir
except ImportError:
    from ball_physics import first_downward_crossing
    from recording_io import load_dir

PLANE_X = -1.37


def _window(s):
    vi = np.where(s.valid)[0]
    if vi.size < 6:
        return None
    i0 = int(vi[0])
    below = np.where(s.kf[i0:, 0] <= PLANE_X)[0]
    if below.size == 0:
        return None
    iend = i0 + int(below[0]) + 1
    return i0, iend


def feats(s):
    w = _window(s)
    if w is None:
        return None
    i0, iend = w
    seg = s.kf[i0:iend]
    t = s.t[i0:iend]
    cr = first_downward_crossing(seg[:, 0], seg[:, 1], seg[:, 2], t, PLANE_X)
    if cr is None:
        return None
    yz_c, t_c = np.array(cr[:2]), cr[2]
    ci = int(np.argmin(np.abs(seg[:, 0] - PLANE_X)))  # nearest sample to crossing for velocity
    vx, vy, vz = seg[ci, 3], seg[ci, 4], seg[ci, 5]
    bi = int(np.argmin(seg[:, 2]))  # lowest point = bounce
    return dict(
        flight_s=float(t_c - t[0]),
        cross_speed=float(np.hypot(np.hypot(vx, vy), vz)),
        descent_deg=float(np.degrees(np.arctan2(-vz, -vx))),
        bounce_x=float(seg[bi, 0]),
        bounce_z=float(seg[bi, 2]),
        cross_y=float(yz_c[0]),
        cross_z=float(yz_c[1]),
    )


def _in_reach_box(r):
    # the generator's actual acceptance gate is on the CROSSING (recording frame; bake adds +0.73 z):
    #   z_c + 0.73 in [0.7,1.5] and |y_c| <= 0.6.  (bounce_x is NOT part of it.)
    return 0.7 <= r["cross_z"] + 0.73 <= 1.5 and -0.6 <= r["cross_y"] <= 0.6


def collect(serves, reach_gate=False):
    rows = [f for s in serves for f in [feats(s)] if f is not None]
    if reach_gate:  # keep only crossings inside the generator's acceptance box (applied to BOTH sets)
        rows = [r for r in rows if _in_reach_box(r)]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", default="/data/PPO-pingpong/data/0629_RL_traj")
    ap.add_argument("--synth", default="/data/PPO-pingpong/data/synth_serves")
    ap.add_argument("--out", default="/data/PPO-pingpong/data/synth_serves/evidence_overlay.png")
    args = ap.parse_args()

    R = load_dir(args.real)
    S = load_dir(args.synth)
    rf, _ = collect(R, reach_gate=True)  # same acceptance box on BOTH sets -> apples-to-apples
    sf, _ = collect(S, reach_gate=True)

    try:
        from scipy.stats import ks_2samp
        have_ks = True
    except Exception:
        have_ks = False

    print(f"\n=== trajectory-geometry match: REAL(n={len(next(iter(rf.values())))}) vs GENERATED(n={len(next(iter(sf.values())))}) ===")
    print(f"{'feature':13s} {'REAL mean±std':>20s} {'GEN mean±std':>20s} {'REAL rng':>16s} {'GEN rng':>16s}" + ("   KS p" if have_ks else ""))
    for k in rf:
        r, s = rf[k], sf[k]
        line = (f"{k:13s} {r.mean():+8.2f}±{r.std():6.2f}    {s.mean():+8.2f}±{s.std():6.2f}   "
                f"[{r.min():+.2f},{r.max():+.2f}]  [{s.min():+.2f},{s.max():+.2f}]")
        if have_ks:
            p = ks_2samp(r, s).pvalue
            line += f"   {p:.2f}"
        print(line)
    if have_ks:
        print("  (KS p>0.05 => cannot reject 'same distribution' for that feature)")

    # Mahalanobis in-distribution: are REAL crossing-states typical draws of the GENERATED set?
    keys = ["flight_s", "cross_speed", "descent_deg", "bounce_x", "cross_z"]
    Rm = np.column_stack([rf[k] for k in keys])
    Sm = np.column_stack([sf[k] for k in keys])
    mu = Sm.mean(0); cov = np.cov(Sm, rowvar=False); inv = np.linalg.pinv(cov)
    def md2(X):
        d = X - mu
        return np.einsum("ij,jk,ik->i", d, inv, d)
    dr, ds = md2(Rm), md2(Sm)
    k = len(keys)
    print(f"\n  Mahalanobis^2 (dim={k}) under GENERATED dist: REAL median={np.median(dr):.1f}  GEN median={np.median(ds):.1f}")
    print(f"  REAL within GEN 95% ellipsoid (chi2_{k},0.95={_chi2_95(k):.1f}): {(dr <= _chi2_95(k)).mean()*100:.0f}%")
    print("  (high % => real serves look like typical generator samples, not out-of-distribution)")

    _plot(R, S, args.out)


def _chi2_95(k):
    return {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 6: 12.59}.get(k, 11.07)


def _plot(R, S, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping figure: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    def draw(serves, color, label, nmax=40, reach_gate=False):
        first = True
        drawn = 0
        for s in serves:
            if drawn >= nmax:
                break
            w = _window(s)
            if w is None:
                continue
            i0, ie = w
            seg = s.kf[i0:ie]
            if reach_gate:
                f = feats(s)
                if f is None or not _in_reach_box(f):
                    continue
            ax[0].plot(seg[:, 0], seg[:, 2], color=color, alpha=0.35, lw=0.8, label=label if first else None)
            ax[1].plot(seg[:, 0], seg[:, 1], color=color, alpha=0.35, lw=0.8, label=label if first else None)
            first = False
            drawn += 1
    draw(R, "tab:blue", "REAL", reach_gate=True)
    draw(S, "tab:orange", "GENERATED", reach_gate=True)
    for a, yl, tl in ((ax[0], "z (m)", "side view: height vs x"), (ax[1], "y (m)", "top view: lateral vs x")):
        a.axvline(PLANE_X, color="k", ls="--", lw=0.8); a.set_xlabel("x (m)"); a.set_ylabel(yl)
        a.set_title(tl); a.legend(); a.grid(alpha=0.3)
    fig.suptitle("Generated (orange) vs Real (blue) ball trajectories — KF tracks, lock-on→hit plane")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\n  overlay figure saved -> {out}")


if __name__ == "__main__":
    main()
