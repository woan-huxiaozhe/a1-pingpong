"""Generate 3 lateral backhand reference variants for per-serve matching (plan: multi-ref).

The single static backhand reference cannot lie on the ball's path across the calibrated
serve spread (y in [-0.17, +0.24]), so the policy never contacts the ball. This mirrors the
working forehand design (x1/forehand: 3 refs + match_ball_direction=True): produce 3 copies
of the canonical swing shifted laterally so that, for each serve, the matched reference
passes THROUGH the incoming ball.

Lateral placement is a pure `joint_yb_2` (column 1, shoulder-roll) offset baked across all
frames -- the same mechanism as the prior +0.329 re-center. All other joints, base_y, fps,
hit_phase and ball_arrive_time_est are preserved (keeps the real-demo swing dynamics intact).

Order matters: UpperBodyMotionCommand._assign_motion_by_ball expects
  motion_ids 0=middle, 1=left (ball +y), 2=right (ball -y),
so env_cfg must list motion_files = [middle, left, right]. +yb_2 -> paddle +y (gain ~0.41
m/rad), so left uses a positive offset and right a negative one.

Usage:
  python create_backhand_variants.py                 # default offsets
  python create_backhand_variants.py --left 0.35 --right -0.27   # calibrate (Phase B)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(DATA_DIR, "backhand_ref.npz")  # current re-centered ref = middle

YB2_COL = 1                       # column index of joint_yb_2 in upper_body_dof
YB2_LIMIT = (-3.14, 0.26)         # A1 joint_yb_2 (shoulder_roll) limits

# Default lateral offsets (rad), starting guesses from gain ~0.41 m/rad and bucket centers
# left ~ +0.145 m, right ~ -0.11 m. Calibrated in Phase B via play_pure_ref.py.
DEFAULT_LEFT = 0.35
DEFAULT_RIGHT = -0.27


def write_variant(base: dict, name: str, yb2_offset: float) -> None:
    dof = base["upper_body_dof"].copy()
    dof[:, YB2_COL] = dof[:, YB2_COL] + yb2_offset

    vmin, vmax = float(dof[:, YB2_COL].min()), float(dof[:, YB2_COL].max())
    lo, hi = YB2_LIMIT
    status = "OK" if (vmin >= lo - 1e-4 and vmax <= hi + 1e-4) else "!! OUT OF LIMIT !!"

    out = os.path.join(DATA_DIR, f"backhand_ref_{name}.npz")
    save_kwargs = dict(
        fps=base["fps"],
        upper_body_dof=dof.astype(np.float32),
        base_y=base["base_y"],
        joint_names=base["joint_names"],
    )
    # preserve the offline-computed phase metadata if present
    for k in ("hit_phase", "ball_arrive_time_est"):
        if k in base:
            save_kwargs[k] = base[k]
    np.savez(out, **save_kwargs)
    print(f"[{name:6s}] yb_2 offset {yb2_offset:+.3f}  -> range [{vmin:+.3f}, {vmax:+.3f}]  "
          f"vs limit [{lo:+.2f}, {hi:+.2f}]  {status}")
    print(f"          wrote {out}  ({dof.shape[0]} frames x {dof.shape[1]} dof)")


def main(left_off: float, right_off: float) -> None:
    assert os.path.isfile(BASE_PATH), f"base ref not found: {BASE_PATH}"
    data = np.load(BASE_PATH, allow_pickle=True)
    base = {k: data[k] for k in data.files}
    print(f"[base] {BASE_PATH}  dof {base['upper_body_dof'].shape}  fps {float(base['fps'])}")
    if "hit_phase" in base:
        print(f"[base] hit_phase={float(base['hit_phase']):.4f}  "
              f"ball_arrive_time_est={float(base['ball_arrive_time_est']):.3f}")

    write_variant(base, "middle", 0.0)
    write_variant(base, "left", left_off)    # motion_id 1: ball predicted +y
    write_variant(base, "right", right_off)  # motion_id 2: ball predicted -y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=float, default=DEFAULT_LEFT, help="yb_2 offset for left variant (+y)")
    ap.add_argument("--right", type=float, default=DEFAULT_RIGHT, help="yb_2 offset for right variant (-y)")
    args = ap.parse_args()
    main(args.left, args.right)
