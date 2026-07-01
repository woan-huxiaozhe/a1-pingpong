#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize a ``play_hittrack.py`` record CSV (the per-step / per-env snapshot written by the
``--record`` flag). Produces two figures for ONE (env, episode):

  * Figure 1 -- joint tracking: per arm joint, (a) actual joint pos vs the policy TARGET pos
    (with the joint's hard position limits drawn as dashed lines; limits far outside the trace
    are annotated at the frame edge instead of squashing the plot), (b) actual joint velocity,
    (c) actual applied torque.
  * Figure 2 -- end-effector: blade-center position / velocity / face-normal (actual) overlaid
    with the model-derived TARGET reference command (``pref`` / ``vref`` / ``nref``, the noisy
    "estimate hit command" the actor sees; the clean privileged reference is drawn faint).

A dotted vertical line marks the hit instant (first ``hit_done`` step, else ``min|tau_true|``).

This script has NO Isaac dependency -- it only needs ``numpy`` + ``matplotlib``. By default it
renders with the headless ``Agg`` backend and saves PNGs next to the CSV; pass ``--show`` to open
an interactive window instead.

Usage:
    # latest record under logs/, env 0, first episode that contains a hit
    python scripts/rsl_rl/plot_hittrack_record.py

    # a specific CSV + episode, save PNGs into a chosen dir
    python scripts/rsl_rl/plot_hittrack_record.py path/to/record.csv --env 0 --ep 3 --outdir /tmp/plots

    # just list the (env, ep) pairs available in a CSV (with hit / success flags)
    python scripts/rsl_rl/plot_hittrack_record.py path/to/record.csv --list
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_AXES = ("x", "y", "z")

# The joint position limits the sim (and the ``joint_limit_margin_penalty`` reward) enforce come
# from the A1 USD, which is generated from this URDF; with ``soft_joint_pos_limit_factor=1.0`` the
# soft limits equal these hard <limit> values. We read them straight from the URDF (stdlib XML, no
# Isaac dependency, no stale hard-coded numbers). Path is resolved relative to this script.
_DEFAULT_URDF = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir,
    "source", "unitree_rl_lab", "data", "robots", "a1", "a1.urdf",
))


def _find_latest_csv() -> str | None:
    """Newest ``*play_record*`` / ``hittrack_play_records`` CSV under ``logs/``, if any."""
    cands = set(glob.glob("logs/**/*play_record*.csv", recursive=True))
    cands |= set(glob.glob("logs/**/hittrack_play_records/*.csv", recursive=True))
    paths = sorted(cands, key=os.path.getmtime)
    return paths[-1] if paths else None


def _joint_names(field_names) -> list[str]:
    """Recover the arm joint names (action-term order) from the ``q_tgt_<name>`` columns."""
    pre = "q_tgt_"
    return [c[len(pre):] for c in field_names if c.startswith(pre)]


def _select_episode(data, env: int, ep_arg: int | None) -> int:
    eps = np.unique(data["ep"][data["env"] == env].astype(int))
    if eps.size == 0:
        raise SystemExit(f"[PLOT] no rows for env={env}. Try --list to see what's available.")
    if ep_arg is not None:
        if ep_arg not in eps:
            raise SystemExit(f"[PLOT] env={env} has no ep={ep_arg}. Available: {eps.tolist()}")
        return ep_arg
    # default: first episode that actually contains a latched hit, else the first one
    for p in eps:
        m = (data["env"] == env) & (data["ep"] == p)
        if (data["hit_done"][m] > 0.5).any():
            return int(p)
    return int(eps[0])


def _hit_index(rows) -> int:
    """Row index of the hit instant: first ``hit_done`` step, else closest ``tau_true`` to 0."""
    hd = rows["hit_done"] > 0.5
    if hd.any():
        return int(np.argmax(hd))
    return int(np.argmin(np.abs(rows["tau_true"])))


def _urdf_joint_limits(joints, urdf_path):
    """Map ``joint_name -> (lower, upper)`` position limits from a URDF ``<limit>`` tag.

    Joints missing from the URDF (or a URDF that cannot be read at all) are silently skipped --
    the limit lines are optional decoration, so a missing file just means we draw none.
    """
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"[PLOT] no joint limits (could not read URDF '{urdf_path}': {exc})", file=sys.stderr)
        return {}
    out = {}
    for jn in joints:
        jel = root.find(f"./joint[@name='{jn}']")
        lim = jel.find("limit") if jel is not None else None
        if lim is None:
            continue
        lo, hi = lim.get("lower"), lim.get("upper")
        if lo is not None and hi is not None:
            out[jn] = (float(lo), float(hi))
    return out


def _draw_limit_lines(ax, lo, hi, label=False):
    """Overlay lower/upper joint-limit lines on a position axis without wrecking its scale.

    The axis stays driven by the actual/target traces; we only extend it toward a limit that is
    within half a data-span of the trace, so a joint pinned against its stop is obvious while a
    joint with lots of headroom keeps its tracking detail. A limit that stays far off-view is
    annotated at the frame edge (with its value) instead of squashing the plot.
    """
    ax.relim()
    ax.autoscale_view()
    d0, d1 = ax.get_ylim()
    span = max(d1 - d0, 1.0e-6)
    view_lo = min(d0, max(lo, d0 - 0.5 * span))
    view_hi = max(d1, min(hi, d1 + 0.5 * span))
    pad = 0.05 * max(view_hi - view_lo, 1.0e-6)
    ax.set_ylim(view_lo - pad, view_hi + pad)
    ax.axhline(lo, color="0.45", ls="--", lw=1.0, alpha=0.8, label="limit" if label else "_nolegend_")
    ax.axhline(hi, color="0.45", ls="--", lw=1.0, alpha=0.8, label="_nolegend_")
    vlo, vhi = ax.get_ylim()
    if lo < vlo:
        ax.text(0.99, 0.02, f"↓ limit {lo:+.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="0.45")
    if hi > vhi:
        ax.text(0.99, 0.98, f"↑ limit {hi:+.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="0.45")


def _plot_joints(plt, t, rows, joints, limits, hit_t, title):
    n = len(joints)
    fig, axes = plt.subplots(n, 3, figsize=(13, 1.9 * n), sharex=True, squeeze=False)
    axes[0, 0].set_title("joint pos: actual vs target [rad]")
    axes[0, 1].set_title("joint velocity [rad/s]")
    axes[0, 2].set_title("applied torque [N·m]")
    for j, jn in enumerate(joints):
        short = jn.replace("joint_", "")
        ax_p, ax_v, ax_t = axes[j]
        ax_p.plot(t, rows[f"q_{jn}"], color="C0", lw=1.3, label="actual")
        ax_p.plot(t, rows[f"q_tgt_{jn}"], color="C3", ls="--", lw=1.2, label="target")
        lim = limits.get(jn)
        if lim is not None:
            _draw_limit_lines(ax_p, lim[0], lim[1], label=(j == 0))
        ax_v.plot(t, rows[f"qd_{jn}"], color="C2", lw=1.2)
        ax_t.plot(t, rows[f"torque_{jn}"], color="C4", lw=1.2)
        for ax in (ax_p, ax_v, ax_t):
            ax.axvline(hit_t, color="k", ls=":", lw=1.0, alpha=0.6)
            ax.grid(True, alpha=0.25)
        ax_p.set_ylabel(short, fontsize=9)
    axes[0, 0].legend(loc="best", fontsize=8)
    for c in range(3):
        axes[-1, c].set_xlabel("time [s]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def _plot_end_effector(plt, t, rows, hit_t, title):
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True, squeeze=False)
    # (row label, actual-prefix, target-prefix, clean-prefix-or-None)
    spec = [
        ("EE position [m]", "ee_", "pref_", "pref_c"),
        ("EE velocity [m/s]", "ee_v", "vref_", "vref_c"),
        ("face normal [unit]", "rn_", "nref_", None),
    ]
    for r, (ylabel, act_pre, tgt_pre, clean_pre) in enumerate(spec):
        for c, a in enumerate(_AXES):
            ax = axes[r][c]
            ax.plot(t, rows[f"{act_pre}{a}"], color="C0", lw=1.4, label="actual")
            ax.plot(t, rows[f"{tgt_pre}{a}"], color="C3", ls="--", lw=1.3, label="target (est.)")
            if clean_pre is not None:
                ax.plot(t, rows[f"{clean_pre}{a}"], color="C7", ls=":", lw=1.0, alpha=0.8,
                        label="target (clean)")
            ax.axvline(hit_t, color="k", ls=":", lw=1.0, alpha=0.6)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(a)
        axes[r][0].set_ylabel(ylabel, fontsize=10)
    for c in range(3):
        axes[-1][c].set_xlabel("time [s]")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot a play_hittrack record CSV (joints + end-effector).")
    ap.add_argument("csv", nargs="?", default=None, help="record CSV (default: newest under logs/).")
    ap.add_argument("--env", type=int, default=0, help="which env id to plot (default 0).")
    ap.add_argument("--ep", type=int, default=None,
                    help="which per-env episode index (default: first episode containing a hit).")
    ap.add_argument("--dt", type=float, default=0.01, help="control step seconds for the time axis (100 Hz).")
    ap.add_argument("--outdir", type=str, default=None, help="where to save PNGs (default: next to the CSV).")
    ap.add_argument("--show", action="store_true", help="open interactive windows instead of saving PNGs.")
    ap.add_argument("--list", action="store_true", help="list (env, ep) pairs in the CSV and exit.")
    ap.add_argument("--urdf", type=str, default=None,
                    help="URDF to read arm joint position limits from (default: repo A1 URDF). "
                         "Limit lines are skipped if it cannot be read.")
    ap.add_argument("--no-limits", dest="no_limits", action="store_true",
                    help="do not draw the joint position-limit lines on the joint-pos plots.")
    args = ap.parse_args()

    csv = args.csv or _find_latest_csv()
    if not csv or not os.path.exists(csv):
        raise SystemExit("[PLOT] no CSV found. Pass a path, or run play_hittrack.py --record first.")

    data = np.atleast_1d(np.genfromtxt(csv, delimiter=",", names=True))
    if data.dtype.names is None or data.size == 0:
        raise SystemExit(f"[PLOT] '{csv}' has no parseable rows.")
    joints = _joint_names(data.dtype.names)
    if not joints:
        raise SystemExit(f"[PLOT] '{csv}' has no q_tgt_* columns -- is this a play_hittrack record?")

    if args.list:
        print(f"[PLOT] {csv}")
        for e in np.unique(data["env"].astype(int)):
            for p in np.unique(data["ep"][data["env"] == e].astype(int)):
                m = (data["env"] == e) & (data["ep"] == p)
                hit = bool((data["hit_done"][m] > 0.5).any())
                succ = bool((data["success"][m] > 0.5).any())
                print(f"  env={e:>3} ep={p:>3} steps={int(m.sum()):>4} hit={'Y' if hit else 'N'} "
                      f"success={'Y' if succ else 'N'}")
        return

    env = args.env
    ep = _select_episode(data, env, args.ep)
    m = (data["env"] == env) & (data["ep"] == ep)
    rows = data[m]
    rows = rows[np.argsort(rows["t"])]
    t = rows["t"] * args.dt

    hi = _hit_index(rows)
    hit_t = float(t[hi])
    pe, ve = float(rows["perr"][hi]), float(rows["verr"][hi])
    succ = bool((rows["success"] > 0.5).any())
    tag = (f"env={env} ep={ep} | {rows.size} steps | hit@{hit_t:.2f}s "
           f"pos_err={pe:.4f} m vel_err={ve:.3f} m/s success={'Y' if succ else 'N'}")
    print(f"[PLOT] {csv}\n[PLOT] {tag}")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limits = {} if args.no_limits else _urdf_joint_limits(joints, args.urdf or _DEFAULT_URDF)
    fig1 = _plot_joints(plt, t, rows, joints, limits, hit_t, f"HitTrack joints -- {tag}")
    fig2 = _plot_end_effector(plt, t, rows, hit_t, f"HitTrack end-effector -- {tag}")

    if args.show:
        plt.show()
    else:
        outdir = args.outdir or os.path.dirname(os.path.abspath(csv))
        os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv))[0]
        p1 = os.path.join(outdir, f"{base}_env{env}_ep{ep}_joints.png")
        p2 = os.path.join(outdir, f"{base}_env{env}_ep{ep}_endeffector.png")
        fig1.savefig(p1, dpi=130)
        fig2.savefig(p2, dpi=130)
        print(f"[PLOT] saved:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
