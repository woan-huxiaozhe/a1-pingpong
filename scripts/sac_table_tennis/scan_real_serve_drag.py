"""Scan IsaacSim ball damping/drag against real serve trajectories.

This launches ``validate_real_serve_isaac.py`` once per parameter pair.  Each
child process creates a fresh IsaacSim app, which is slower than an in-process
loop but avoids Kit shutdown/state reuse issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "sac_table_tennis" / "validate_real_serve_isaac.py"
ISAAC_PYTHON = Path("/data/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh")
ISAAC_PYTHONPATH = ":".join(
    [
        "/data/miniforge3/envs/isaac/lib/python3.11/site-packages",
        "/data/IsaacLab/source/isaaclab",
        "/data/IsaacLab/source/isaaclab_assets",
        "/data/IsaacLab/source/isaaclab_tasks",
        str(REPO_ROOT / "source" / "unitree_rl_lab"),
    ]
)


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--velocity-source", choices=("file", "fit"), default="fit")
    parser.add_argument("--linear-damping-values", type=parse_float_list, default="0,0.05,0.10,0.20,0.35")
    parser.add_argument("--drag-k-values", type=parse_float_list, default="0,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "logs" / "sac_table_tennis" / "real_serve_drag_scan"))
    return parser.parse_args()


def score(row: dict) -> float:
    total = max(float(row.get("total", 1)), 1.0)
    missed = float(row.get("missed", total))
    if row.get("crossed", 0) == 0:
        return float("inf")
    return (
        4.0 * missed / total
        + abs(float(row.get("dtau_mean", 0.0))) / 0.08
        + abs(float(row.get("dz_mean", 0.0))) / 0.08
        + abs(float(row.get("dvx_mean", 0.0))) / 0.8
        + float(row.get("yz_norm_p50", 0.0)) / 0.10
    )


def run_one(args: argparse.Namespace, out_dir: Path, linear_damping: float, drag_k: float) -> dict:
    name = f"damp_{linear_damping:.3f}_drag_{drag_k:.3f}".replace(".", "p")
    log_path = out_dir / f"{name}.log"
    cmd = [
        str(ISAAC_PYTHON),
        str(VALIDATE_SCRIPT),
        "--device",
        args.device,
        "--velocity-source",
        args.velocity_source,
        "--linear-damping",
        str(linear_damping),
        "--drag-k",
        str(drag_k),
        "--quiet-progress",
        "--skip-app-close",
    ]
    if args.max_trajectories > 0:
        cmd.extend(["--max-trajectories", str(args.max_trajectories)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-codex"
    env["PYTHONPATH"] = ISAAC_PYTHONPATH

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")

    summary = None
    for line in proc.stdout.splitlines():
        if line.startswith("[summary] "):
            summary = json.loads(line[len("[summary] ") :])
    if summary is None:
        summary = {
            "linear_damping": linear_damping,
            "drag_k": drag_k,
            "velocity_source": args.velocity_source,
            "returncode": proc.returncode,
            "error": "missing summary",
        }
    summary["returncode"] = proc.returncode
    try:
        summary["log"] = str(log_path.relative_to(REPO_ROOT))
    except ValueError:
        summary["log"] = str(log_path)
    summary["score"] = score(summary)
    return summary


def print_row(row: dict) -> None:
    print(
        f"{row.get('linear_damping', 0):>6.3f} "
        f"{row.get('drag_k', 0):>6.3f} "
        f"{int(row.get('crossed', 0)):>3d}/{int(row.get('total', 0)):<3d} "
        f"dz={float(row.get('dz_mean', float('nan'))):+7.4f} "
        f"dvx={float(row.get('dvx_mean', float('nan'))):+7.4f} "
        f"dt={float(row.get('dtau_mean', float('nan'))):+7.4f} "
        f"yz50={float(row.get('yz_norm_p50', float('nan'))):6.4f} "
        f"score={float(row.get('score', float('inf'))):6.3f} "
        f"{row.get('log', '')}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not ISAAC_PYTHON.exists():
        raise SystemExit(f"missing IsaacSim python: {ISAAC_PYTHON}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("  damp   drag crossed        dz       dvx        dt   yz50  score log", flush=True)
    for damping in args.linear_damping_values:
        for drag_k in args.drag_k_values:
            row = run_one(args, out_dir, damping, drag_k)
            rows.append(row)
            print_row(row)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: row["score"])
    print("\n[best]")
    for row in ranked[:8]:
        print_row(row)
    print(f"\n[save] {summary_path.relative_to(REPO_ROOT)}")
    print(f"[save] {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
