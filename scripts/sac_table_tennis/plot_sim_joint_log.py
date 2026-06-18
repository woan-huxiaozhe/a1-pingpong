"""Plot per-step SAC play joint logs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_LOG_DIR = Path("logs/sac_table_tennis/sim_logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAC play joint target/state/torque CSV logs.")
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=None,
        help="CSV log to plot. Defaults to the newest CSV in --log_dir.",
    )
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--env_id", type=int, default=0, help="Vector env id to plot.")
    parser.add_argument("--episode", type=int, default=None, help="Optional 1-based episode filter.")
    parser.add_argument(
        "--x_axis",
        choices=("time", "global_step", "episode_step"),
        default="time",
        help="Horizontal axis for the plot.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Display an interactive window after plotting.")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--max_points",
        type=int,
        default=8000,
        help="Downsample long logs to at most this many points per line. Use 0 to disable.",
    )
    return parser.parse_args()


def newest_csv(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No CSV logs found in {log_dir}")
    return files[0]


def joint_names_from_header(header: list[str]) -> list[str]:
    prefix = "q_target_"
    joints = [name[len(prefix) :] for name in header if name.startswith(prefix)]
    missing: list[str] = []
    for joint in joints:
        for required_prefix in ("q_sim_", "qd_sim_", "tau_sim_"):
            column = f"{required_prefix}{joint}"
            if column not in header:
                missing.append(column)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if not joints:
        raise ValueError("CSV has no q_target_* joint columns.")
    return joints


def load_rows(path: Path, env_id: int, episode: int | None) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        header = list(reader.fieldnames)
        rows = [
            row
            for row in reader
            if int(row["env_id"]) == env_id and (episode is None or int(row["episode"]) == episode)
        ]
    if not rows:
        episode_text = "all episodes" if episode is None else f"episode {episode}"
        raise ValueError(f"No rows found for env_id={env_id}, {episode_text} in {path}")
    return header, rows


def output_path_for(log_path: Path, env_id: int, episode: int | None) -> Path:
    parts = [log_path.stem, f"env{env_id}"]
    if episode is not None:
        parts.append(f"episode{episode}")
    return log_path.with_name("__".join(parts) + "__joints.png")


def column_values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def thin_indices(length: int, max_points: int) -> range:
    if max_points <= 0 or length <= max_points:
        return range(length)
    stride = max(1, math.ceil(length / max_points))
    return range(0, length, stride)


def main() -> None:
    args = parse_args()
    log_path = args.log if args.log is not None else newest_csv(args.log_dir)
    header, rows = load_rows(log_path, args.env_id, args.episode)
    joints = joint_names_from_header(header)

    if args.x_axis == "time":
        x_label = "time [s]"
        x_values = column_values(rows, "time_s")
    else:
        x_label = args.x_axis
        x_values = column_values(rows, args.x_axis)

    indices = thin_indices(len(rows), args.max_points)
    x_values = [x_values[index] for index in indices]
    series: dict[str, list[float]] = {}
    for joint in joints:
        for prefix in ("q_target", "q_sim", "qd_sim", "tau_sim"):
            column = f"{prefix}_{joint}"
            values = column_values(rows, column)
            series[column] = [values[index] for index in indices]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.fontsize": 8,
            "font.size": 9,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(joints),
        ncols=3,
        sharex=True,
        figsize=(15.0, max(8.0, 1.8 * len(joints))),
        constrained_layout=True,
    )
    if len(joints) == 1:
        axes = [axes]

    title = f"{log_path.name} | env={args.env_id}"
    if args.episode is not None:
        title += f" | episode={args.episode}"
    fig.suptitle(title)

    for row_index, joint in enumerate(joints):
        pos_ax, vel_ax, tau_ax = axes[row_index]

        pos_ax.plot(x_values, series[f"q_target_{joint}"], label="target", linewidth=1.1)
        pos_ax.plot(x_values, series[f"q_sim_{joint}"], label="sim", linewidth=1.1)
        pos_ax.set_ylabel(joint)
        if row_index == 0:
            pos_ax.set_title("target vs sim joint pos [rad]")
            pos_ax.legend(loc="best")

        vel_ax.plot(x_values, series[f"qd_sim_{joint}"], color="tab:green", linewidth=1.1)
        if row_index == 0:
            vel_ax.set_title("sim joint velocity [rad/s]")

        tau_ax.plot(x_values, series[f"tau_sim_{joint}"], color="tab:red", linewidth=1.1)
        if row_index == 0:
            tau_ax.set_title("sim torque [Nm]")

        if row_index == len(joints) - 1:
            pos_ax.set_xlabel(x_label)
            vel_ax.set_xlabel(x_label)
            tau_ax.set_xlabel(x_label)

    output_path = args.output if args.output is not None else output_path_for(log_path, args.env_id, args.episode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    print(f"[PLOT] log={log_path}")
    print(f"[PLOT] rows={len(rows)} plotted={len(x_values)} env_id={args.env_id} episode={args.episode}")
    print(f"[PLOT] saved={output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
