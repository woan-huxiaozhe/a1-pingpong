# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained ``A1-Pingpong-HitTrack`` rsl-rl PPO checkpoint to a self-contained,
normalization-baked ``torch.jit`` ``policy.pt`` for the isaaclab-free deploy node.

Why this script exists (see design doc §8 "策略导出缺口"):
  * The training ``model_*.pt`` is an rsl-rl runner checkpoint (state-dicts + optimizer + iter),
    NOT a callable — reconstructing it needs the rsl-rl / isaaclab stack, which the deploy node
    (system python, torch only) does not have.
  * ``play_hittrack.py`` loads such a checkpoint correctly (it calls ``handle_deprecated_rsl_rl_cfg``
    to migrate the legacy cfg; the generic ``play.py`` dies with ``KeyError: 'class_name'`` here).
  * The installed rsl-rl-lib (5.x) keeps observation normalization *inside* the actor ``MLPModel``
    (``obs_normalizer``), not as a separate object — so ``OnPolicyRunner.export_policy_to_jit`` /
    ``export_policy_to_onnx`` already trace a flat-tensor-in/flat-tensor-out module with the
    normalizer baked in. This script just loads the checkpoint (the ``play_hittrack.py`` way) and
    calls those built-in exporters, so the deploy node feeds RAW 68-dim obs and gets actions, with
    no normalizer bookkeeping and no rsl-rl dependency at runtime.

Run in the conda isaac env (python 3.11 + isaaclab + rsl-rl):
    source /home/woan/miniforge3/etc/profile.d/conda.sh 2>/dev/null || \
        source /data/miniforge3/etc/profile.d/conda.sh; conda activate /data/miniforge3/envs/isaac
    python scripts/rsl_rl/export_hittrack_policy.py --task A1-Pingpong-HitTrack --headless \
        --num_envs 1 --checkpoint logs/rsl_rl/a1_tabletennis_hittrack/<run>/model_xxxx.pt

Writes ``<checkpoint_dir>/exported/policy.pt`` (and ``policy.onnx``).
"""

from __future__ import annotations

import argparse
from importlib.metadata import version as _pkg_version

from isaaclab.app import AppLauncher  # noqa: E402

# local import (scripts/rsl_rl is on sys.path[0] when run as a script)
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export an A1-Pingpong-HitTrack PPO checkpoint to jit/onnx.")
parser.add_argument("--task", type=str, default="A1-Pingpong-HitTrack", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 is enough for export).")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
cli_args.add_rsl_rl_args(parser)          # provides --checkpoint / --load_run / --device / ...
AppLauncher.add_app_launcher_args(parser)  # provides --headless
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    # --- env (a minimal HitTrackPlayEnvCfg instance is needed only to build the runner) ---
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # --- load exactly like play_hittrack.py (migrate legacy cfg, else KeyError: 'class_name') ---
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[EXPORT] loading PPO checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    runner.alg.eval_mode()

    # --- export: OnPolicyRunner's own exporters already trace a flat-tensor MLP with the
    # obs_normalizer baked in (see rsl_rl.models.mlp_model._TorchMLPModel.forward). ---
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(export_dir, filename="policy.pt")
    runner.export_policy_to_onnx(export_dir, filename="policy.onnx")
    print(f"[EXPORT] wrote {export_dir}/policy.pt (obs=68 -> action=7, normalizer baked in) + policy.onnx")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
