"""MDP terms for the A1-Pingpong-HitTrack task.

HitTrack OWNS the model-derived hit-reference tracking kernels and the env-facing reference
manager (the pure-torch modules below) plus its own tracking observations / rewards / termination.
It REUSES the generic and Catch-shared MDP terms -- the joint-delta action, racket-state
observations, joint-smoothing reward regularizers, the ready-pose reset and the SAC time-out / NaN
terminations -- from the ``table_tennis_sac`` package by re-export, so the HitTrack env cfg can
address everything as ``mdp.*`` without duplicating that infrastructure. ``table_tennis_sac.mdp``
no longer defines any HitTrack term.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Explicit re-export of the built-in class-based DR term (the wildcard above already pulls it in,
# but naming it makes it visible to static tooling and documents the sim-to-real PD randomization).
from isaaclab.envs.mdp.events import randomize_actuator_gains  # noqa: F401

# Reuse the generic / Catch-shared terms (racket_pos/vel/ang_vel/normal/axes, joint_pos_delta_history,
# JointDeltaTargetActionCfg, joint_acc/jerk/limit/effort regularizers, reset_robot_to_ready_pose,
# sac_time_out, joint_state_nan, ...). Private helpers (``_racket_body_state``) are imported
# explicitly where needed; ``import *`` skips them.
from unitree_rl_lab.tasks.table_tennis_sac.mdp import *  # noqa: F401, F403

# HitTrack-specific modules: pure-torch kernels, the env-facing reference manager, and the
# tracking obs / reward / termination terms.
from .tracking import *  # noqa: F401, F403
from .reference_source import *  # noqa: F401, F403
from .reference_planner import *  # noqa: F401, F403
from .reference_commands import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
