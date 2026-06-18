"""MDP terms for the independent A1 table-tennis SAC task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .rewards import (  # noqa: F401  explicit re-export of the Ace three-tier terminal rewards
    sac_flat_return,
    sac_miss_approach,
    sac_racket_spin_penalty,
    sac_table_proximity,
)
from .terminations import *  # noqa: F401, F403
