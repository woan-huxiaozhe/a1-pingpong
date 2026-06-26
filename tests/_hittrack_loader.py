"""Isaac-free loader for the ``table_tennis_sac.mdp`` pure-torch modules.

Importing these modules through the normal package path executes
``table_tennis_sac/mdp/__init__.py``, which runs ``from isaaclab.envs.mdp import *`` and
transitively pulls in USD (``pxr``) -- unavailable outside the Isaac Sim runtime. This loader
registers lightweight stub parent packages in ``sys.modules`` (so the real ``mdp/__init__`` is
never executed) and points the ``mdp`` stub's ``__path__`` at the on-disk mdp directory, so the
pure kernels -- and their intra-package imports (e.g. ``reference_planner`` -> ``hitting``) --
resolve straight from their files with only torch installed.

Mirrors the file-path bypass already used by ``tests/test_ideal_racket_velocity.py`` and the
Isaac-free convention documented in ``tests/test_sac_table_tennis_pipeline.py``.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

_MDP_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "source",
        "unitree_rl_lab",
        "unitree_rl_lab",
        "tasks",
        "table_tennis_sac",
        "mdp",
    )
)

_PKG = "unitree_rl_lab.tasks.table_tennis_sac.mdp"


def _ensure_stub_packages() -> None:
    parts = _PKG.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            mod = types.ModuleType(name)
            # Point each stub at its real on-disk directory so intra-package submodule imports
            # (mdp kernels here, but also real siblings like ``table_tennis_sac.sac`` that other
            # test files import) resolve from disk -- WITHOUT executing the isaaclab-importing
            # package __init__. This keeps the loader collection-order-independent.
            d = _MDP_DIR
            for _ in range(len(parts) - i):
                d = os.path.dirname(d)
            mod.__path__ = [d]
            sys.modules[name] = mod


def load_pure(modname: str):
    """Load ``table_tennis_sac.mdp.<modname>`` from file, bypassing the package __init__."""
    _ensure_stub_packages()
    return importlib.import_module(f"{_PKG}.{modname}")
