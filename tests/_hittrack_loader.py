"""Isaac-free loader for the HitTrack pure-torch MDP modules.

Importing these modules through the normal package path executes the package ``mdp/__init__.py``,
which runs ``from isaaclab.envs.mdp import *`` and transitively pulls in USD (``pxr``) -- unavailable
outside the Isaac Sim runtime. This loader registers lightweight stub parent packages in
``sys.modules`` (so the real ``mdp/__init__`` is never executed) and points each stub's ``__path__``
at the matching on-disk directory, so the pure kernels -- and their intra-package imports -- resolve
straight from their files with only torch installed.

Two task packages are stubbed:
  * ``a1_pingpong_hittrack.mdp`` -- the HitTrack kernels live here (``tracking``, ``reference_source``,
    ``reference_planner``, ``reference_commands``). This is the DEFAULT package for ``load_pure``.
  * ``table_tennis_sac.mdp`` -- the shared hit-physics kernel ``hitting`` (which ``reference_planner``
    imports) still lives here, and other test files import real siblings like ``table_tennis_sac.sac``.

Mirrors the file-path bypass already used by ``tests/test_ideal_racket_velocity.py`` and the
Isaac-free convention documented in ``tests/test_sac_table_tennis_pipeline.py``.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

_TASKS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "source",
        "unitree_rl_lab",
        "unitree_rl_lab",
        "tasks",
    )
)

# ``load_pure`` defaults to the HitTrack package; ``hitting`` still lives in table_tennis_sac.
_DEFAULT_PKG = "a1_pingpong_hittrack"
_STUB_PKGS = ("a1_pingpong_hittrack", "table_tennis_sac")


def _stub(name: str, path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        # Point each stub at its real on-disk dir so submodule imports resolve from disk WITHOUT
        # executing the isaaclab-importing package __init__. Collection-order-independent.
        mod.__path__ = [path]
        sys.modules[name] = mod


def _ensure_stub_packages() -> None:
    _stub("unitree_rl_lab", os.path.dirname(_TASKS_DIR))
    _stub("unitree_rl_lab.tasks", _TASKS_DIR)
    for pkg in _STUB_PKGS:
        _stub(f"unitree_rl_lab.tasks.{pkg}", os.path.join(_TASKS_DIR, pkg))
        _stub(f"unitree_rl_lab.tasks.{pkg}.mdp", os.path.join(_TASKS_DIR, pkg, "mdp"))


def load_pure(modname: str, pkg: str = _DEFAULT_PKG):
    """Load ``<pkg>.mdp.<modname>`` from file, bypassing the package __init__.

    ``pkg`` defaults to the HitTrack package (``a1_pingpong_hittrack``); pass
    ``pkg="table_tennis_sac"`` for the shared ``hitting`` kernel that still lives there.
    """
    _ensure_stub_packages()
    return importlib.import_module(f"unitree_rl_lab.tasks.{pkg}.mdp.{modname}")
