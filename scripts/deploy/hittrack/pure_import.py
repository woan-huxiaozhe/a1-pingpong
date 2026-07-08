"""Isaac-free 加载训练侧纯 torch 函数（plan_hit_reference / compute_joint_delta_target），
不触发 mdp/__init__ 的 isaaclab 导入。手法同 tests/_hittrack_loader.py：给父包注册轻量
stub，__path__ 指向真实目录，子模块从文件解析。单一物理来源，避免与训练代码漂移。"""
from __future__ import annotations
import importlib, os, sys, types

_TASKS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "source", "unitree_rl_lab", "unitree_rl_lab", "tasks"))

def _stub(name, path):
    if name not in sys.modules:
        m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m

def _ensure():
    _stub("unitree_rl_lab", os.path.dirname(_TASKS))
    _stub("unitree_rl_lab.tasks", _TASKS)
    for pkg in ("a1_pingpong_hittrack", "table_tennis_sac"):
        _stub(f"unitree_rl_lab.tasks.{pkg}", os.path.join(_TASKS, pkg))
        _stub(f"unitree_rl_lab.tasks.{pkg}.mdp", os.path.join(_TASKS, pkg, "mdp"))

def load_plan_hit_reference():
    _ensure()
    importlib.import_module("unitree_rl_lab.tasks.table_tennis_sac.mdp.hitting")  # 先解析 reference_planner 的依赖
    return importlib.import_module(
        "unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp.reference_planner").plan_hit_reference

def load_compute_joint_delta_target():
    _ensure()
    return importlib.import_module(
        "unitree_rl_lab.tasks.table_tennis_sac.control").compute_joint_delta_target
