"""TEMP: FK Jacobian probe (raw sim, no gym env) — measures dPaddle/dq for each
right-arm joint at the backhand hit pose. Based on diag_minimal.py scaffolding.

Used to compute the joint offset that re-centers the reference paddle +Y by a
target amount. Prints per-joint (dx,dy,dz)/rad and recommended offsets.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--target_dy", type=float, default=0.153)
parser.add_argument("--delta", type=float, default=0.12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, "/data/PPO-pingpong/source/unitree_rl_lab")
from unitree_rl_lab.assets.robots.a1 import A1_TABLE_TENNIS_CFG

ARM = ["joint_yb_1", "joint_yb_2", "joint_yb_3", "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7"]
Q_HIT = [1.636, -0.922, 1.666, -1.000, -0.067, 1.356, -1.741]  # backhand_ref hit frame
LIFT = -0.18
PADDLE = "Link_yb_paddle"
DEV = "cuda:0"

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=DEV))
cfg = A1_TABLE_TENNIS_CFG.copy()
cfg.prim_path = "/World/Robot"
robot = Articulation(cfg)
sim.reset()
robot.reset()
robot.update(dt=0.005)

arm_ids = robot.find_joints(ARM)[0]
lift_id = robot.find_joints(["joint_lift"])[0][0]
paddle_idx = robot.find_bodies(PADDLE)[0][0]
print(f"arm_ids={arm_ids} lift_id={lift_id} paddle_idx={paddle_idx}", flush=True)


def read_paddle(qarm):
    full = robot.data.joint_pos[0].clone()
    for k, jid in enumerate(arm_ids):
        full[jid] = qarm[k]
    full[lift_id] = LIFT
    pos = full.unsqueeze(0)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
    robot.set_joint_position_target(pos)
    for _ in range(20):
        sim.step()
        robot.update(dt=0.005)
    p = robot.data.body_pos_w[0, paddle_idx].clone()
    base = robot.data.root_pos_w[0].clone()
    return (p - base).cpu().numpy()  # paddle relative to robot base


qh = torch.tensor(Q_HIT, dtype=torch.float32, device=DEV)
base = read_paddle(qh)
print(f"\nbaseline paddle (rel base) x={base[0]:+.3f} y={base[1]:+.3f} z={base[2]:+.3f}", flush=True)
print("(pure-ref env-rel paddle y was ~ -0.106; base y=0 so should match)\n", flush=True)

print("  joint      dX/drad   dY/drad   dZ/drad   |dY|", flush=True)
J = np.zeros((7, 3))
for i in range(7):
    qp = qh.clone()
    qp[i] += args.delta
    d = (read_paddle(qp) - base) / args.delta
    J[i] = d
    print(f"  {ARM[i]:11s} {d[0]:+8.3f} {d[1]:+8.3f} {d[2]:+8.3f}   {abs(d[1]):.3f}", flush=True)

jbest = int(np.argmax(np.abs(J[:, 1])))
dq = args.target_dy / J[jbest, 1]
pred = J[jbest] * dq
print(f"\n[recommend] dominant-Y joint = {ARM[jbest]} (dY/drad={J[jbest,1]:+.3f})", flush=True)
print(f"   offset {ARM[jbest]} by {dq:+.3f} rad  -> predicted paddle delta "
      f"dx={pred[0]:+.3f} dy={pred[1]:+.3f} dz={pred[2]:+.3f}", flush=True)
Jy = J[:, 1]
dq_all = Jy * args.target_dy / (Jy @ Jy)
pred_all = J.T @ dq_all
print(f"[recommend] min-norm all-joint dq={np.round(dq_all,3)} -> "
      f"dx={pred_all[0]:+.3f} dy={pred_all[1]:+.3f} dz={pred_all[2]:+.3f}", flush=True)

simulation_app.close()
