"""Probe A1 SAC ready-pose paddle height and finite-difference FK sensitivities."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe SAC ready-pose paddle height.")
parser.add_argument(
    "--pose",
    type=float,
    nargs=7,
    default=[1.533406, -0.523925, 1.60474, -1.183103, -0.007649, 1.042375, -1.845288],
)
parser.add_argument("--lift", type=float, default=-0.28)
parser.add_argument("--target_z", type=float, default=1.16)
parser.add_argument("--delta", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

sys.path.insert(0, "/data/PPO-pingpong/source/unitree_rl_lab")
from unitree_rl_lab.assets.robots.a1 import A1_TABLE_TENNIS_CFG  # noqa: E402

ARM = ["joint_yb_1", "joint_yb_2", "joint_yb_3", "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7"]
PADDLE = "Link_yb_paddle"
DEVICE = "cuda:0"


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=DEVICE))
    cfg = A1_TABLE_TENNIS_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.init_state.pos = (-1.7, 0.0, 0.0)
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot = Articulation(cfg)

    sim.reset()
    robot.reset()
    robot.update(dt=0.005)

    arm_ids = robot.find_joints(ARM)[0]
    lift_id = robot.find_joints(["joint_lift"])[0][0]
    paddle_idx = robot.find_bodies(PADDLE)[0][0]
    lift_limits = robot.data.soft_joint_pos_limits[0, lift_id].detach().cpu().numpy()

    def read_paddle(qarm: torch.Tensor, lift: float = args.lift) -> np.ndarray:
        full = robot.data.default_joint_pos[0].clone()
        full[lift_id] = lift
        for i, joint_id in enumerate(arm_ids):
            full[joint_id] = qarm[i]
        pos = full.unsqueeze(0)
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        robot.set_joint_position_target(pos)
        for _ in range(80):
            sim.step()
            robot.update(dt=0.005)
        return robot.data.body_pos_w[0, paddle_idx].detach().cpu().numpy().copy()

    q0 = torch.tensor(args.pose, dtype=torch.float32, device=DEVICE)
    p0 = read_paddle(q0, args.lift)
    print(f"pose={np.round(np.array(args.pose), 6).tolist()}", flush=True)
    print(f"lift={args.lift:+.4f}", flush=True)
    print(f"lift_limits=[{lift_limits[0]:+.4f}, {lift_limits[1]:+.4f}]", flush=True)
    print(f"paddle_world=({p0[0]:+.4f}, {p0[1]:+.4f}, {p0[2]:+.4f})", flush=True)
    print(f"target_z={args.target_z:+.4f} dz_needed={args.target_z - p0[2]:+.4f}", flush=True)
    print("joint finite differences:", flush=True)

    jac = []
    for i, name in enumerate(ARM):
        qp = q0.clone()
        qp[i] += args.delta
        pp = read_paddle(qp, args.lift)
        d = (pp - p0) / args.delta
        jac.append(d)
        print(f"  {name}: dx={d[0]:+.4f} dy={d[1]:+.4f} dz={d[2]:+.4f} per rad", flush=True)

    p_lift = read_paddle(q0, args.lift + args.delta)
    d_lift = (p_lift - p0) / args.delta
    print(f"  joint_lift: dx={d_lift[0]:+.4f} dy={d_lift[1]:+.4f} dz={d_lift[2]:+.4f} per m", flush=True)
    if abs(d_lift[2]) > 1.0e-6:
        lift_new = args.lift + (args.target_z - p0[2]) / d_lift[2]
        lift_clamped = float(np.clip(lift_new, lift_limits[0], lift_limits[1]))
        print(f"candidate_lift={lift_new:+.4f}", flush=True)
        print(f"candidate_lift_clamped={lift_clamped:+.4f}", flush=True)

    jac = np.asarray(jac)
    z_jac = jac[:, 2]
    if np.max(np.abs(z_jac)) > 1.0e-6:
        best = int(np.argmax(np.abs(z_jac)))
        dq = (args.target_z - p0[2]) / z_jac[best]
        q_new = np.array(args.pose, dtype=np.float64)
        q_new[best] += dq
        print(f"best_z_joint={ARM[best]} dq={dq:+.4f}", flush=True)
        print(f"candidate_pose={np.round(q_new, 6).tolist()}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
