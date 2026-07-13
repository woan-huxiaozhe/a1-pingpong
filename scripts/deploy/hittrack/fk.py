"""纯 torch 正向运动学：X1_URDF_V1_2 右臂 r1..r7 的 7 关节齐次链 + T_mount/T_paddle -> 世界系桨面位姿。
常数见 config.py（T_MOUNT 由新 a1.usd 仿真反解，FK-vs-sim 静态残差 0.48mm）。无 isaaclab / RBDL 依赖。"""
from __future__ import annotations
import torch
from . import config as C

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (torch.cos(torch.tensor(r)), torch.sin(torch.tensor(r)),
        torch.cos(torch.tensor(p)), torch.sin(torch.tensor(p)),
        torch.cos(torch.tensor(y)), torch.sin(torch.tensor(y)))
    Rz = torch.tensor([[cy,-sy,0.],[sy,cy,0.],[0.,0.,1.]])
    Ry = torch.tensor([[cp,0.,sp],[0.,1.,0.],[-sp,0.,cp]])
    Rx = torch.tensor([[1.,0.,0.],[0.,cr,-sr],[0.,sr,cr]])
    return Rz @ Ry @ Rx

def _homog(R, t):
    T = torch.eye(4, dtype=torch.float64); T[:3,:3] = R; T[:3,3] = torch.as_tensor(t, dtype=torch.float64); return T

# 预计算：每关节的固定 origin 变换 + T_mount + T_paddle（模块加载时算一次）
_T_ORIGIN = [_homog(_rpy(*j["rpy"]).double(), j["xyz"]) for j in C.URDF_JOINTS]
_T_MOUNT = _homog(torch.tensor(C.T_MOUNT_R, dtype=torch.float64), C.T_MOUNT_T)
_T_PADDLE = _homog(torch.tensor(C.T_PADDLE_R, dtype=torch.float64), C.T_PADDLE_T)
_OFFSET = torch.tensor([0., 0., C.RACKET_OFFSET_Z], dtype=torch.float64)

def _rotz(theta):  # theta: [...]
    c, s = torch.cos(theta), torch.sin(theta)
    z = torch.zeros_like(c); o = torch.ones_like(c)
    R = torch.stack([torch.stack([c,-s,z],-1), torch.stack([s,c,z],-1), torch.stack([z,z,o],-1)], -2)
    return R  # [...,3,3]

def racket_pose_world(q):
    """q: [7] 或 [N,7] -> (blade_center_world[...,3], normal_world[...,3])."""
    single = (q.dim() == 1)
    qq = q.reshape(1, 7) if single else q
    qq = qq.double()
    N = qq.shape[0]
    T = _T_MOUNT.unsqueeze(0).expand(N, 4, 4).clone()
    for i in range(7):
        Ti = _T_ORIGIN[i].unsqueeze(0).expand(N,4,4)
        Rj = _rotz(qq[:, i])                      # [N,3,3]
        Tj = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(N,1,1)
        Tj[:, :3, :3] = Rj
        T = T @ Ti @ Tj
    T = T @ _T_PADDLE.unsqueeze(0)
    R = T[:, :3, :3]; t = T[:, :3, 3]
    center = t + (R @ _OFFSET).reshape(N, 3)
    normal = R @ torch.tensor([0., 1., 0.], dtype=torch.float64)
    center = center.to(q.dtype); normal = normal.to(q.dtype)
    return (center[0], normal[0]) if single else (center, normal)
