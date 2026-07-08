# tests/test_deploy_obs.py
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "deploy"))
from hittrack.obs import JointDeltaHistory, assemble_obs
from hittrack.tau_anchor import TauAnchor
from hittrack.pure_import import load_plan_hit_reference
from hittrack import config as C

def test_history_dims_and_reset():
    h = JointDeltaHistory(); q = torch.zeros(7)
    h.reset(q)
    assert h.deltas().shape == (35,)
    assert torch.allclose(h.deltas(), torch.zeros(35))          # 重置后全零
    h.push(q + 0.1)
    d = h.deltas()
    assert torch.allclose(d[:7], torch.full((7,), 0.1), atol=1e-6)  # 最新一帧 delta

def test_history_clip():
    h = JointDeltaHistory(); h.reset(torch.zeros(7)); h.push(torch.full((7,), 5.0))
    assert torch.all(h.deltas() <= 1.0) and torch.all(h.deltas() >= -1.0)

def test_obs_layout_68_and_error_consistency():
    plan = load_plan_hit_reference()
    q = torch.tensor(C.READY_JOINT_POS, dtype=torch.float32)
    ta = TauAnchor(); ta.update(0.05, 1.0, -4.0, 0.0, -0.5, pred_t=0.4, recv_time=0.0)
    h = JointDeltaHistory(); h.reset(q)
    last_action = torch.zeros(7)
    obs = assemble_obs(q, ta, tau_live=0.4, last_action=last_action, plan_fn=plan, hist=h)
    assert obs.shape == (68,)
    # 布局: joint_pos_rel[0:7] history[7:42] hit_reference_command[42:52]
    #       racket_pos[52:55] racket_normal[55:58] hit_ref_pos_error[58:61] last_action[61:68]
    # hit_reference_command = p_ref[42:45] v_ref[45:48] n_ref[48:51] tau[51]
    assert torch.allclose(obs[:7], q - torch.tensor(C.DEFAULT_JOINT_POS), atol=1e-5)  # joint_pos_rel
    p_ref = obs[42:45]; racket_pos = obs[52:55]; err = obs[58:61]
    assert torch.allclose(err, racket_pos - p_ref, atol=1e-5)                          # hit_ref_pos_error
    assert abs(float(obs[51]) - 0.4) < 1e-5                                            # tau 在 hit_reference_command 末位
    assert torch.allclose(obs[61:68], last_action)                                     # last_action 尾 7
