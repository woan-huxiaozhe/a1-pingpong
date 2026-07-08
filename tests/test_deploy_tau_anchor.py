import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "deploy"))
from hittrack.tau_anchor import TauAnchor

def test_no_anchor_initially():
    a = TauAnchor()
    assert not a.has_anchor()

def test_extrapolates_countdown():
    a = TauAnchor()
    a.update(0.1, 1.0, -4.0, 0.0, -0.5, pred_t=0.5, recv_time=100.0)
    assert a.has_anchor()
    assert abs(a.tau_live(100.0) - 0.5) < 1e-9      # 刚收到
    assert abs(a.tau_live(100.2) - 0.3) < 1e-9      # 0.2s 后
    assert a.tau_live(100.7) < 0.0                  # 过击球面后为负
    assert a.ball_state() == (0.1, 1.0, -4.0, 0.0, -0.5)

def test_reset_clears():
    a = TauAnchor(); a.update(0,0,0,0,0, 0.3, 10.0); a.reset()
    assert not a.has_anchor()
