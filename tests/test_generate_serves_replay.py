import json
import numpy as np

from unitree_rl_lab.tasks.a1_pingpong_hittrack.serve_gen.generate_serves import _load_replay_latents


def _write_fit(path):
    np.savez(
        path,
        model="dragbounce",
        serve_id=np.array([10, 20, 30], dtype=np.int64),
        pos0_mm=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        v0_mm_s=np.array([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
        drag=np.array([1.1e-4, 1.2e-4, 1.3e-4]),
        alpha_z=np.array([0.90, 0.91, 0.92]),
        alpha_xy=np.array([0.70, 0.71, 0.72]),
    )


def test_replay_latents_select_order_and_skip_missing(tmp_path):
    fit = tmp_path / "fit.npz"
    _write_fit(fit)
    out = _load_replay_latents(str(fit), [30, 10, 99])  # 99 absent -> skipped
    assert [r["serve_id"] for r in out] == [30, 10]
    assert out[0]["drag"] == 1.3e-4
    assert out[0]["alpha_z"] == 0.92
    np.testing.assert_allclose(out[1]["pos0_mm"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1]["v0_mm_s"], [-1.0, 0.0, 0.0])
