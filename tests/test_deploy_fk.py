import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "deploy"))
from hittrack.fk import racket_pose_world

_FX = os.path.join(os.path.dirname(__file__), "fixtures", "deploy_fk_fixture.json")

def test_fk_matches_sim_fixture():
    cases = json.load(open(_FX))
    for c in cases:
        q = torch.tensor(c["q"], dtype=torch.float64)
        center, normal = racket_pose_world(q)
        assert torch.allclose(center.double(), torch.tensor(c["blade_center"], dtype=torch.float64), atol=1e-3), \
            (center.tolist(), c["blade_center"])
        assert torch.allclose(normal.double(), torch.tensor(c["normal"], dtype=torch.float64), atol=1e-3), \
            (normal.tolist(), c["normal"])

def test_fk_batched():
    cases = json.load(open(_FX))
    q = torch.tensor([c["q"] for c in cases], dtype=torch.float64)
    center, normal = racket_pose_world(q)
    assert center.shape == (len(cases), 3) and normal.shape == (len(cases), 3)
