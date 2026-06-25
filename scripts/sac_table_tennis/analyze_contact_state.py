"""Offline analysis of rollout_contact_state.py output. No Isaac needed.

Quantifies, per contact and split by episode outcome:
  - paddle FACE NORMAL: actual vs ideal (angle between, elevation above horizontal)
  - paddle VELOCITY: |v_paddle| actual (body & blade-center) vs ideal (~1.3 m/s), elevation
  - resulting BALL launch: actual (fwd,up) vs analytic ideal v_out, and the reflection
    predicted from the ACTUAL normal (closes the loop: does a wrong normal explain the lob?)
"""
from __future__ import annotations
import json, math, sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "logs/sac_table_tennis/contact_state_0624_best.json"
E = 0.75  # paddle restitution

d = json.load(open(PATH))
side = d["robot_side"]; recs = d["records"]
print(f"file={PATH}\ncompleted={d['completed']} counts={d['counts']} contacts={d['n_contacts']} "
      f"robot_side={side} opp_center_x={d['opp_center_x']:.3f} theta={d['neutral_theta_deg']:.0f}deg\n")

def v(x): return np.array(x, float)
def unit(a):
    n = np.linalg.norm(a); return a / n if n > 1e-9 else a
def ang_between(a, b):
    c = np.dot(unit(a), unit(b)); return math.degrees(math.acos(max(-1, min(1, c))))
def elev(a):  # elevation above horizontal (deg), +up
    h = math.hypot(a[0], a[1]); return math.degrees(math.atan2(a[2], h))
def fwd_comp(a):  # forward = toward opponent (+x when side=-1)
    return -side * a[0]

rows = []
for r in recs:
    n_a = v(r["n_actual"]); n_i = v(r["n_ideal"])
    vpb = v(r["v_paddle_body"]); vpc = v(r["v_paddle_center"]); vpi = v(r["v_paddle_ideal"])
    v_in = v(r["v_in"]); v_out_i = v(r["v_out_ideal"])
    # reflection predicted from the ACTUAL normal, using blade-center (contact) velocity:
    vp_dot = np.dot(v_in - vpc, n_a)
    v_out_pred = v_in - (1 + E) * vp_dot * n_a
    out = "valid_return" if "valid_return" in r["outcome"] else (
          "bad_hit" if "bad_hit" in r["outcome"] else "+".join(r["outcome"]) or "none")
    rows.append(dict(
        outcome=out,
        normal_gap_deg=ang_between(n_a, n_i),
        n_actual_elev=elev(n_a),
        n_ideal_elev=elev(n_i),
        vpaddle_body=np.linalg.norm(vpb),
        vpaddle_body_elev=elev(vpb),
        vpaddle_center=np.linalg.norm(vpc),
        vpaddle_ideal=np.linalg.norm(vpi),
        vpaddle_ideal_elev=elev(vpi),
        ball_out_speed=math.hypot(r["actual_out_vx_fwd"], r["actual_out_vz_up"]),
        ball_out_angle=math.degrees(math.atan2(r["actual_out_vz_up"], r["actual_out_vx_fwd"])),
        ideal_out_speed=np.linalg.norm(v_out_i),
        ideal_out_angle=elev(v_out_i) if fwd_comp(v_out_i) >= 0 else 180 - elev(v_out_i),
        pred_out_speed=math.hypot(fwd_comp(v_out_pred), v_out_pred[2]),
        pred_out_angle=math.degrees(math.atan2(v_out_pred[2], fwd_comp(v_out_pred))),
        center_offset=r["hit_center_offset"],
        landing_x=r.get("landing_x", float("nan")),
        v_in_fwd=fwd_comp(v_in), v_in_speed=np.linalg.norm(v_in),
    ))

keys = ["normal_gap_deg", "n_actual_elev", "n_ideal_elev",
        "vpaddle_body", "vpaddle_body_elev", "vpaddle_center", "vpaddle_ideal", "vpaddle_ideal_elev",
        "ball_out_speed", "ball_out_angle", "ideal_out_speed", "ideal_out_angle",
        "pred_out_speed", "pred_out_angle", "center_offset", "landing_x", "v_in_speed", "v_in_fwd"]

def summarize(label, subset):
    if not subset:
        print(f"--- {label}: 0 contacts ---"); return
    print(f"--- {label}: {len(subset)} contacts ---")
    print(f"{'metric':22s} {'mean':>8s} {'median':>8s} {'p10':>8s} {'p90':>8s}")
    for k in keys:
        a = np.array([r[k] for r in subset], float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        print(f"{k:22s} {a.mean():8.3f} {np.median(a):8.3f} {np.percentile(a,10):8.3f} {np.percentile(a,90):8.3f}")
    print()

summarize("ALL contacts", rows)
for oc in ["valid_return", "bad_hit"]:
    summarize(oc, [r for r in rows if r["outcome"] == oc])

# Headline discriminators
vr = [r["normal_gap_deg"] for r in rows if r["outcome"] == "valid_return"]
bh = [r["normal_gap_deg"] for r in rows if r["outcome"] == "bad_hit"]
print("=== HEADLINE ===")
print(f"normal_gap (actual vs ideal): ALL median={np.median([r['normal_gap_deg'] for r in rows]):.1f}deg")
if vr and bh:
    print(f"  valid_return median={np.median(vr):.1f}deg  vs  bad_hit median={np.median(bh):.1f}deg")
print(f"paddle face elevation: actual median={np.median([r['n_actual_elev'] for r in rows]):.1f}deg "
      f"vs ideal median={np.median([r['n_ideal_elev'] for r in rows]):.1f}deg")
print(f"paddle SPEED |v_paddle|: actual(body) median={np.median([r['vpaddle_body'] for r in rows]):.2f} "
      f"center={np.median([r['vpaddle_center'] for r in rows]):.2f} "
      f"ideal median={np.median([r['vpaddle_ideal'] for r in rows]):.2f} m/s")
print(f"ball launch angle: actual median={np.median([r['ball_out_angle'] for r in rows]):.1f}deg "
      f"vs ideal median={np.median([r['ideal_out_angle'] for r in rows]):.1f}deg "
      f"(reflection-from-actual-normal pred median={np.median([r['pred_out_angle'] for r in rows]):.1f}deg)")
