#!/usr/bin/env python
"""Evaluate a checkpoint in the city and separate how episodes end."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from flydrive.config import EnvConfig
from flydrive.env_city import CityEnv
from flydrive.brain import FlyBrain
from flydrive.wholebrain import WholeCNS


@torch.no_grad()
def evaluate(model, ckpt, n_envs=32, steps=1500, seed=123, dev="cuda", fallback_cfg=None, force_cfg=False):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    city_cfg = fallback_cfg          # for checkpoints written before the config was stored
    if ck.get("city_cfg") and not force_cfg:
        from flydrive.city import CityConfig
        cc = dict(ck["city_cfg"]); cc["building_height"] = tuple(cc.get("building_height", (6.0, 20.0)))
        city_cfg = CityConfig(**cc)
    env = CityEnv(EnvConfig(n_envs=n_envs), city_cfg=city_cfg, device=dev, seed=seed)
    from flydrive.wholebrain import readout_kind
    if model == "expert":            # the scripted reference driver, in this checkpoint's city
        from flydrive.expert import expert_action
        brain = None
    else:
        brain = (WholeCNS(env.cfg, device=dev, readout=readout_kind(ck["brain"])) if model == "cns"
                 else FlyBrain(env.cfg)).to(dev).eval()
        brain.load_state_dict(ck["brain"], strict=False)
    obs, prop = env.reset_all(), env.proprio()
    st = brain.init_state(n_envs, dev) if brain is not None else None
    ends = dict(off_road=0, wrong_turn=0, collision=0, stalled=0, timeout=0, finished=0)
    n_ep, speed = 0, []
    for t in range(steps):
        if brain is None:
            a = expert_action(env)
        else:
            a, _, _, st, _ = brain.act(obs, prop, st, deterministic=True)
        obs, prop, r, d, info = env.step(a)
        speed.append(env.speed.mean().item())
        if d.any():
            for k in ("off_road", "wrong_turn", "collision", "stalled", "timeout"):
                ends[k] += int((info[k] & d).sum())
            fin = d & ~(info["off_road"] | info["wrong_turn"] | info["collision"] | info["stalled"] | info["timeout"])
            ends["finished"] += int(fin.sum())
            n_ep += int(d.sum())
            if st is not None:
                st.reset_(d, brain.init_state(n_envs, dev))
    s = env.pop_stats() or {}
    return dict(step=ck.get("global_step", 0) or 0, episodes=n_ep, ends=ends, red_per_episode=s.get("red", float("nan")),
                distance=s.get("distance", float("nan")), speed=sum(speed) / len(speed), length=s.get("length", float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="model:run  e.g. fly:city_fly cns:city_cns")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--fillet", type=float, nargs=2, default=None, help="geometry for checkpoints without a stored config")
    ap.add_argument("--city-vmax", type=float, default=None)
    ap.add_argument("--city-v3", action="store_true")
    ap.add_argument("--out", default=None, help="write the rows as JSON (for scripts/plot_city_results.py)")
    ap.add_argument("--force-cfg", action="store_true", help="evaluate in the city built from the CLI flags even if the checkpoint stores its own")
    ap.add_argument("--seed", type=int, default=123, help="environment seed (routes, signals, traffic)")
    a = ap.parse_args()
    rows = []
    from flydrive.city import CityConfig
    fb = CityConfig()
    if a.fillet: fb.fillet_right, fb.fillet_left = a.fillet
    if a.city_vmax: fb.v_max = a.city_vmax
    if a.city_v3: fb.irregular = True; fb.buildings = True
    print("%-10s %7s %5s %8s %8s %8s %8s %8s %8s %7s %7s %8s" % ("run", "step", "eps", "offroad", "wrongturn", "collide", "stall", "timeout", "finish", "red/ep", "dist", "speed"))
    for spec in a.runs:
        model, run = spec.split(":")[:2]
        ck = f"runs/{run}/best.pt"
        if not os.path.exists(ck):      # a resumed run only writes best.pt once it beats its parent
            ck = f"runs/{run}/last.pt"
        r = evaluate(model, ck, steps=a.steps, seed=a.seed, fallback_cfg=fb, force_cfg=a.force_cfg)
        e, n = r["ends"], max(r["episodes"], 1)
        label = "expert" if model == "expert" else run
        print("%-10s %6.0fk %5d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %7.2f %7.0f %8.1f" % (
            label, r["step"] / 1e3, r["episodes"], e["off_road"] / n, e["wrong_turn"] / n, e["collision"] / n, e["stalled"] / n, e["timeout"] / n, e["finished"] / n,
            r["red_per_episode"], r["distance"], r["speed"]), flush=True)
        rows.append(dict(run=label, model=model, episodes=r["episodes"], red_per_episode=r["red_per_episode"], distance=r["distance"], speed=r["speed"],
                         **{k: v / n for k, v in e.items()}))
    if a.out:
        import json
        json.dump(rows, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
