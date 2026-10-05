"""How well does a city policy steer where the route turns?

Runs the policy and the expert side by side on the policy's own states and
reports, per command (L / S / R) near corners, the mean expert steer, the mean
policy steer and the steering error -- an under-steering clone shows up as a
policy mean far below the expert's.
"""
import argparse, sys
sys.path.insert(0, ".")
import torch
from flydrive.config import EnvConfig
from flydrive.env_city import CityEnv
from flydrive.brain import FlyBrain
from flydrive.wholebrain import WholeCNS, readout_kind
from flydrive.city import CityConfig
from flydrive.expert import expert_action


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--model", default="fly"); ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cc = dict(ck["city_cfg"]); cc["building_height"] = tuple(cc.get("building_height", (6.0, 20.0)))
    env = CityEnv(EnvConfig(n_envs=args.envs), city_cfg=CityConfig(**cc), device=dev, seed=args.seed)
    brain = (WholeCNS(env.cfg, device=dev, readout=readout_kind(ck["brain"])) if args.model == "cns"
             else FlyBrain(env.cfg)).to(dev).eval()
    brain.load_state_dict(ck["brain"], strict=False)
    st = brain.init_state(args.envs, dev)
    obs, prop = env.reset_all(), env.proprio()
    names = ("L approach", "L arc", "R approach", "R arc")
    acc = {k: dict(n=0, exp=0.0, pol=0.0, err=0.0, thr_exp=0.0, thr_pol=0.0, v=0.0) for k in names}
    for t in range(args.steps):
        a, _, _, st, _ = brain.act(obs, prop, st, deterministic=True)
        a_exp = expert_action(env)
        _, _, arc, _ = env.routes.nearest_centre(env.route, env.pos, near=env.prev_arc)
        turn, dc = env.routes.next_corner(env.route, arc)
        moving = env.speed > 1.0
        masks = {"L approach": (turn == 0) & (dc > 0) & (dc < 12) & moving, "L arc": (turn == 0) & (dc <= 0) & (dc > -22) & moving,
                 "R approach": (turn == 2) & (dc > 0) & (dc < 12) & moving, "R arc": (turn == 2) & (dc <= 0) & (dc > -14) & moving}
        for name in names:
            m = masks[name]
            if m.any():
                d = acc[name]; d["n"] += int(m.sum())
                d["exp"] += float(a_exp[m, 0].sum()); d["pol"] += float(a[m, 0].sum())
                d["err"] += float(((a[m, 0] - a_exp[m, 0]) ** 2).sum())
                d["thr_exp"] += float(a_exp[m, 1].sum()); d["thr_pol"] += float(a[m, 1].sum()); d["v"] += float(env.speed[m].sum())
        obs, prop, r, d, info = env.step(a)
        if d.any():
            st.reset_(d, brain.init_state(args.envs, dev))
    print("moving, by corner phase: steer mean expert / policy, steer rmse, throttle mean expert / policy, speed")
    for name in names:
        d = acc[name]; n = max(d["n"], 1)
        print(f"  {name:11s}: n {d['n']:6d}  steer {d['exp']/n:+.2f} / {d['pol']/n:+.2f}  rmse {(d['err']/n)**0.5:.2f}  "
              f"throttle {d['thr_exp']/n:+.2f} / {d['thr_pol']/n:+.2f}  v {d['v']/n:.1f} m/s")


if __name__ == "__main__":
    main()
