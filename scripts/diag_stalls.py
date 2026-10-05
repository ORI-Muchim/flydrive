"""Where does a city policy stall?  Classifies every stall termination.

Categories: green_no_restart (stopped near the line, light already green),
behind_car (a car within 12 m ahead), red_far (stopped well before a red),
other.  For green_no_restart it also reports where the lamp sits in the eye.
"""
import argparse, math, sys
sys.path.insert(0, ".")
import torch
from flydrive.config import EnvConfig
from flydrive.env_city import CityEnv
from flydrive.brain import FlyBrain
from flydrive.wholebrain import WholeCNS, readout_kind
from flydrive.city import CityConfig


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--model", default="fly"); ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--expert", action="store_true"); ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cc = dict(ck["city_cfg"]); cc["building_height"] = tuple(cc.get("building_height", (6.0, 20.0)))
    env = CityEnv(EnvConfig(n_envs=args.envs), city_cfg=CityConfig(**cc), device=dev, seed=args.seed)
    brain = (WholeCNS(env.cfg, device=dev, readout=readout_kind(ck["brain"])) if args.model == "cns"
             else FlyBrain(env.cfg)).to(dev).eval()
    brain.load_state_dict(ck["brain"], strict=False)
    if args.expert:
        from flydrive.expert import expert_action
    obs, prop = env.reset_all(), env.proprio()
    st = brain.init_state(args.envs, dev)
    B = args.envs
    cats = dict(green_no_restart=0, behind_car=0, red_far=0, other=0)
    elev, azim, thr = [], [], []
    green_since = torch.full((B,), -1.0, device=dev)   # clock when the light last turned green while waiting
    prev_sig = torch.zeros(B, device=dev)
    restart_delays = []
    n_ep = 0
    for t in range(args.steps):
        a = expert_action(env) if args.expert else brain.act(obs, prop, st, deterministic=True)[0]
        if not args.expert:
            st = brain.act(obs, prop, st, deterministic=True)[3]
        # env.step respawns finished cars before it returns, so keep the pre-step pose
        pre = {k: getattr(env, k).clone() for k in ("pos", "heading", "clock", "speed", "route", "npc_route", "npc_s", "prev_arc")}
        obs, prop, r, d, info = env.step(a)
        sig, d_stop = info["signal"], info["d_stop"]
        near = (d_stop < 12.0) & (d_stop > -3.0)
        turned_green = (prev_sig < 1.5) & (sig >= 1.5) & near & (pre["speed"] < 0.5)
        green_since = torch.where(turned_green, torch.full((B,), float(t), device=dev), green_since)
        moving = (pre["speed"] > 1.0) & (green_since >= 0) & ~d
        if moving.any():
            restart_delays += ((t - green_since) * env.cfg.dt)[moving].tolist(); green_since[moving] = -1.0
        prev_sig = sig.clone()
        if d.any():
            n_ep += int(d.sum())
            stalled = info["stalled"] & d
            if stalled.any():
                post = {k: getattr(env, k).clone() for k in pre}
                for k, v in pre.items():          # look at the world as it was when the car stalled
                    getattr(env, k).copy_(v)
                p_npc, _ = env._npc_pose()
                fwd = torch.stack([torch.cos(env.heading), torch.sin(env.heading)], -1)
                left = torch.stack([-fwd[:, 1], fwd[:, 0]], -1)
                rel = p_npc - env.pos.unsqueeze(1)
                fx = (rel * fwd.unsqueeze(1)).sum(-1); fy = (rel * left.unsqueeze(1)).sum(-1)
                car_ahead = ((fx > 0) & (fx < 12.0) & (fy.abs() < 3.2)).any(-1)
                _, _, arc, _ = env.routes.nearest_centre(env.route, env.pos, near=env.prev_arc)
                _, lamp = env.routes.next_stop(env.route, arc)
                lxy = env.city.lamp_xy[lamp]
                for i in torch.nonzero(stalled).flatten().tolist():
                    if sig[i] >= 1.5 and near[i]:
                        cats["green_no_restart"] += 1
                        if cats["green_no_restart"] <= 6:
                            # Can the eye tell this green from a red?  Re-render with the clock
                            # shifted until this lamp is red and compare the two images.
                            base = env.observe()[i]
                            saved = env.clock[i].item(); best = None
                            for sh in (3.0, 6.0, 9.0, 12.0, 15.0):
                                env.clock[i] = saved + sh
                                if env.city.signal_state(env.clock)[i, lamp[i]] < 0.5:
                                    best = (env.observe()[i] - base).abs(); break
                            env.clock[i] = saved
                            dl = (lxy[i] - env.pos[i]).norm().item()
                            print(f"    stall: d_stop {d_stop[i]:.1f} m, lamp {dl:.1f} m away, speed {env.speed[i]:.2f}, "
                                  f"throttle {a[i,1]:+.2f}, red-vs-green image diff: max {best.max():.3f}, "
                                  f"n>0.05: {(best > 0.05).sum().item()} of {best.numel()}" if best is not None else "    stall: (no red shift found)")
                        rl = lxy[i] - env.pos[i]; dist = rl.norm().item()
                        az = math.degrees(math.atan2((rl * left[i]).sum().item(), (rl * fwd[i]).sum().item()))
                        el = math.degrees(math.atan2(env.ccfg.lamp_height - env.cfg.car.eye_height, dist))
                        elev.append(el); azim.append(az); thr.append(a[i, 1].item())
                    elif car_ahead[i]: cats["behind_car"] += 1
                    elif sig[i] < 1.5 and d_stop[i] >= 12.0: cats["red_far"] += 1
                    else: cats["other"] += 1
                for k, v in post.items():
                    getattr(env, k).copy_(v)
            green_since[d] = -1.0
            if not args.expert:
                st.reset_(d, brain.init_state(B, dev))
    tot = sum(cats.values())
    print(f"{n_ep} episodes, {tot} stalls: " + ", ".join(f"{k} {v}" for k, v in cats.items()))
    if elev:
        e, z = torch.tensor(elev), torch.tensor(azim)
        print(f"  green_no_restart lamp position in the eye: elevation {e.mean():.0f}° (min {e.min():.0f}, max {e.max():.0f}), "
              f"azimuth {z.mean():.0f}° (left +; min {z.min():.0f}, max {z.max():.0f}); throttle output {sum(thr)/len(thr):+.2f}")
    if restart_delays:
        rd = torch.tensor(restart_delays)
        print(f"  restarts after green: {len(rd)} (median {rd.median():.1f} s, 90% {rd.quantile(0.9):.1f} s)")


if __name__ == "__main__":
    main()
