"""Classify every collision of a city driver (expert or checkpoint).

rear_end: the other car ahead, same heading, ego faster.
rear_ended: the other car behind, same heading.
oncoming: headings opposed.  crossing: headings perpendicular, inside an
intersection.  Reports the ego speed, whether the ego was turning, and the
signal for the ego's approach at the moment of impact.
"""
import argparse, math, sys
sys.path.insert(0, ".")
import torch
from flydrive.config import EnvConfig
from flydrive.env_city import CityEnv
from flydrive.brain import FlyBrain
from flydrive.wholebrain import WholeCNS, readout_kind
from flydrive.city import CityConfig, STRAIGHT


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?"); ap.add_argument("--model", default="fly"); ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--expert", action="store_true"); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fillet", type=float, nargs=2, default=(9.0, 14.0)); ap.add_argument("--city-vmax", type=float, default=10.0)
    ap.add_argument("--city-v3", action="store_true"); ap.add_argument("--no-crawl", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        cc = dict(ck["city_cfg"]); cc["building_height"] = tuple(cc.get("building_height", (6.0, 20.0)))
        city_cfg = CityConfig(**cc)
    else:
        kw = dict(fillet_right=args.fillet[0], fillet_left=args.fillet[1], v_max=args.city_vmax)
        if args.city_v3: kw.update(irregular=True, buildings=True)
        city_cfg = CityConfig(**kw)
    env = CityEnv(EnvConfig(n_envs=args.envs), city_cfg=city_cfg, device=dev, seed=args.seed)
    brain = None
    if not args.expert:
        brain = (WholeCNS(env.cfg, device=dev, readout=readout_kind(ck["brain"])) if args.model == "cns"
                 else FlyBrain(env.cfg)).to(dev).eval()
        brain.load_state_dict(ck["brain"], strict=False)
        st = brain.init_state(args.envs, dev)
    from flydrive.expert import expert_action
    obs, prop = env.reset_all(), env.proprio()
    B = args.envs
    cats = dict(rear_end=0, rear_ended=0, oncoming=0, crossing=0, side=0)
    turning, in_inter, sig_red, spd = 0, 0, 0, []
    n_ep = 0; n_col = 0
    ends = dict(off_road=0, wrong_turn=0, collision=0, stalled=0, timeout=0, finished=0); reds = 0
    wrong = dict(wrong_branch=0, lane_drift=0)
    keys = ("pos", "heading", "clock", "speed", "route", "npc_route", "npc_s", "prev_arc", "npc_v")
    prev_dstop = torch.full((B,), 1e9, device=dev); prev_sig = torch.full((B,), 2.0, device=dev)
    for t in range(args.steps):
        if args.expert:
            a = expert_action(env, crawl_shared=not args.no_crawl)
        else:
            a, _, _, st, _ = brain.act(obs, prop, st, deterministic=True)
        pre = {k: getattr(env, k).clone() for k in keys}
        obs, prop, r, d, info = env.step(a)
        reds += int(info["red_run"].sum())
        if info["red_run"].any() and reds <= 10:
            for i in torch.nonzero(info["red_run"]).flatten().tolist():
                print(f"    red run: speed {float(pre['speed'][i]):.1f} -> {float(env.speed[i]):.1f}, d_stop before {float(prev_dstop[i]):.1f}, "
                      f"signal before {float(prev_sig[i]):.0f} now {float(info['signal'][i]):.0f}, d_stop now {float(info['d_stop'][i]):.1f}, d_arc {float(info['progress'][i]):.2f}, lat {float(info['lateral'][i]):.1f}")
        prev_dstop = info["d_stop"].clone(); prev_sig = info["signal"].clone()
        if d.any():
            n_ep += int(d.sum())
            for kk in ("off_road", "wrong_turn", "collision", "stalled", "timeout"):
                ends[kk] += int((info[kk] & d).sum())
            ends["finished"] += int((d & ~(info["off_road"] | info["wrong_turn"] | info["collision"] | info["stalled"] | info["timeout"])).sum())
            tmo = info["timeout"] & d
            if tmo.any() and ends["timeout"] <= 40:
                p_npc, t_npc = env._npc_pose()   # post-respawn for the ego, but the pre-step values are what matter
                for i in torch.nonzero(tmo).flatten().tolist()[:10]:
                    fwd_i = torch.stack([torch.cos(pre["heading"][i]), torch.sin(pre["heading"][i])])
                    left_i = torch.stack([-fwd_i[1], fwd_i[0]])
                    pp, tt = env.routes.point_at(pre["npc_route"][i], pre["npc_s"][i])
                    rr = pp - pre["pos"][i]; fxi = rr @ fwd_i; fyi = rr @ left_i
                    k = int(rr.norm(dim=-1).argmin())
                    print(f"    timeout: speed {float(pre['speed'][i]):.1f}, d_stop {float(prev_dstop[i]):.1f}, signal {float(prev_sig[i]):.0f}, "
                          f"nearest car fx {float(fxi[k]):+.1f} fy {float(fyi[k]):+.1f} v {float(pre['npc_v'][i, k]):.1f}, lap {float(info['lap_arc'][i]):.0f} m")
            wt = info["wrong_turn"] & d
            if wt.any():
                # Which kind of wrong turn: the wrong branch (heading ~90 deg off
                # the route) or lane discipline (drifted wide, heading still along it)?
                for i in torch.nonzero(wt).flatten().tolist():
                    _, lat_i, arc_i, tan_i = env.routes.nearest_centre(pre["route"][i:i+1], pre["pos"][i:i+1], near=pre["prev_arc"][i:i+1])
                    herr = math.degrees(abs(math.remainder(float(pre["heading"][i]) - math.atan2(float(tan_i[0, 1]), float(tan_i[0, 0])), 2 * math.pi)))
                    turn_i, dc_i = env.routes.next_corner(pre["route"][i:i+1], arc_i)
                    kind = "wrong_branch" if herr > 45 else "lane_drift"
                    wrong[kind] += 1
                    if sum(wrong.values()) <= 10:
                        print(f"    wrong turn: {kind:12s} heading off route {herr:5.1f} deg, lat {float(lat_i[0]):+.1f} m, speed {float(pre['speed'][i]):.1f}, "
                              f"cmd {int(turn_i[0])} d_corner {float(dc_i[0]):.0f} m")
            col = info["collision"] & d
            if col.any():
                post = {k: getattr(env, k).clone() for k in keys}
                for k, v in pre.items(): getattr(env, k).copy_(v)
                p_npc, t_npc = env._npc_pose()
                fwd = torch.stack([torch.cos(env.heading), torch.sin(env.heading)], -1)
                left = torch.stack([-fwd[:, 1], fwd[:, 0]], -1)
                rel = p_npc - env.pos.unsqueeze(1)
                fx = (rel * fwd.unsqueeze(1)).sum(-1); fy = (rel * left.unsqueeze(1)).sum(-1)
                dist = rel.norm(dim=-1)
                _, _, arc, _ = env.routes.nearest_centre(env.route, env.pos, near=env.prev_arc)
                turn, d_corner = env.routes.next_corner(env.route, arc)
                d_stop, lamp = env.routes.next_stop(env.route, arc)
                state = env.city.signal_state(env.clock)
                my = state[torch.arange(B, device=dev), lamp]
                near_c = torch.cdist(env.pos, env.city.centres).min(dim=1).values
                for i in torch.nonzero(col).flatten().tolist():
                    k = int(dist[i].argmin())
                    cosang = float((t_npc[i, k] * fwd[i]).sum())
                    same, opp = cosang > 0.7, cosang < -0.7
                    if same and fx[i, k] > 0: cats["rear_end"] += 1
                    elif same and fx[i, k] <= 0: cats["rear_ended"] += 1
                    elif opp: cats["oncoming"] += 1
                    elif near_c[i] < env.ccfg.road_width / 2 + 2: cats["crossing"] += 1
                    else: cats["side"] += 1
                    n_col += 1
                    if n_col <= 12:
                        kind = [k for k, v in cats.items() if v][-1] if False else ("rear_end" if same and fx[i, k] > 0 else "rear_ended" if same else "oncoming" if opp else "crossing/side")
                        print(f"    {kind:11s} ego v {env.speed[i]:.1f} npc v {env.npc_v[i, k]:.1f}  npc at fx {fx[i, k]:+.1f} fy {fy[i, k]:+.1f}  "
                              f"d_stop {d_stop[i]:.1f} sig {my[i]:.0f}  turn {int(turn[i])} d_corner {d_corner[i]:.0f}  centre {near_c[i]:.1f} m")
                    turning += int((turn[i] != STRAIGHT) and (-20 < d_corner[i] < 15)); in_inter += int(near_c[i] < env.ccfg.road_width / 2 + 2)
                    sig_red += int(my[i] < 0.5 and d_stop[i] < 12); spd.append(float(env.speed[i]))
                for k, v in post.items(): getattr(env, k).copy_(v)
            if brain is not None:
                st.reset_(d, brain.init_state(B, dev))
    print("outcome: " + " ".join(f"{k} {v/max(n_ep,1):.2f}" for k, v in ends.items()) + f" | red/ep {reds/max(n_ep,1):.2f}")
    print(f"wrong turns: wrong_branch {wrong['wrong_branch']}, lane_drift {wrong['lane_drift']}")
    print(f"{n_ep} episodes, {n_col} collisions ({100*n_col/max(n_ep,1):.0f}%): " + ", ".join(f"{k} {v}" for k, v in cats.items()))
    if n_col:
        print(f"  at impact: turning {turning}/{n_col}, inside intersection {in_inter}/{n_col}, own signal red {sig_red}/{n_col}, "
              f"ego speed mean {sum(spd)/len(spd):.1f} m/s")


if __name__ == "__main__":
    main()
