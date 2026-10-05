#!/usr/bin/env python
"""Train the fly brain to drive, and watch it learn while it does."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from flydrive import viz
from flydrive.brain import FlyBrain
from flydrive.config import BrainConfig, EnvConfig, TrainConfig
from flydrive.env import FlyDriveEnv
from flydrive.env_city import CityEnv
from flydrive.ppo import PPO
from flydrive.rollout import DriveCamRecorder, RolloutRecorder
from flydrive.wholebrain import WholeCNS

HEADER = ("  iter    steps     ret    dist  crash   v(m/s)  |lat|   ent     kl    "
          "expl-v    sps   elapsed")


def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fly01")
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--minibatch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="iterations that train only the value head first (fine-tuning from a clone)")
    ap.add_argument("--bc-anchor", type=float, default=0.0,
                    help="weight of the pull toward the resumed policy's action mean (fine-tuning from a clone)")
    ap.add_argument("--bc-anchor-final", type=float, default=None, help="anchor weight at the end of training (linear schedule)")
    ap.add_argument("--bc-anchor-throttle-scale", type=float, default=1.0, help="anchor weight on throttle relative to steering")
    ap.add_argument("--red-penalty", type=float, default=None, help="city: penalty for crossing a stop line on red")
    ap.add_argument("--red-approach-w", type=float, default=None, help="city: per-step penalty x speed while approaching a red within 15 m")
    ap.add_argument("--red-wait-bonus", type=float, default=None, help="city: per-step reward for standing still at a red within 15 m")
    ap.add_argument("--oncoming-w", type=float, default=None, help="city: per-step penalty x metres left of the lane centre beyond 0.8 m")
    ap.add_argument("--follow-w", type=float, default=None, help="city: per-step penalty x speed x closeness to a car ahead within 12 m")
    ap.add_argument("--follow-rel-w", type=float, default=None, help="city: per-step penalty x closeness x closing speed toward the car ahead")
    ap.add_argument("--box-speed-w", type=float, default=None, help="city: per-step penalty x (speed - 3)+ inside an intersection shared with another car")
    ap.add_argument("--n-cities", type=int, default=1, help="city: train on this many layouts at once (seed, 1001, 1002, ...), the batch split evenly")
    ap.add_argument("--reseed-every", type=int, default=0, help="city: rebuild the city from a new seed every N iterations (domain randomisation)")
    ap.add_argument("--seed-pool", type=int, nargs=2, default=(1000, 1100), help="city: [lo, hi) range of layout seeds to draw from")
    ap.add_argument("--freeze-actor-norm", action="store_true",
                    help="cns: keep the descending-neuron input statistics fixed (fine-tuning a ridge-fitted readout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="fuse the brain with torch.compile")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--env", default="track", choices=["track", "city"],
                    help="track: the closed loop; city: signalled grid with other cars")
    ap.add_argument("--model", default="fly", choices=["fly", "cns"],
                    help="fly: the hand-built pathway; cns: the whole MaleCNS connectome")
    ap.add_argument("--video-every", type=int, default=None,
                    help="iterations between rollout videos (each costs ~2 min on the cns model)")
    ap.add_argument("--snap-every", type=int, default=None,
                    help="iterations between neural snapshots")
    ap.add_argument("--city-v3", action="store_true", help="city: irregular block sizes and box buildings")
    ap.add_argument("--lookahead", type=float, default=None, help="city: heading-error term against the route this far ahead (m)")
    ap.add_argument("--corner-bonus", type=float, default=None, help="city: reward for passing a corner on the commanded route")
    ap.add_argument("--fillet", type=float, nargs=2, default=None, metavar=("RIGHT", "LEFT"), help="city: corner radii (m)")
    ap.add_argument("--city-vmax", type=float, default=None, help="city: speed cap (m/s)")
    ap.add_argument("--corner-spawn", type=float, default=None, help="city: fraction of spawns just before a corner")
    ap.add_argument("--steer-reward", type=float, default=None, help="city: per-step reward for steering the commanded way")
    ap.add_argument("--w-heading", type=float, default=None, help="weight of the route-heading error term")
    ap.add_argument("--command-range", type=float, default=None, help="city: metres before a corner the turn command starts")
    ap.add_argument("--wrong-turn-penalty", type=float, default=None, help="city: penalty for leaving the commanded route")
    ap.add_argument("--motor-lr-scale", type=float, default=None,
                    help="cns only: readout lr as a multiple of --lr")
    ap.add_argument("--trunk-lr-scale", type=float, default=None,
                    help="cns only: single-neuron/synaptic lr as a multiple of --lr (0 = frozen brain)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    run = os.path.join("runs", args.name)
    os.makedirs(run, exist_ok=True)

    env_cfg = EnvConfig(n_envs=args.envs)
    train_cfg = TrainConfig(total_steps=args.steps, rollout=args.rollout, critic_warmup=args.critic_warmup,
                            bc_anchor=args.bc_anchor, bc_anchor_final=args.bc_anchor_final,
                            bc_anchor_throttle_scale=args.bc_anchor_throttle_scale,
                            n_minibatch=args.minibatch, epochs=args.epochs,
                            lr=args.lr, seed=args.seed)
    if args.video_every is not None:
        train_cfg.video_every = args.video_every
    if args.snap_every is not None:
        train_cfg.snap_every = args.snap_every
    if args.w_heading is not None:
        env_cfg.w_heading = args.w_heading
    city_cfg = None
    if args.env == "city":
        from flydrive.city import CityConfig
        city_cfg = CityConfig()
        if args.command_range is not None:
            city_cfg.command_range = args.command_range
        if args.wrong_turn_penalty is not None:
            city_cfg.wrong_turn_penalty = args.wrong_turn_penalty
        if args.red_penalty is not None:
            city_cfg.red_penalty = args.red_penalty
        if args.red_approach_w is not None:
            city_cfg.red_approach_w = args.red_approach_w
        if args.red_wait_bonus is not None:
            city_cfg.red_wait_bonus = args.red_wait_bonus
        if args.oncoming_w is not None:
            city_cfg.oncoming_w = args.oncoming_w
        if args.follow_w is not None:
            city_cfg.follow_w = args.follow_w
        if args.follow_rel_w is not None:
            city_cfg.follow_rel_w = args.follow_rel_w
        if args.box_speed_w is not None:
            city_cfg.box_speed_w = args.box_speed_w
        if args.city_v3:
            city_cfg.irregular = True; city_cfg.buildings = True
        if args.lookahead is not None:
            city_cfg.lookahead = args.lookahead
        if args.corner_bonus is not None:
            city_cfg.corner_bonus = args.corner_bonus
        if args.fillet is not None:
            city_cfg.fillet_right, city_cfg.fillet_left = args.fillet
        if args.city_vmax is not None:
            city_cfg.v_max = args.city_vmax
        if args.corner_spawn is not None:
            city_cfg.corner_spawn_frac = args.corner_spawn
        if args.steer_reward is not None:
            city_cfg.steer_reward = args.steer_reward
    Env = (lambda c, **kw: CityEnv(c, city_cfg=city_cfg, **kw)) if args.env == "city" else FlyDriveEnv
    if args.env == "city" and args.n_cities > 1:
        from flydrive.env_city import MultiCityEnv
        seeds = [args.seed] + [1000 + i for i in range(1, args.n_cities)]   # evaluation seeds 123 / 21 / 5 stay out
        env = MultiCityEnv(env_cfg, city_cfg=city_cfg, device=dev, seeds=seeds)
        print(f"training on {args.n_cities} layouts at once, seeds {seeds}, {env.n} cars each")
    else:
        env = Env(env_cfg, device=dev, seed=args.seed)
    env_cfg = env.cfg                       # the city widens proprioception
    if args.model == "cns":
        kind = "linear"
        if args.resume:
            from flydrive.wholebrain import readout_kind
            kind = readout_kind(torch.load(args.resume, map_location="cpu", weights_only=False)["brain"])
        brain = WholeCNS(env_cfg, device=dev, motor_lr_scale=args.motor_lr_scale,
                         trunk_lr_scale=args.trunk_lr_scale, readout=kind).to(dev)
        print(brain.describe())
    else:
        brain = FlyBrain(env_cfg, BrainConfig()).to(dev)
    ppo = PPO(env, brain, train_cfg, device=dev, compile_model=args.compile)
    if args.resume:
        ppo.load(args.resume)
        print(f"resumed from {args.resume} at step {ppo.global_step:,}")
        if args.bc_anchor > 0:
            ppo.set_reference(torch.load(args.resume, map_location=dev, weights_only=False)["brain"])
            print(f"anchored to {args.resume} with weight {args.bc_anchor}")
        if args.freeze_actor_norm and hasattr(brain, "dn_norm"):
            brain.dn_norm.frozen = True
            print("actor input statistics frozen")

    # A separate small env drives the recorder so it never disturbs training.
    vis_env = Env(EnvConfig(n_envs=8), device=dev, seed=args.seed + 77)
    Rec = DriveCamRecorder if args.model == "cns" else RolloutRecorder
    recorder = Rec(vis_env, brain)

    n_iter = max(1, args.steps // (args.rollout * args.envs))
    print(f"device      {dev}  ({torch.cuda.get_device_name(0) if dev=='cuda' else ''})")
    print(f"ommatidia   {env.eye.n_omma} per eye x 2 = {2*env.eye.n_omma}")
    print(f"parameters  {brain.n_params():,} learnable")
    print(f"batch       {args.envs} envs x {args.rollout} steps = {args.envs*args.rollout:,} per iteration")
    print(f"iterations  {n_iter:,}  ->  {args.steps:,} env steps")
    print(f"run dir     {run}")
    print(HEADER)

    t0 = time.time()
    best = -1e9
    if args.resume:
        prior = [v for v in ppo.history.get("ep_distance", []) if v == v]
        if prior:
            best = max(prior)   # do not let the first resumed iteration clobber best.pt
    try:
        import random
        for it in range(ppo.iteration, n_iter):
            if args.env == "city" and args.reseed_every and it > ppo.iteration and it % args.reseed_every == 0:
                # a new layout: routes, phases, traffic and buildings all change
                new_seed = random.randint(args.seed_pool[0], args.seed_pool[1] - 1)
                ppo.obs = env.reseed(new_seed); ppo.prop = env.proprio()
                ppo.state = brain.init_state(ppo.B, dev); ppo.blank = brain.init_state(ppo.B, dev)
                if getattr(ppo, "ref", None) is not None:
                    ppo.ref_state = ppo.ref.init_state(ppo.B, dev); ppo.ref_blank = ppo.ref.init_state(ppo.B, dev)
                env.pop_stats()
            ppo.anneal(it / n_iter)
            stats = ppo.iterate()
            h = ppo.history

            if it % 2 == 0 or it == n_iter - 1:
                print(f"{it:6d} {ppo.global_step/1e3:8.0f}k "
                      f"{h['ep_return'][-1]:7.1f} {h['ep_distance'][-1]:7.1f} "
                      f"{h['crash_rate'][-1]:6.2f} {h['speed'][-1]:7.2f} "
                      f"{h['abs_lateral'][-1]:6.2f} {h['entropy'][-1]:6.2f} "
                      f"{h['kl'][-1]:7.4f} {h['explained_var'][-1]:7.3f} "
                      f"{h['sps'][-1]:7.0f} {fmt_time(time.time()-t0):>9}"
                      + (f"  anchor {h['anchor'][-1]:.4f}" if train_cfg.bc_anchor > 0 else ""), flush=True)

            if it % train_cfg.dash_every == 0 or it == n_iter - 1:
                viz.draw_dashboard(
                    h, os.path.join(run, "dashboard.png"),
                    labels=viz.CNS_LABELS if args.model == "cns" else None,
                    title=f"fly brain -- driving   |   run '{args.name}'",
                    subtitle=(f"{ppo.global_step:,} env steps   "
                              f"{brain.n_params():,} params   "
                              f"{2*env.eye.n_omma} ommatidia   "
                              f"elapsed {fmt_time(time.time()-t0)}"))
                with open(os.path.join(run, "history.json"), "w") as f:
                    json.dump(h, f)

            if (it > 0 and it % (train_cfg.snap_every * (4 if args.model == "cns" else 1)) == 0) \
                    or it == n_iter - 1:
                brain.eval()
                recorder.record(n_steps=60, png=os.path.join(run, "snapshot.png"),
                                title=f"fly brain driving   |   {ppo.global_step:,} steps",
                                subtitle=f"run '{args.name}'")
                brain.train()
                torch.cuda.empty_cache()

            if (not args.no_video) and it > 0 and it % train_cfg.video_every == 0:
                brain.eval()
                out = recorder.record(
                    os.path.join(run, f"rollout_{ppo.global_step//1000:06d}k.mp4"),
                    n_steps=300,
                    title=f"fly brain driving   |   {ppo.global_step:,} steps",
                    subtitle=f"run '{args.name}'")
                brain.train()
                print(f"       video written  ({out['distance']:.0f} m, {out['crashes']} crashes)")

            score = h["ep_distance"][-1]
            if score == score and score > best:
                best = score
                ppo.save(os.path.join(run, "best.pt"))
            if it % train_cfg.ckpt_every == 0:
                ppo.save(os.path.join(run, "last.pt"))
    except KeyboardInterrupt:
        print("\ninterrupted -- saving")
    finally:
        ppo.save(os.path.join(run, "last.pt"))
        viz.draw_dashboard(ppo.history, os.path.join(run, "dashboard.png"),
                           labels=viz.CNS_LABELS if args.model == "cns" else None,
                           title=f"fly brain -- driving   |   run '{args.name}'",
                           subtitle=f"{ppo.global_step:,} env steps   "
                                    f"elapsed {fmt_time(time.time()-t0)}")
        with open(os.path.join(run, "history.json"), "w") as f:
            json.dump(ppo.history, f)
        print(f"\nsaved to {run}  (best mean distance {best:.1f} m)")


if __name__ == "__main__":
    main()
