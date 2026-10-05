#!/usr/bin/env python
"""Record a video of a trained (or untrained) fly brain driving."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from flydrive.brain import FlyBrain
from flydrive.config import BrainConfig, EnvConfig
from flydrive.env import FlyDriveEnv
from flydrive.env_city import CityEnv
from flydrive.rollout import DriveCamRecorder, RolloutRecorder
from flydrive.wholebrain import WholeCNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint to load; omit for a random-init brain")
    ap.add_argument("--out", default="runs/watch.mp4")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--title", default=None)
    ap.add_argument("--env", default="track", choices=["track", "city"],
                    help="track: the closed loop; city: signalled grid with other cars")
    ap.add_argument("--model", default="fly", choices=["fly", "cns"],
                    help="fly: hand-built pathway; cns: the whole MaleCNS connectome")
    ap.add_argument("--view", default="neural", choices=["neural", "human"],
                    help="neural: the full brain panel; human: a colour driver's-eye camera")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EnvConfig(n_envs=8)
    if args.env == "city":
        city_cfg = None
        if args.ckpt:   # drive in the geometry the policy was trained for
            ck0 = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            if ck0.get("city_cfg"):
                from flydrive.city import CityConfig
                cc = dict(ck0["city_cfg"]); cc["building_height"] = tuple(cc.get("building_height", (6.0, 20.0)))
                city_cfg = CityConfig(**cc)
        env = CityEnv(cfg, city_cfg=city_cfg, device=dev, seed=args.seed)
    else:
        env = FlyDriveEnv(cfg, device=dev, seed=args.seed)
    cfg = env.cfg
    if args.model == "cns":
        kind = "linear"
        if args.ckpt:
            from flydrive.wholebrain import readout_kind
            kind = readout_kind(torch.load(args.ckpt, map_location="cpu", weights_only=False)["brain"])
        brain = WholeCNS(cfg, device=dev, readout=kind).to(dev).eval()
        if args.view == "neural":
            args.view = "human"   # the optic-lobe panels do not exist for this model
    else:
        brain = FlyBrain(cfg, BrainConfig()).to(dev).eval()
    label = "untrained"
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        missing, unexpected = brain.load_state_dict(ck["brain"], strict=False)
        if missing or unexpected:
            print(f"  note: missing {list(missing)[:4]}  unexpected {list(unexpected)[:4]}")
        label = f"{ck['global_step']:,} env steps"
        print(f"loaded {args.ckpt}  ({label})")

    Rec = DriveCamRecorder if args.view == "human" else RolloutRecorder
    rec = Rec(env, brain)
    out = rec.record(args.out, n_steps=args.steps, fps=args.fps,
                     deterministic=not args.stochastic, progress=True,
                     title=args.title or f"fly brain driving   |   {label}",
                     subtitle=os.path.basename(args.ckpt or "random init"))
    print(f"wrote {args.out}   distance {out['distance']:.0f} m, crashes {out['crashes']}")


if __name__ == "__main__":
    main()
