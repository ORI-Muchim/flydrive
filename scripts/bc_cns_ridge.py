#!/usr/bin/env python
"""Imitate the city expert with the *frozen* whole brain: fit only the linear
descending-neuron readout, in closed form.

The brain runs forward with no gradient; the readout's own input features
(normalised descending-neuron rates) are collected alongside the expert's
actions, and a ridge regression onto atanh(action) gives the motor weights
directly.  DAgger rounds let the clone drive, relabel with the expert, and
refit -- forward passes only, so each round takes seconds.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from flydrive.config import EnvConfig, TrainConfig
from flydrive.city import CityConfig
from flydrive.env_city import CityEnv
from flydrive.wholebrain import WholeCNS
from flydrive.expert import expert_action
from flydrive.ppo import PPO
DEV = "cuda"


@torch.no_grad()
def features(brain, image, prop, state):
    """The exact vector the motor layer reads, plus the next recurrent state."""
    mu, _, _, new_state, tel = brain(image, prop, state, telemetry=True)
    rate = tel["rate"]
    z = brain.dn_norm(rate[:, brain.dn_idx]) * brain.dn_scale
    return z, mu, new_state


@torch.no_grad()
def warm_stats(env, brain, T=120):
    """Set the readout's input statistics once, on expert driving, then freeze.

    Statistics that keep moving between a fit and its deployment shift every
    feature the linear readout was fitted on; the first version of this clone
    left the road within 9 m for exactly that reason."""
    obs, prop = env.reset_all(), env.proprio(); st = brain.init_state(env.B, DEV)
    brain.train()
    for t in range(T):
        _, _, _, st, _ = brain(obs, prop, st)
        obs, prop, r, d, info = env.step(expert_action(env))
        if d.any(): st.reset_(d, brain.init_state(env.B, DEV))
    brain.commit_norms(); brain.eval()


@torch.no_grad()
def collect(env, brain, T, policy_mix=0.0, noise=0.05):
    obs, prop = env.reset_all(), env.proprio(); st = brain.init_state(env.B, DEV)
    Z, A = [], []
    brain.eval()                                     # statistics frozen: features match deployment
    for t in range(T):
        z, mu, st = features(brain, obs, prop, st)
        a_exp = expert_action(env)
        drive = torch.where(torch.rand(env.B, 1, device=DEV) < policy_mix, mu, a_exp)
        Z.append(z.half().cpu()); A.append(torch.cat([a_exp, prop[:, :1]], 1).cpu())   # the dataset lives on the CPU; speed rides along for the launch weight
        obs, prop, r, d, info = env.step((drive + noise * torch.randn_like(drive)).clamp(-1, 1))
        if d.any(): st.reset_(d, brain.init_state(env.B, DEV))
    return torch.cat(Z), torch.cat(A)


def mlp_fit(brain, Z, A, epochs=6, bs=2048, lr=1e-3):
    """Supervised fit of the MLP readout on precomputed (frozen-brain) features."""
    opt = torch.optim.Adam(brain.motor.parameters(), lr=lr)
    n = Z.shape[0]; last = 0.0
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0; k = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            z = Z[idx].to(DEV).float(); a = A[idx, :2].to(DEV)
            loss = ((torch.tanh(brain.motor(z)) - a) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); k += 1
        last = tot / k
    return last


def ridge_fit(brain, Z, A, lam=1e-2, thr_weight=0.0):
    """Closed-form readout.  ``A`` carries [steer, throttle, speed/v_max]; the
    throttle column is fitted with launches from standstill and braking
    up-weighted, the hedged decisions a plain least squares averages away."""
    speed = A[:, 2]; A = A[:, :2]
    w = torch.ones(A.shape[0])
    w = w + thr_weight * ((A[:, 1] > 0.5) & (speed < 0.2)).float() + thr_weight * (A[:, 1] < -0.5).float()   # launches and braking, equal weight
    # Normal equations accumulated in chunks, so the dataset never has to sit
    # on the GPU as one double matrix (that is what ran the MIG slice out).
    n, d = Z.shape[0], Z.shape[1] + 1
    XtX = torch.zeros(d, d, device=DEV, dtype=torch.float64); XtY = torch.zeros(d, 2, device=DEV, dtype=torch.float64)
    XtWX = torch.zeros(d, d, device=DEV, dtype=torch.float64); XtWy = torch.zeros(d, device=DEV, dtype=torch.float64)
    for i in range(0, n, 16384):
        z = Z[i:i + 16384].to(DEV).float()
        X = torch.cat([z, torch.ones(z.shape[0], 1, device=DEV)], 1).double()
        Y = torch.atanh(A[i:i + 16384].to(DEV).clamp(-0.97, 0.97)).double()
        XtX += X.T @ X; XtY += X.T @ Y
        wi = w[i:i + 16384].to(DEV).double()
        XtWX += X.T @ (X * wi.unsqueeze(1)); XtWy += X.T @ (Y[:, 1] * wi)
    reg = lam * n * torch.eye(d, device=DEV, dtype=torch.float64)
    W = torch.linalg.solve(XtX + reg, XtY)
    if thr_weight > 0:
        W[:, 1] = torch.linalg.solve(XtWX + reg * w.mean().item(), XtWy)
    with torch.no_grad():
        brain.motor.weight.copy_(W[:-1].T.float()); brain.motor.bias.copy_(W[-1].float())
    err = 0.0
    for i in range(0, n, 16384):
        z = Z[i:i + 16384].to(DEV).float()
        X = torch.cat([z, torch.ones(z.shape[0], 1, device=DEV)], 1).double()
        err += ((torch.tanh((X @ W).float()) - A[i:i + 16384].to(DEV)) ** 2).sum().item()
    return err / (2 * n)


@torch.no_grad()
def outcome(env, brain, steps):
    obs, prop = env.reset_all(), env.proprio(); st = brain.init_state(env.B, DEV)
    ends = dict(off_road=0, wrong_turn=0, collision=0, stalled=0, timeout=0, finished=0); n = 0
    for t in range(steps):
        a, _, _, st, _ = brain.act(obs, prop, st, deterministic=True)
        obs, prop, r, d, info = env.step(a)
        if d.any():
            for k in ("off_road", "wrong_turn", "collision", "stalled", "timeout"): ends[k] += int((info[k] & d).sum())
            ends["finished"] += int((d & ~(info["off_road"] | info["wrong_turn"] | info["collision"] | info["stalled"] | info["timeout"])).sum()); n += int(d.sum())
            st.reset_(d, brain.init_state(env.B, DEV))
    s = env.pop_stats() or {}
    return n, {k: v / max(n, 1) for k, v in ends.items()}, s.get("red", float("nan")), s.get("distance", float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bc_cns"); ap.add_argument("--init", default="runs/city_cns/last.pt")
    ap.add_argument("--envs", type=int, default=96); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dagger", type=int, default=3); ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--fillet", type=float, nargs=2, default=(9.0, 14.0)); ap.add_argument("--city-vmax", type=float, default=10.0)
    ap.add_argument("--readout", default="linear", choices=["linear", "mlp"]); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=300000); ap.add_argument("--max-mix", type=float, default=0.7)
    ap.add_argument("--cmd-gain", type=float, default=None, help="effective turn-command current into PFL3 (default: the checkpoint's)")
    ap.add_argument("--thr-weight", type=float, default=0.0, help="extra weight on launch/brake samples in the throttle fit")
    ap.add_argument("--city-v3", action="store_true", help="irregular blocks + buildings")
    args = ap.parse_args()
    cc = CityConfig(fillet_right=args.fillet[0], fillet_left=args.fillet[1], v_max=args.city_vmax,
                    irregular=args.city_v3, buildings=args.city_v3)
    env = CityEnv(EnvConfig(n_envs=args.envs), city_cfg=cc, device=DEV, seed=7)
    from flydrive.wholebrain import load_compatible
    brain = WholeCNS(env.cfg, device=DEV, readout=args.readout).to(DEV)
    ck = torch.load(args.init, map_location=DEV, weights_only=False); load_compatible(brain, ck["brain"])
    if args.cmd_gain is not None:
        import math
        with torch.no_grad():   # softplus^-1 so that softplus(cmd_gain) == the requested current
            brain.cmd_gain.fill_(math.log(math.expm1(args.cmd_gain)))
        print(f"turn command current set to {args.cmd_gain}")
    ev = CityEnv(EnvConfig(n_envs=16), city_cfg=cc, device=DEV, seed=123)
    def report(tag):
        n, ends, red, dist = outcome(ev, brain, 900)
        print(f"  {tag} outcome over {n} eps: " + " ".join(f"{k} {v:.2f}" for k, v in ends.items()) + f" | red/ep {red:.2f} | dist {dist:.0f} m", flush=True)
        return dist
    t0 = time.time(); warm_stats(env, brain); Z, A = collect(env, brain, args.steps)
    fit = (lambda Z, A: mlp_fit(brain, Z, A, args.epochs)) if args.readout == "mlp" else (lambda Z, A: ridge_fit(brain, Z, A, args.lam, args.thr_weight))
    mse = fit(Z, A); print(f"round 0: {Z.shape[0]:,} samples, readout mse {mse:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    dist = report("round 0")
    for r in range(args.dagger):
        mix = min(args.max_mix, 0.3 if dist < 60 else 0.5 + 0.1 * r)   # lean on the expert while the clone is weak
        Z2, A2 = collect(env, brain, args.steps, policy_mix=mix)
        Z = torch.cat([Z, Z2])[-args.max_samples:]; A = torch.cat([A, A2])[-args.max_samples:]
        mse = fit(Z, A); print(f"dagger {r+1} (mix {mix:.2f}): {Z.shape[0]:,} samples, readout mse {mse:.4f}", flush=True)
        dist = report(f"dagger {r+1}")
    n, ends, red, dist = outcome(ev, brain, 1500)
    print(f"clone outcome over {n} episodes: " + " ".join(f"{k} {v:.2f}" for k, v in ends.items()) + f" | red/ep {red:.2f} | dist {dist:.0f} m", flush=True)
    os.makedirs(f"runs/{args.name}", exist_ok=True)
    ppo = PPO(env, brain, TrainConfig(rollout=8), device=DEV); ppo.global_step = 0; ppo.save(f"runs/{args.name}/last.pt"); print(f"saved runs/{args.name}/last.pt")


if __name__ == "__main__":
    main()
