#!/usr/bin/env python
"""Warm-start a policy from the scripted city expert by behavioural cloning.

Collects expert rollouts (with a little action noise so the data covers the
states a slightly-wrong policy visits), fits the recurrent policy's mean action
to the expert's over 32-step sequences, checks the clone by outcome, and saves a
PPO-format checkpoint so `train.py --resume` can fine-tune it.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from flydrive.config import EnvConfig, TrainConfig
from flydrive.city import CityConfig
from flydrive.env_city import CityEnv
from flydrive.brain import FlyBrain
from flydrive.wholebrain import WholeCNS
from flydrive.expert import expert_action
from flydrive.ppo import PPO

DEV = "cuda"


def city_cfg_from(args):
    cc = CityConfig(fillet_right=args.fillet[0], fillet_left=args.fillet[1], v_max=args.city_vmax)
    if args.city_v3: cc.irregular = True; cc.buildings = True
    return cc


@torch.no_grad()
def collect(env, T, noise):
    obs, prop = env.reset_all(), env.proprio()
    O, P, A, D = [], [], [], []
    for t in range(T):
        a = expert_action(env)
        a_noisy = (a + noise * torch.randn_like(a)).clamp(-1, 1)
        O.append(obs.half()); P.append(prop); A.append(a)          # learn the clean action
        obs, prop, r, d, info = env.step(a_noisy); D.append(d.float())
    return torch.stack(O), torch.stack(P), torch.stack(A), torch.stack(D)


@torch.no_grad()
def collect_on_policy(env, brain, T, mix=0.5):
    """DAgger: the clone drives (blended with the expert), the expert labels."""
    obs, prop = env.reset_all(), env.proprio(); st = brain.init_state(env.B, DEV)
    O, P, A, D = [], [], [], []
    for t in range(T):
        a_exp = expert_action(env)
        a_pol, _, _, st, _ = brain.act(obs, prop, st, deterministic=True)
        a = torch.where(torch.rand(env.B, 1, device=DEV) < mix, a_pol, a_exp)
        O.append(obs.half()); P.append(prop); A.append(a_exp)
        obs, prop, r, d, info = env.step(a); D.append(d.float())
        if d.any(): st.reset_(d, brain.init_state(env.B, DEV))
    return torch.stack(O), torch.stack(P), torch.stack(A), torch.stack(D)


@torch.no_grad()
def refresh_norms(brain, O, P, n=60):
    """Re-estimate any running input statistics on the imitation data.

    A model initialised from an RL checkpoint carries statistics from *that*
    policy's states; if the expert drives differently the normalised inputs
    saturate the clip and the output tanh, and the gradient is exactly zero --
    the whole-brain clone sat at a constant loss until this was added.
    """
    if not hasattr(brain, "commit_norms"):
        return
    brain.train()
    state = brain.init_state(O.shape[1], DEV)
    for t in range(min(n, O.shape[0])):          # no_grad + train mode: statistics accumulate
        _, _, _, state, _ = brain(O[t].float(), P[t], state)
    brain.commit_norms()


def fit(brain, O, P, A, D, epochs, seq=32, mb=64, lr=1e-3, thr_weight=0.0):
    """Sequence BC.  ``thr_weight`` up-weights the throttle error where the
    expert commits hard (|throttle| > 0.5): launching on green from a standstill
    and braking for a red are rare in the data, and an unweighted regression
    hedges them to ~0 -- which is exactly the clone that sits through a green."""
    T, B = A.shape[:2]
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    blank = brain.init_state(mb, DEV)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for b0 in range(0, B, mb):
            idx = slice(b0, min(b0 + mb, B)); nb = idx.stop - idx.start
            state = brain.init_state(nb, DEV)
            for t0 in range(0, T, seq):
                state = state.detach(); loss = 0.0
                for t in range(t0, min(t0 + seq, T)):
                    mu, _, _, state, _ = brain(O[t, idx].float(), P[t, idx], state)
                    err = (mu - A[t, idx]) ** 2
                    # Emphasise the slow states: whether to launch, creep or hold is
                    # decided below ~3 m/s, and those samples are a small minority.
                    slow = P[t, idx, 0] < 0.3
                    braking = A[t, idx, 1] < -0.5
                    w_thr = 1.0 + thr_weight * slow.float() + 0.5 * thr_weight * braking.float()
                    loss = loss + 0.5 * (err[:, 0].mean() + (err[:, 1] * w_thr).mean())
                    d = D[t, idx].bool()
                    if d.any():
                        state = type(state)(*[torch.where(d.view(-1, *([1] * (c.dim() - 1))), nw[:nb], c) for c, nw in zip(state.tensors(), blank.tensors())])
                opt.zero_grad(); (loss / seq).backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 5.0); opt.step()
                tot += loss.item(); n += 1
        print(f"  epoch {ep+1}/{epochs}  action mse {tot/n/seq:.4f}", flush=True)


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
    ap.add_argument("--model", default="fly", choices=["fly", "cns"])
    ap.add_argument("--name", default="bc_fly")
    ap.add_argument("--envs", type=int, default=128); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--fillet", type=float, nargs=2, default=(9.0, 14.0)); ap.add_argument("--city-vmax", type=float, default=10.0)
    ap.add_argument("--city-v3", action="store_true"); ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--dagger", type=int, default=0, help="rounds of on-policy relabelling"); ap.add_argument("--dagger-epochs", type=int, default=6)
    ap.add_argument("--mb", type=int, default=64); ap.add_argument("--seq", type=int, default=32); ap.add_argument("--eval-envs", type=int, default=32)
    ap.add_argument("--thr-weight", type=float, default=0.0, help="extra weight on hard-throttle samples (launch on green, brake for red)")
    args = ap.parse_args()
    cc = city_cfg_from(args)
    env = CityEnv(EnvConfig(n_envs=args.envs), city_cfg=cc, device=DEV, seed=7)
    brain = (WholeCNS(env.cfg, device=DEV) if args.model == "cns" else FlyBrain(env.cfg)).to(DEV)
    if args.init:
        ck = torch.load(args.init, map_location=DEV, weights_only=False); brain.load_state_dict(ck["brain"], strict=False)
    t0 = time.time(); O, P, A, D = collect(env, args.steps, args.noise)
    print(f"collected {args.steps} x {args.envs} = {args.steps*args.envs:,} expert steps in {time.time()-t0:.0f}s")
    refresh_norms(brain, O, P)
    brain.train(); fit(brain, O, P, A, D, args.epochs, seq=args.seq, mb=min(args.mb, args.envs), thr_weight=args.thr_weight)
    for r in range(args.dagger):
        brain.eval(); O2, P2, A2, D2 = collect_on_policy(env, brain, args.steps, mix=0.5 + 0.15 * r)
        O = torch.cat([O, O2], 1); P = torch.cat([P, P2], 1); A = torch.cat([A, A2], 1); D = torch.cat([D, D2], 1)
        print(f"dagger round {r+1}: dataset now {A.shape[0]} x {A.shape[1]} sequences", flush=True)
        refresh_norms(brain, O2, P2)
        brain.train(); fit(brain, O, P, A, D, args.dagger_epochs, seq=args.seq, mb=min(args.mb, args.envs), lr=5e-4, thr_weight=args.thr_weight)
    brain.eval()
    ev = CityEnv(EnvConfig(n_envs=args.eval_envs), city_cfg=cc, device=DEV, seed=123)
    n, ends, red, dist = outcome(ev, brain, 1500)
    print(f"clone outcome over {n} episodes: " + " ".join(f"{k} {v:.2f}" for k, v in ends.items()) + f" | red/ep {red:.2f} | dist {dist:.0f} m")
    os.makedirs(f"runs/{args.name}", exist_ok=True)
    ppo = PPO(env, brain, TrainConfig(rollout=8), device=DEV)   # only to write a compatible checkpoint
    ppo.global_step = 0; ppo.save(f"runs/{args.name}/last.pt"); print(f"saved runs/{args.name}/last.pt")


if __name__ == "__main__":
    main()
