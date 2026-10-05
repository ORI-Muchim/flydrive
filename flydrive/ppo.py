"""Recurrent PPO for the fly brain.

The policy carries state -- temporal filters in the medulla, a ring attractor in
the central complex, a recurrent premotor pool -- so the update cannot shuffle
timesteps.  Instead each minibatch is a subset of *environments*, replayed
forward through the whole rollout from the stored initial brain state, with
truncated BPTT over the segment.  Resets are replayed too, so the recomputed
trajectory matches the one that was collected.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .brain import FlyBrain
from .config import TrainConfig
from .env import FlyDriveEnv


class Rollout:
    """Flat storage for one on-policy segment."""

    def __init__(self, T, B, obs_shape, device, n_proprio=3):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=device, dtype=dt)
        self.obs = z(T, B, *obs_shape)
        self.proprio = z(T, B, n_proprio)
        self.actions = z(T, B, 2)
        self.logp = z(T, B)
        self.values = z(T, B)
        self.rewards = z(T, B)
        self.dones = z(T, B)
        self.adv = z(T, B)
        self.ret = z(T, B)
        self.mu_ref = z(T, B, 2)     # the reference policy's action mean (bc_anchor)
        self.T, self.B = T, B


class PPO:
    def __init__(self, env: FlyDriveEnv, brain: nn.Module, cfg: TrainConfig | None = None,
                 device="cuda", compile_model: bool = False):
        self.env = env
        self.brain = brain
        # The network is tiny but deep in *kernel count*, so it is entirely
        # launch-bound; fusing it is worth more than any batch-size change.
        self.fwd = torch.compile(brain, dynamic=False) if compile_model else brain
        self.cfg = cfg or TrainConfig()
        self.device = torch.device(device)
        # eps must stay well below the smallest useful gradient: parameters deep
        # in the optic lobe carry far smaller gradients than the shared scalar
        # biases, and a large eps silently freezes them.
        groups = (brain.param_groups(self.cfg.lr) if hasattr(brain, "param_groups")
                  else [{"params": list(brain.parameters()), "lr": self.cfg.lr}])
        self.opt = torch.optim.Adam(groups, eps=1e-8)
        for g in self.opt.param_groups:
            g["initial_lr"] = g["lr"]

        self.T, self.B = self.cfg.rollout, env.B
        self.buf = Rollout(self.T, self.B, env.obs_shape, device, getattr(env.cfg, "n_proprio", 3))
        self.state = brain.init_state(self.B, device)
        self.blank = brain.init_state(self.B, device)
        self.obs = env.reset_all()
        self.prop = env.proprio()
        self.global_step = 0
        self.iteration = 0
        self.warmup_until = cfg.critic_warmup   # critic-only iterations, counted from the start (or the resume)
        self.ref = None                          # frozen reference policy (bc_anchor)
        self.history = {k: [] for k in (
            "step", "ep_return", "ep_length", "ep_distance", "crash_rate",
            "reward", "value_loss", "policy_loss", "entropy", "kl", "clipfrac", "anchor",
            "explained_var", "lr", "sps", "speed", "abs_lateral", "log_std",
            "t4_act", "t5_act", "hs_act", "ring_conc",
        )}
        self._ep_window = deque(maxlen=20)
        self._t_start = time.time()
        if hasattr(brain, "commit_norms"):
            self._warm_up_norms()

    def set_reference(self, state_dict):
        """Freeze a copy of the brain as the anchor for ``bc_anchor``.

        Fine-tuning a cloned policy with PPO alone throws the clone away within
        a few dozen iterations: the per-step advantages say "faster" long before
        they say "stay in lane".  Pulling the action mean toward what the clone
        would have done on the same observations keeps the expert's structure
        while the reward fixes what the clone got wrong (stalls, wrong turns).
        """
        import copy
        # Sparse (CSR) buffers -- the connectome itself -- cannot be deep-copied;
        # they are read-only, so the reference simply shares them.
        shared = []                       # (module name, attribute, tensor)
        for mod_name, mod in self.brain.named_modules():
            for attr, val in list(vars(mod).items()):
                if torch.is_tensor(val) and val.layout != torch.strided:
                    shared.append((mod_name, attr, val)); setattr(mod, attr, None)
            for attr, val in list(mod._buffers.items()):
                if val is not None and val.layout != torch.strided:
                    shared.append((mod_name, attr, val)); mod._buffers[attr] = None
        try:
            self.ref = copy.deepcopy(self.brain).eval()
        finally:
            for m in (self.brain, getattr(self, "ref", None)):
                if m is None:
                    continue
                for mod_name, attr, val in shared:
                    mod = m.get_submodule(mod_name) if mod_name else m
                    if attr in mod._buffers:
                        mod._buffers[attr] = val
                    else:
                        setattr(mod, attr, val)
        self.ref.load_state_dict(state_dict, strict=False)
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref_state = self.ref.init_state(self.B, self.device)
        self.ref_blank = self.ref.init_state(self.B, self.device)

    @torch.no_grad()
    def _warm_up_norms(self):
        """One throwaway rollout so the actor's input statistics, the critic's
        input statistics and the return scale all start from real data.

        Without it the first update runs against statistics snapped from a
        single batch and produces a KL of 1-6 -- one wasted, possibly harmful
        update per run, and a spike that pollutes every KL average.
        """
        self.collect()
        self.brain.commit_norms()
        self.global_step = 0
        self.state = self.brain.init_state(self.B, self.device)
        self.env.pop_stats()

    # -- collection --------------------------------------------------------

    @torch.no_grad()
    def collect(self):
        buf, env, brain = self.buf, self.env, self.brain
        self.h0 = self.state.clone()
        speeds, lats = [], []
        for t in range(self.T):
            buf.obs[t] = self.obs
            buf.proprio[t] = self.prop
            mu, log_std, value, self.state, _ = self.fwd(self.obs, self.prop, self.state)
            action = mu + log_std.exp() * torch.randn_like(mu)
            logp = brain._logp(mu, log_std, action)
            buf.actions[t] = action
            buf.logp[t] = logp
            buf.values[t] = value
            if self.ref is not None:
                # The reference is recurrent too: it follows the same
                # observations with its own hidden state.
                mu_ref, _, _, self.ref_state, _ = self.ref(self.obs, self.prop, self.ref_state)
                buf.mu_ref[t] = mu_ref

            self.obs, self.prop, reward, done, info = env.step(action)
            buf.rewards[t] = reward
            buf.dones[t] = done.float()
            # An environment that reset must start from a blank brain.
            if done.any():
                self.state.reset_(done, self.blank)
                if self.ref is not None:
                    self.ref_state.reset_(done, self.ref_blank)
            speeds.append(info["speed"].mean())
            lats.append(info["lateral"].abs().mean())

        _, _, last_value, _, _ = self.fwd(self.obs, self.prop, self.state)
        self.global_step += self.T * self.B
        self._compute_gae(last_value)
        self.state = self.state.detach()
        return (torch.stack(speeds).mean().item(), torch.stack(lats).mean().item())

    @torch.no_grad()
    def _compute_gae(self, last_value):
        cfg, buf = self.cfg, self.buf
        adv = torch.zeros_like(buf.rewards[0])
        for t in reversed(range(self.T)):
            nonterminal = 1.0 - buf.dones[t]
            next_value = last_value if t == self.T - 1 else buf.values[t + 1]
            delta = buf.rewards[t] + cfg.gamma * next_value * nonterminal - buf.values[t]
            adv = delta + cfg.gamma * cfg.gae_lambda * nonterminal * adv
            buf.adv[t] = adv
        buf.ret.copy_(buf.adv + buf.values)
        if hasattr(self.brain, "update_return_stats"):
            self.brain.update_return_stats(buf.ret)

    # -- update ------------------------------------------------------------

    def update(self):
        cfg, buf, brain = self.cfg, self.buf, self.brain
        adv = (buf.adv - buf.adv.mean()) / (buf.adv.std() + 1e-8)

        n_mb = cfg.n_minibatch
        mb_size = self.B // n_mb
        logs = {k: [] for k in ("value_loss", "policy_loss", "entropy", "kl", "clipfrac", "anchor")}
        stop = False
        # A policy cloned from an expert arrives with an untrained critic; letting
        # its garbage advantages drive the actor wrecks the clone in a few
        # iterations.  During the warm-up only the value head moves (the trunk is
        # shared with the actor, so its gradients are dropped too).
        warmup = self.iteration < self.warmup_until
        critic_ids = {id(p) for p in brain.value.parameters()} if warmup else set()
        beta = cfg.bc_anchor
        if cfg.bc_anchor_final is not None:      # let the policy leave the clone gradually
            frac = min(self.global_step / max(cfg.total_steps, 1), 1.0)
            beta = cfg.bc_anchor + (cfg.bc_anchor_final - cfg.bc_anchor) * frac
        self.anchor_weight = beta

        for _ in range(cfg.epochs):
            perm = torch.randperm(self.B, device=self.device)
            for m in range(n_mb):
                idx = perm[m * mb_size:(m + 1) * mb_size]
                state = self.h0.index(idx)
                blank = self.blank.index(idx)

                new_logp, new_val, ent, new_mu = [], [], [], []
                ckpt = getattr(brain, "grad_checkpoint", False)
                for t in range(self.T):
                    if ckpt:
                        # Recompute each step's activations during the backward
                        # pass instead of storing them.  A network the size of a
                        # whole nervous system produces ~12 tensors of
                        # (batch x neurons) per step, which over a 32-step
                        # segment does not fit; one extra forward pass does.
                        mu, log_std, v, state = self._ckpt_step(
                            buf.obs[t, idx], buf.proprio[t, idx], state)
                    else:
                        mu, log_std, v, state, _ = self.fwd(
                            buf.obs[t, idx], buf.proprio[t, idx], state)
                    new_logp.append(brain._logp(mu, log_std, buf.actions[t, idx]))
                    new_val.append(v)
                    new_mu.append(mu)
                    ent.append(brain.entropy(log_std))
                    d = buf.dones[t, idx].bool()
                    if d.any():
                        # Replay the resets exactly as they happened during
                        # collection.  Rebuild through type(state) so the same
                        # loop drives the hand-built brain and the whole-CNS
                        # model, whose state objects differ.
                        state = type(state)(*[
                            torch.where(d.view(-1, *([1] * (cur.dim() - 1))), new, cur)
                            for cur, new in zip(state.tensors(), blank.tensors())
                        ])
                new_logp = torch.stack(new_logp)
                new_val = torch.stack(new_val)
                entropy = torch.stack(ent).mean()

                ratio = (new_logp - buf.logp[:, idx]).exp()
                a = adv[:, idx]
                pg1 = -a * ratio
                pg2 = -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                policy_loss = torch.max(pg1, pg2).mean()
                # Measure the value error on the return's own scale, so a model
                # that normalises its head does not see the loss -- and with it
                # every other gradient, through the global clip -- grow as the
                # policy improves and returns get larger.
                vscale = float(getattr(brain, "ret_std", 1.0))
                value_loss = 0.5 * ((new_val - buf.ret[:, idx]) / vscale).pow(2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
                if self.ref is not None:
                    diff2 = (torch.stack(new_mu) - buf.mu_ref[:, idx]) ** 2
                    anchor = diff2.mean()                                     # logged: plain MSE to the clone
                    # The clone's steering is worth keeping; its throttle is what the
                    # reward has to fix, so it may be held more loosely.
                    anchor_loss = 0.5 * (diff2[..., 0].mean() + cfg.bc_anchor_throttle_scale * diff2[..., 1].mean())
                    loss = loss + beta * anchor_loss
                else:
                    anchor = torch.zeros(())
                if warmup:
                    loss = cfg.vf_coef * value_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if warmup:
                    for p in brain.parameters():
                        if id(p) not in critic_ids:
                            p.grad = None
                nn.utils.clip_grad_norm_(brain.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    logratio = new_logp - buf.logp[:, idx]
                    approx_kl = ((logratio.exp() - 1) - logratio).mean()
                    logs["kl"].append(approx_kl.item())
                    logs["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                logs["value_loss"].append(value_loss.item())
                logs["policy_loss"].append(policy_loss.item())
                logs["anchor"].append(anchor.item())
                logs["entropy"].append(entropy.item())

                if approx_kl.item() > cfg.target_kl * 1.5:
                    stop = True
                    break
            if stop:
                break

        # Only now may the actor's input statistics move: every forward pass of
        # this iteration, collection and replay alike, used the same ones.
        if hasattr(brain, "commit_norms"):
            brain.commit_norms()

        with torch.no_grad():
            var_y = buf.ret.var()
            ev = 1 - (buf.ret - buf.values).var() / (var_y + 1e-8)
        out = {k: float(sum(v) / max(len(v), 1)) for k, v in logs.items()}
        out["explained_var"] = ev.item()
        return out

    def _ckpt_step(self, obs, prop, state):
        cls = type(state)

        def run(img, pro, *tensors):
            mu, log_std, v, ns, _ = self.fwd(img, pro, cls(*tensors))
            return (mu, log_std, v) + tuple(ns.tensors())

        out = checkpoint(run, obs, prop, *state.tensors(), use_reentrant=False)
        return out[0], out[1], out[2], cls(*out[3:])

    # -- one full iteration -------------------------------------------------

    def anneal(self, frac_done: float):
        if self.cfg.anneal_lr:
            for g in self.opt.param_groups:
                g["lr"] = g["initial_lr"] * max(1.0 - frac_done, 0.05)

    def iterate(self):
        t0 = time.time()
        speed, lat = self.collect()
        stats = self.update()
        self.iteration += 1

        ep = self.env.pop_stats()
        if ep:
            self._ep_window.append(ep)
        window = self._ep_window[-1] if self._ep_window else None

        with torch.no_grad():
            neuro = self.probe()

        h = self.history
        h["step"].append(self.global_step)
        h["reward"].append(self.buf.rewards.mean().item())
        h["ep_return"].append(window["return"] if window else float("nan"))
        h["ep_length"].append(window["length"] if window else float("nan"))
        h["ep_distance"].append(window["distance"] if window else float("nan"))
        h["crash_rate"].append(window["crash"] if window else float("nan"))
        h["speed"].append(speed)
        h["abs_lateral"].append(lat)
        h["lr"].append(self.opt.param_groups[0]["lr"])
        h["sps"].append(self.T * self.B / (time.time() - t0))
        h["log_std"].append(self.brain.log_std.mean().item())
        for k, v in stats.items():
            h[k].append(v)
        for k, v in neuro.items():
            h[k].append(v)
        return stats

    @torch.no_grad()
    def probe(self):
        """Cheap read-out of what the network is doing right now."""
        _, _, _, _, tel = self.brain(self.obs, self.prop, self.state, telemetry=True)
        if "T4" not in tel:      # whole-CNS model: report population activity
            rate = tel["rate"]
            return {"t4_act": rate.mean().item(),
                    "t5_act": (rate > 0).float().mean().item(),
                    "hs_act": tel["dn"].abs().mean().item(),
                    "ring_conc": rate.std().item()}
        epg = tel["epg"]
        # Concentration of the heading bump: 1 = sharp, 0 = flat.
        conc = torch.linalg.norm(
            torch.stack([(epg * torch.cos(self.brain.epg_angles)).sum(-1),
                         (epg * torch.sin(self.brain.epg_angles)).sum(-1)], -1), dim=-1)
        return {
            "t4_act": tel["T4"].mean().item(),
            "t5_act": tel["T5"].mean().item(),
            "hs_act": tel["hs"].abs().mean().item(),
            "ring_conc": conc.mean().item(),
        }

    # -- checkpointing ------------------------------------------------------

    def save(self, path):
        torch.save({
            "brain": self.brain.state_dict(),
            "opt": self.opt.state_dict(),
            "history": self.history,
            "global_step": self.global_step,
            "iteration": self.iteration,
            "city_cfg": (asdict(self.env.ccfg) if hasattr(self.env, "ccfg") else None),
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        # Tolerate buffers added to the model after a run was started; they
        # keep their fresh defaults, which every such buffer is designed for.
        missing, unexpected = self.brain.load_state_dict(ck["brain"], strict=False)
        if missing or unexpected:
            print(f"  resume: missing {list(missing)}  unexpected {list(unexpected)}")
        if "opt" in ck:   # released checkpoints ship without it; Adam then restarts its moments
            self.opt.load_state_dict(ck["opt"])
        else:
            print("  resume: no optimizer state in checkpoint, starting Adam fresh")
        # Keep any history keys this version records that the checkpoint predates.
        hist = ck.get("history", {})
        self.history = {k: hist.get(k, []) for k in self.history} | {
            k: v for k, v in hist.items() if k not in self.history}
        self.global_step = ck.get("global_step", 0)
        self.iteration = ck.get("iteration", 0)
        self.warmup_until = self.iteration + self.cfg.critic_warmup
