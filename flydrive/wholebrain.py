"""The whole MaleCNS connectome run as the driving policy.

Every reconstructed neuron in the male central nervous system -- brain, optic
lobes, neck connective and ventral nerve cord -- is a unit; every reconstructed
synapse is an edge.  The adjacency matrix is *not* learned: weights are the
measured synapse counts and signs are the released neurotransmitter predictions.

What is learned is what a connectome cannot tell you:

  * a per-neuron membrane time constant, gain and bias,
  * one synaptic gain per presynaptic cell type,
  * how luminance enters the retina, and how the descending neurons are read
    out as steering and throttle.

About half a million free parameters against 24.4 million fixed synapses.  The
question the model asks is whether that wiring, by itself, is a better prior
than random sparse connectivity of the same size -- so the honest thing is to
run it and report what happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import connectome as cx
from .config import EnvConfig


class SpMM(torch.autograd.Function):
    """y = W @ h with a precomputed transpose, so the backward is one more spmm.

    Letting autograd transpose a 24-million-edge CSR matrix on every backward
    pass costs more than the multiplication itself.
    """

    @staticmethod
    def forward(ctx, h, W, Wt):
        ctx.Wt = Wt
        return W @ h

    @staticmethod
    def backward(ctx, g):
        return ctx.Wt @ g.contiguous(), None, None


class RunningNorm(nn.Module):
    """Per-feature standardisation with running statistics.

    LayerNorm normalises *within* a sample, so every descending neuron is left
    as a small fluctuation riding on its own large constant offset.  A linear
    readout can only recover the fluctuation by tuning its weights so the
    offsets cancel, which gradient descent finds extremely slowly -- ridge
    regression on the same, centred features reaches R^2 = 0.65 in closed form
    while the trained critic sat at 0.00.

    Two details matter here.  Descending-neuron rates are around 2.6e-3 with a
    per-feature variance near 4e-7, so a conventional ``eps`` of 1e-4 is 250x
    larger than the quantity it guards and turns the whole thing into a fixed
    scale factor.  And the running statistics track raw first and second
    moments rather than an average of within-batch variances, so drift between
    batches is counted.

    Statistics update during rollout collection only (which runs under
    ``no_grad``), never during the replayed update.
    """

    def __init__(self, n, momentum=0.05, clip=10.0, deferred=False):
        super().__init__()
        self.momentum, self.clip, self.deferred = momentum, clip, deferred
        self.frozen = False    # fine-tuning a fitted readout: keep the statistics it was fitted for
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("sq", torch.ones(n))
        self.register_buffer("started", torch.zeros((), dtype=torch.bool))
        if deferred:
            # Deferred mode is for the *actor*: its output enters the PPO
            # importance ratio, so the statistics used to normalise must be
            # identical during rollout collection and the replayed update.
            # New statistics accumulate here and are applied by commit(),
            # which PPO calls only between iterations.
            self.register_buffer("pend_mean", torch.zeros(n))
            self.register_buffer("pend_sq", torch.ones(n))

    def forward(self, x):
        if self.training and not torch.is_grad_enabled() and not self.frozen:
            with torch.no_grad():
                m, q = x.mean(0), (x * x).mean(0)
                if not bool(self.started):
                    # First batch ever: snap the active statistics directly so
                    # iteration 0 is already internally consistent.
                    self.mean.copy_(m); self.sq.copy_(q)
                    if self.deferred:
                        self.pend_mean.copy_(m); self.pend_sq.copy_(q)
                    self.started.fill_(True)
                else:
                    w = self.momentum
                    tm, tq = (self.pend_mean, self.pend_sq) if self.deferred else (self.mean, self.sq)
                    tm.mul_(1 - w).add_(m, alpha=w)
                    tq.mul_(1 - w).add_(q, alpha=w)
        var = (self.sq - self.mean * self.mean).clamp_min(1e-12)
        return ((x - self.mean) * torch.rsqrt(var)).clamp(-self.clip, self.clip)

    @torch.no_grad()
    def commit(self):
        if self.deferred:
            self.mean.copy_(self.pend_mean)
            self.sq.copy_(self.pend_sq)


@dataclass
class CNSState:
    v: torch.Tensor        # (B, N) membrane potential
    motor: torch.Tensor    # (B, 2) efference copy of the last command

    def detach(self):
        return CNSState(*[t.detach() for t in self.tensors()])

    def tensors(self):
        return (self.v, self.motor)

    def clone(self):
        return CNSState(*[t.clone() for t in self.tensors()])

    def reset_(self, mask, template):
        for cur, new in zip(self.tensors(), template.tensors()):
            m = mask.view(-1, *([1] * (cur.dim() - 1)))
            cur.copy_(torch.where(m, new, cur))

    def index(self, idx):
        return CNSState(*[t[idx] for t in self.tensors()])


def _axial_to_xy(h1, h2):
    """Axial hexagonal coordinates to Cartesian, for retinotopic resampling."""
    return h1 + 0.5 * h2, h2 * (math.sqrt(3.0) / 2.0)


class WholeCNS(nn.Module):
    """The MaleCNS connectome as a leaky rate network with a learned readout."""

    grad_checkpoint = True   # the activation tensors do not fit otherwise
    # With 1311 unit-variance inputs, one Adam step at the base rate moves every
    # motor weight by lr in a coherent direction and shifts the command by
    # lr * sum|z| ~ 0.3 (KL ~ 0.8).  KL scales with lr^2, so a tenth of the base
    # rate on the readout lands near the 0.005-0.01 the hand-built model trains
    # at, while the 162k single-neuron parameters keep the full rate.
    MOTOR_LR_SCALE = 0.1
    TRUNK_LR_SCALE = 1.0
    N_BUCKETS = 256          # population summary the critic reads
    R_MAX = 3.0        # saturating firing rate
    V_CLAMP = 20.0     # hard bound on the membrane potential

    def __init__(self, env_cfg: EnvConfig, conn: cx.Connectome | None = None,
                 tau_ms: float = 35.0, device="cuda", n_sub_steps: int = 1,
                 motor_lr_scale: float | None = None, trunk_lr_scale: float | None = None,
                 readout: str = "linear"):
        super().__init__()
        self.readout = readout
        if motor_lr_scale is not None:
            self.MOTOR_LR_SCALE = motor_lr_scale
        if trunk_lr_scale is not None:
            self.TRUNK_LR_SCALE = trunk_lr_scale
        self.env_cfg = env_cfg
        self.dt = env_cfg.dt
        self.n_sub = n_sub_steps
        self.c = conn if conn is not None else cx.load(verbose=False)
        c = self.c
        N = c.n
        self.N = N
        self.H, self.W_ = env_cfg.eye.n_row, env_cfg.eye.n_col

        # -- the fixed synaptic matrix -------------------------------------
        w = torch.from_numpy(c.weight.astype(np.float32))
        pre = torch.from_numpy(c.pre.astype(np.int64))
        post = torch.from_numpy(c.post.astype(np.int64))
        # Normalise each neuron's total input drive to 1.  Raw synapse counts
        # with a mean in-degree of 150 give the loop a gain of order 100, and
        # the network saturates or diverges within a few steps.  What survives
        # normalisation is the *relative* strength of each neuron's inputs,
        # which is the part the connectome actually measures.
        indeg = torch.zeros(N).index_add_(0, post, w.abs())
        w = w / indeg[post].clamp_min(1e-3)
        A = torch.sparse_coo_tensor(torch.stack([post, pre]), w, (N, N)).coalesce()
        At = torch.sparse_coo_tensor(torch.stack([pre, post]), w, (N, N)).coalesce()
        self.register_buffer("_dummy", torch.zeros(1))
        self.A = A.to_sparse_csr().to(device)
        self.At = At.to_sparse_csr().to(device)
        # Presynaptic cell type of every edge, for the per-type gain.
        types, type_id = np.unique(c.type, return_inverse=True)
        self.n_types = len(types)
        self.type_names = types
        self.register_buffer("neuron_type", torch.from_numpy(type_id.astype(np.int64)))

        # -- learnable single-neuron parameters ----------------------------
        alpha0 = math.exp(-self.dt / (tau_ms * 1e-3 * self.n_sub))
        self.alpha_raw = nn.Parameter(torch.full((N,), math.log(alpha0 / (1 - alpha0))))
        self.gain_raw = nn.Parameter(torch.zeros(N))
        self.bias = nn.Parameter(torch.zeros(N))
        self.type_gain_raw = nn.Parameter(torch.zeros(self.n_types))
        self.syn_scale = nn.Parameter(torch.tensor(0.0))

        # -- where light enters --------------------------------------------
        inj_idx, inj_uv, inj_eye, inj_sign = self._build_retina()
        self.register_buffer("inj_idx", inj_idx)         # (M,) neuron index
        self.register_buffer("inj_uv", inj_uv)           # (M, 2) in [-1, 1]
        self.register_buffer("inj_eye", inj_eye)         # (M,) 0 = left, 1 = right
        self.register_buffer("inj_sign", inj_sign)       # (M,) +1 R cells, -1 lamina
        self.in_gain = nn.Parameter(torch.tensor(2.5))
        self.in_bias = nn.Parameter(torch.tensor(0.0))

        # -- proprioception, injected into the ventral nerve cord -----------
        vnc = np.where(np.isin(c.superclass, ("vnc_sensory", "ascending_neuron")))[0]
        rng = np.random.default_rng(0)
        vnc = rng.choice(vnc, size=min(600, len(vnc)), replace=False)
        self.register_buffer("prop_idx", torch.from_numpy(vnc.astype(np.int64)))
        self.prop_w = nn.Parameter(torch.randn(3, len(vnc)) * 0.15)
        # Turn commands (left / straight / right) enter the central complex
        # through PFL3, the neurons that carry the fly's own steering goal.
        self.n_prop = env_cfg.n_proprio
        pfl3 = np.where(c.type == "PFL3")[0]
        self.register_buffer("pfl3_l", torch.from_numpy(pfl3[c.side[pfl3] == "L"].astype(np.int64)))
        self.register_buffer("pfl3_r", torch.from_numpy(pfl3[c.side[pfl3] == "R"].astype(np.int64)))
        self.cmd_gain = nn.Parameter(torch.tensor(1.5))

        # -- reading the descending neurons ---------------------------------
        dn = np.where(c.superclass == "descending_neuron")[0]
        self.register_buffer("dn_idx", torch.from_numpy(dn.astype(np.int64)))
        self.n_dn = len(dn)
        # The actor reads per-feature-centred descending-neuron rates.  With a
        # LayerNorm instead, a linear readout recovers lane position at R^2
        # 0.34; centred, 0.83 -- the same conditioning failure that silenced
        # the critic.  Deferred statistics keep the PPO ratio exact.
        self.dn_norm = RunningNorm(self.n_dn, deferred=True)
        self.dn_scale = 1.0
        self.motor = nn.Linear(self.n_dn, 2)
        # The readout is the one learned map from the frozen brain to the wheel.
        # "linear" is the cleanest claim; "mlp" keeps the brain frozen but lets
        # the readout resolve what a linear map of firing rates cannot, such
        # as "a red lamp ahead" combined with "still moving".
        if readout == "mlp":
            self.motor = nn.Sequential(nn.Linear(self.n_dn, 128), nn.Tanh(), nn.Linear(128, 2))

        # The critic is a training device, not part of the fly, so it is allowed
        # a wider view than the motor pathway.  Reading only the descending
        # neurons, it collapses to predicting the mean return: after LayerNorm
        # almost all of their variance is a fixed across-unit pattern and only
        # about a tenth of it varies over time, which the critic never finds.
        # Pooling the whole population into fixed random buckets gives it a
        # broad, low-dimensional view of brain state instead.
        rng = np.random.default_rng(1234)
        buckets = rng.integers(0, self.N_BUCKETS, size=N)
        counts = np.bincount(buckets, minlength=self.N_BUCKETS).astype(np.float32)
        self.register_buffer("bucket_idx", torch.from_numpy(buckets.astype(np.int64)))
        self.register_buffer("bucket_norm", torch.from_numpy(1.0 / np.maximum(counts, 1)))
        n_critic = self.n_dn + self.N_BUCKETS + env_cfg.n_proprio
        # The critic does not enter the ratio, so it can use running statistics.
        self.critic_norm = RunningNorm(n_critic)
        # The head predicts a standardised return and is rescaled afterwards.
        # Once the critic's inputs are centred it has no constant component to
        # amplify, so an untransformed head can only walk its output bias
        # towards the mean return at the optimiser's step size -- reaching a
        # mean of 17 at lr 1e-3 would take tens of thousands of updates.  The
        # return scale also drifts upwards as the policy improves.
        self.register_buffer("ret_mean", torch.zeros(()))
        self.register_buffer("ret_std", torch.ones(()))
        self.register_buffer("ret_started", torch.zeros((), dtype=torch.bool))
        self.value = nn.Sequential(
            nn.Linear(n_critic, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.9))
        last = self.motor[-1] if isinstance(self.motor, nn.Sequential) else self.motor
        nn.init.orthogonal_(last.weight, gain=0.1)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.0, 0.7]))

    # -- construction helpers ---------------------------------------------

    def _build_retina(self):
        """Pick the neurons that receive light, and where each one looks.

        Photoreceptors are used wherever the release traced them; the retina is
        largely outside the imaged volume, so for every remaining column the
        light is injected into the lamina monopolar cells instead, with the
        inverted sign the histaminergic photoreceptors would have applied.
        """
        c = self.c
        cols = cx.photoreceptor_columns(c)
        is_r = c.type == "R1-R6"
        is_lam = np.isin(c.type, ("L1", "L2", "L3"))
        have = np.isfinite(cols[:, 0]) & np.isfinite(cols[:, 1])

        r_ok = is_r & have
        covered = set(zip(cols[r_ok, 0].tolist(), cols[r_ok, 1].tolist(), c.side[r_ok].tolist()))
        lam_ok = is_lam & have
        lam_keys = list(zip(cols[lam_ok, 0].tolist(), cols[lam_ok, 1].tolist(),
                            c.side[lam_ok].tolist()))
        lam_sel = np.array([k not in covered for k in lam_keys], dtype=bool)
        lam_idx = np.where(lam_ok)[0][lam_sel] if len(lam_keys) else np.array([], int)

        idx = np.concatenate([np.where(r_ok)[0], lam_idx]).astype(np.int64)
        sign = np.concatenate([np.ones(int(r_ok.sum())), -np.ones(len(lam_idx))]).astype(np.float32)

        x, y = _axial_to_xy(cols[idx, 0], cols[idx, 1])
        # Normalise each eye's column map onto the rendered image.
        eye = (c.side[idx] == "R").astype(np.int64)
        u = np.zeros_like(x); v = np.zeros_like(y)
        for e in (0, 1):
            m = eye == e
            if m.sum() < 4:
                continue
            u[m] = 2 * (x[m] - x[m].min()) / max(x[m].ptp(), 1e-6) - 1
            v[m] = 2 * (y[m] - y[m].min()) / max(y[m].ptp(), 1e-6) - 1
        return (torch.from_numpy(idx), torch.from_numpy(np.stack([u, v], 1).astype(np.float32)),
                torch.from_numpy(eye), torch.from_numpy(sign))

    # -- state -------------------------------------------------------------

    def init_state(self, batch, device=None, dtype=torch.float32):
        device = device or self.bias.device
        return CNSState(v=torch.zeros(batch, self.N, device=device, dtype=dtype),
                        motor=torch.zeros(batch, 2, device=device, dtype=dtype))

    # -- forward -----------------------------------------------------------

    def _inject(self, image):
        """Sample the rendered image at each light-sensitive neuron's column."""
        B = image.shape[0]
        M = self.inj_idx.shape[0]
        # grid_sample wants (B, C, H, W) and a (B, Hout, Wout, 2) grid.
        grid = self.inj_uv.view(1, 1, M, 2).expand(B, 1, M, 2)
        both = []
        for e in (0, 1):
            s = F.grid_sample(image[:, e:e + 1], grid, mode="bilinear",
                              padding_mode="border", align_corners=True)
            both.append(s.view(B, M))
        lum = torch.where(self.inj_eye.view(1, M).bool(), both[1], both[0])
        return (lum - 0.5) * self.inj_sign.view(1, M) * F.softplus(self.in_gain) + self.in_bias

    def forward(self, image, proprio, state: CNSState, telemetry: bool = False):
        B = image.shape[0]
        alpha = torch.sigmoid(self.alpha_raw)
        scale = F.softplus(self.syn_scale + 0.5413)
        edge_gain = F.softplus(self.type_gain_raw + 0.5413)[self.neuron_type]

        drive = torch.zeros(B, self.N, device=image.device, dtype=image.dtype)
        drive.index_add_(1, self.inj_idx, self._inject(image))
        drive.index_add_(1, self.prop_idx, proprio[:, :3] @ self.prop_w)
        if self.n_prop > 3 and self.pfl3_l.numel() and self.pfl3_r.numel():
            cmd = proprio[:, 3:6]; g = F.softplus(self.cmd_gain)
            drive.index_add_(1, self.pfl3_l, ((cmd[:, 0:1] + 0.5 * cmd[:, 1:2]) * g).expand(-1, self.pfl3_l.numel()))
            drive.index_add_(1, self.pfl3_r, ((cmd[:, 2:3] + 0.5 * cmd[:, 1:2]) * g).expand(-1, self.pfl3_r.numel()))

        v = state.v
        for _ in range(self.n_sub):
            r = self._rate(v) * edge_gain
            syn = SpMM.apply(r.t().contiguous(), self.A, self.At).t() * scale
            v = alpha * v + (1 - alpha) * (syn + drive + self.bias)
            v = v.clamp(-self.V_CLAMP, self.V_CLAMP)

        rate = self._rate(v)
        dn = self.dn_norm(rate[:, self.dn_idx]) * self.dn_scale
        mu = torch.tanh(self.motor(dn))

        pooled = torch.zeros(B, self.N_BUCKETS, device=rate.device, dtype=rate.dtype)
        pooled = pooled.index_add(1, self.bucket_idx, rate) * self.bucket_norm
        crit = self.critic_norm(
            torch.cat([rate[:, self.dn_idx], pooled, proprio], dim=1))
        value = self.value(crit).squeeze(-1) * self.ret_std + self.ret_mean
        new_state = CNSState(v=v, motor=mu)

        tel = None
        if telemetry:
            tel = {"rate": rate, "dn": dn, "mu": mu, "value": value, "v": v,
                   "image": image}
        return mu, self.log_std.expand_as(mu), value, new_state, tel

    # -- helpers shared with the hand-built model --------------------------

    def act(self, image, proprio, state, deterministic=False, telemetry=False):
        mu, log_std, value, new_state, tel = self(image, proprio, state, telemetry)
        action = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        return action, self._logp(mu, log_std, action), value, new_state, tel

    def param_groups(self, lr):
        """Optimizer groups: the motor readout learns slower than the brain."""
        head = list(self.motor.parameters()) + [self.log_std]
        critic = list(self.value.parameters())
        skip = {id(p) for p in head + critic}
        # "Trunk": the 162k single-neuron parameters, the per-type synaptic
        # gains and the input maps -- everything that changes what the brain
        # computes.  Centred descending-neuron features turn any coherent
        # shift here into an O(1) change of the command, so it needs its own,
        # usually much smaller, rate (0 freezes the brain and trains only the
        # readouts, which is the cleanest test of what the wiring provides).
        trunk = [p for p in self.parameters() if id(p) not in skip]
        return [{"params": trunk, "lr": lr * self.TRUNK_LR_SCALE},
                {"params": critic, "lr": lr},
                {"params": head, "lr": lr * self.MOTOR_LR_SCALE}]

    @torch.no_grad()
    def commit_norms(self):
        """Apply the actor's pending input statistics; PPO calls this between iterations."""
        self.dn_norm.commit()

    @torch.no_grad()
    def update_return_stats(self, ret, momentum=0.05):
        """Track the scale of the GAE return so the value head can stay O(1).

        The last layer is rescaled so the value output is *unchanged* by the new
        statistics (PopArt).  Without that step the update is a positive
        feedback loop -- GAE builds its return out of the value estimate, so a
        larger scale produces larger returns, which enlarge the scale again; the
        statistics ran away from 0.8 to 9.7 within eight iterations and the
        exploding value loss then starved the policy gradient through gradient
        clipping.
        """
        old_mean, old_std = self.ret_mean.clone(), self.ret_std.clone()
        # Own flag: critic_norm.started is already set by the first forward
        # pass, so borrowing it would make even the first return update an EMA
        # step and leave the head predicting around zero for a while.
        w = momentum if bool(self.ret_started) else 1.0
        self.ret_started.fill_(True)
        self.ret_mean.mul_(1 - w).add_(ret.mean(), alpha=w)
        self.ret_std.mul_(1 - w).add_(ret.std().clamp_min(1e-3), alpha=w)
        ratio = old_std / self.ret_std
        last = self.value[-1]
        last.weight.mul_(ratio)
        last.bias.mul_(ratio).add_((old_mean - self.ret_mean) / self.ret_std)

    def _rate(self, v):
        """Saturating rectifier: linear near threshold, capped at a maximum
        firing rate, which is what keeps a 24-million-synapse recurrent loop
        from running away."""
        x = F.relu(F.softplus(self.gain_raw + 0.5413) * v)
        return self.R_MAX * torch.tanh(x / self.R_MAX)

    @staticmethod
    def _logp(mu, log_std, action):
        var = torch.exp(2 * log_std)
        return (-0.5 * ((action - mu) ** 2 / var + 2 * log_std + math.log(2 * math.pi))).sum(-1)

    @staticmethod
    def entropy(log_std):
        return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def describe(self):
        return (f"WholeCNS: {self.N:,} neurons, {self.c.n_edges:,} fixed synapses, "
                f"{self.n_params():,} learnable parameters\n"
                f"  light enters at {self.inj_idx.shape[0]:,} neurons "
                f"({int((self.inj_sign > 0).sum()):,} photoreceptors, "
                f"{int((self.inj_sign < 0).sum()):,} lamina)\n"
                f"  read out from {self.n_dn:,} descending neurons")


def readout_kind(state_dict) -> str:
    """Which readout a saved WholeCNS used (an MLP has a second motor layer)."""
    return "mlp" if any(k.startswith("motor.2.") for k in state_dict) else "linear"


def load_compatible(brain, state_dict):
    """Load a checkpoint, skipping motor weights whose shapes do not match."""
    own = brain.state_dict()
    keep = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    return brain.load_state_dict(keep, strict=False)
