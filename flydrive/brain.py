"""A connectome-structured model of the Drosophila visual-motor pathway.

The architecture -- which cell types exist, what each one reads from, and where
its inputs sit on the retinotopic lattice -- is taken from the fly literature.
The *weights* are learned, which is the same compromise Lappalainen et al.
(2024) make with flyvis: the connectome is a strong structural prior, not a set
of trained parameters, because it records connections and synapse counts but no
synaptic gains and no plasticity rule.

    R1-R6 photoreceptors            luminance, with local contrast adaptation
      |
    lamina      L1 (ON), L2 (OFF), L3 (sustained)
      |
    medulla     Mi1, Tm3, Mi4, Mi9   (ON)     each a first-order temporal
                Tm1, Tm2, Tm9, CT1   (OFF)    filter with its own time constant
      |
    T4 a/b/c/d (ON) and T5 a/b/c/d (OFF)
                three-arm elementary motion detectors: fast centre input plus
                two delayed, spatially offset arms, one enhancing the preferred
                direction and one suppressing the null direction
      |
    lobula plate tangential cells    HS x3 and VS x10 per eye, wide-field
                                     integrators of the T4/T5 array
    lobula columnar pool             coarse retinotopic features -- this is the
                                     positional channel; HS/VS only see motion
      |
    central complex                  EPG ring attractor (heading), PFL3 (steering)
      |
    descending neurons               DNa01/DNa02-like premotor pool
      |
    steering and throttle

Subtype ``a`` is front-to-back on the lattice in *both* eyes, because the right
eye is built as the mirror image of the left (see :mod:`flydrive.eye`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BrainConfig, EnvConfig

# Lattice offsets (drow, dcol) of the preferred direction of each subtype.
SUBTYPE_OFFSETS = {
    "a": (0, 1),    # front-to-back (regressive)
    "b": (0, -1),   # back-to-front (progressive)
    "c": (1, 0),    # upward
    "d": (-1, 0),   # downward
}
SUBTYPES = ("a", "b", "c", "d")
HORIZONTAL = (0, 1)   # indices of subtypes a, b
VERTICAL = (2, 3)     # indices of subtypes c, d


def shift2d(x: torch.Tensor, drow: int, dcol: int) -> torch.Tensor:
    """Shift a retinotopic map by one lattice step, replicating at the border.

    ``shift2d(x, 0, 1)[..., r, c]`` is ``x[..., r, c + 1]``: the value one step
    towards the back of the eye.
    """
    if drow == 0 and dcol == 0:
        return x
    lead, (H, W) = x.shape[:-2], x.shape[-2:]
    flat = x.reshape(-1, 1, H, W)
    padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
    r0, c0 = 1 + drow, 1 + dcol
    out = padded[..., r0:r0 + H, c0:c0 + W]
    return out.reshape(*lead, H, W)


def alpha_from_tau(tau: float, dt: float) -> float:
    """Leak coefficient of a first-order filter with time constant ``tau``."""
    return math.exp(-dt / max(tau, 1e-4))


def inv_sigmoid(p: float) -> float:
    p = min(max(p, 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------------


@dataclass
class BrainState:
    """Everything the network carries from one timestep to the next."""

    adapt: torch.Tensor      # (B, 2, H, W)   photoreceptor adaptation level
    lamina: torch.Tensor     # (B, 2, H, W)   lamina high-pass reference
    medulla: torch.Tensor    # (B, 8, 2, H, W) one filter state per medulla cell
    epg: torch.Tensor        # (B, n_epg)     ring-attractor bump
    premotor: torch.Tensor   # (B, n_hidden)  premotor pool
    motor: torch.Tensor      # (B, 2)         efference copy of the last action

    def detach(self) -> "BrainState":
        return BrainState(*[t.detach() for t in self.tensors()])

    def tensors(self):
        return (self.adapt, self.lamina, self.medulla, self.epg,
                self.premotor, self.motor)

    def clone(self) -> "BrainState":
        return BrainState(*[t.clone() for t in self.tensors()])

    def reset_(self, mask: torch.Tensor, template: "BrainState"):
        """In-place reset of the environments selected by ``mask`` (B,)."""
        for cur, new in zip(self.tensors(), template.tensors()):
            m = mask.view(-1, *([1] * (cur.dim() - 1)))
            cur.copy_(torch.where(m, new, cur))

    def index(self, idx) -> "BrainState":
        return BrainState(*[t[idx] for t in self.tensors()])


class FlyBrain(nn.Module):
    """The optic lobe, central complex and premotor stages as one policy."""

    # (name, time constant in seconds).  Fast cells carry the transient, slow
    # cells provide the delay line the motion detectors correlate against.
    MEDULLA = (
        ("Mi1", 0.025), ("Tm3", 0.045), ("Mi4", 0.120), ("Mi9", 0.160),   # ON
        ("Tm2", 0.022), ("Tm1", 0.035), ("CT1", 0.150), ("Tm9", 0.200),   # OFF
    )
    N_LOBULA_CH = 8
    N_AZ_BANDS = 9

    def __init__(self, env_cfg: EnvConfig, cfg: BrainConfig | None = None):
        super().__init__()
        self.cfg = cfg or BrainConfig()
        self.env_cfg = env_cfg
        self.dt = env_cfg.dt
        H, W = env_cfg.eye.n_row, env_cfg.eye.n_col
        self.H, self.W = H, W
        c = self.cfg

        # -- photoreceptor adaptation and lamina --------------------------
        self.adapt_alpha = nn.Parameter(torch.tensor(inv_sigmoid(alpha_from_tau(0.60, self.dt))))
        self.lamina_alpha = nn.Parameter(torch.tensor(inv_sigmoid(alpha_from_tau(0.070, self.dt))))
        self.lamina_gain = nn.Parameter(torch.tensor(4.0))
        # L3 is the sustained, luminance-encoding lamina cell (Ketkar et al.
        # 2020): besides contrast it passes absolute brightness on to the
        # object pathway.  Only the lobula columnar pool receives it -- the
        # motion pathway keeps its contrast input, so T4/T5 tuning is unchanged.
        # Without it the photoreceptor adaptation (tau 0.6 s) erases a static
        # traffic lamp within two seconds and a stopped car is blind to it.
        self.l3_lum_gain = nn.Parameter(torch.tensor(0.0))   # starts silent, so older checkpoints behave as trained

        # -- medulla: one learnable time constant per cell type ------------
        self.medulla_names = [n for n, _ in self.MEDULLA]
        taus = torch.tensor([inv_sigmoid(alpha_from_tau(t, self.dt)) for _, t in self.MEDULLA])
        self.medulla_alpha = nn.Parameter(taus)
        # Each medulla cell reads a weighted mix of L1/L2/L3.  ON cells start
        # dominated by L1, OFF cells by L2.
        mix = torch.zeros(len(self.MEDULLA), 3)
        mix[0:3, 0] = 1.0     # Mi1, Tm3, Mi4  <- L1 (ON)
        mix[3, 1] = 1.0       # Mi9 is sign-inverting: OFF-driven
        mix[3, 2] = 0.3       # ... plus a sustained component
        mix[4:7, 1] = 1.0     # Tm2, Tm1, CT1  <- L2 (OFF)
        mix[7, 0] = 1.0       # Tm9 is the sign-inverting arm of the OFF pathway
        mix[7, 2] = -0.3
        self.medulla_mix = nn.Parameter(mix + 0.02 * torch.randn_like(mix))

        # -- T4/T5: three arms per subtype, weights kept positive -----------
        # centre (fast), preferred-side delayed, null-side delayed
        w0 = torch.tensor([[1.0, 1.0, 0.3]]).repeat(4, 1)
        self.t4_w = nn.Parameter(torch.log(torch.expm1(w0)))   # inverse softplus
        self.t5_w = nn.Parameter(torch.log(torch.expm1(w0.clone())))
        self.t4_bias = nn.Parameter(torch.zeros(4))
        self.t5_bias = nn.Parameter(torch.zeros(4))
        self.t_gain = nn.Parameter(torch.tensor(8.0))

        # -- lobula plate tangential cells ---------------------------------
        # Weight maps are shared between the eyes: the right eye is the mirror
        # of the left, so the same map describes the same anatomical cell.
        self.hs_w = nn.Parameter(self._init_hs())    # (n_hs, 4, H, W)
        self.vs_w = nn.Parameter(self._init_vs())    # (n_vs, 4, H, W)
        self.n_lptc = 2 * (c.n_hs + c.n_vs)

        # -- lobula columnar pool: the positional channel -------------------
        self.lobula = nn.Conv2d(3, self.N_LOBULA_CH, kernel_size=3, padding=1)
        self.n_lc = 2 * self.N_LOBULA_CH * self.N_AZ_BANDS

        # -- central complex ------------------------------------------------
        n_epg = c.n_epg
        self.er_proj = nn.Linear(self.n_lc, n_epg)           # ring neurons
        n_prop = env_cfg.n_proprio
        self.n_prop = n_prop
        self.omega_proj = nn.Linear(self.n_lptc + n_prop, 1)   # HS-driven turn estimate
        self.ring_rec = nn.Parameter(self._init_mexican_hat())
        self.ring_beta = nn.Parameter(torch.tensor(1.2))
        self.register_buffer("epg_angles", torch.arange(n_epg) * (2 * math.pi / n_epg))
        self.register_buffer("ring_k", torch.arange(n_epg // 2 + 1).float())
        self.goal_proj = nn.Linear(self.n_lptc + self.n_lc + n_prop, 2)
        self.pfl3_offset = nn.Parameter(torch.tensor([math.pi / 4, -math.pi / 4]))

        # -- premotor pool and descending neurons ---------------------------
        n_feat = self.n_lptc + self.n_lc + 2 + 2 + 1 + n_prop
        self.n_feat = n_feat
        self.pre_in = nn.Linear(n_feat, c.n_hidden)
        self.pre_rec = nn.Linear(c.n_hidden, c.n_hidden, bias=False)
        self.pre_norm = nn.LayerNorm(c.n_hidden)
        self.dn = nn.Linear(c.n_hidden, c.n_dn)
        self.motor = nn.Linear(c.n_dn, 2)
        self.value = nn.Sequential(nn.Linear(c.n_hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((2,), c.log_std_init))

        nn.init.orthogonal_(self.pre_rec.weight, gain=0.6)
        # A very small head gain is standard in PPO, but it also scales down the
        # gradient reaching the entire optic lobe behind it.
        nn.init.orthogonal_(self.motor.weight, gain=0.1)
        # Start the throttle open.  Braking is stronger than acceleration, so a
        # policy centred on zero throttle decelerates in expectation and the
        # agent never sees what driving fast is worth.
        with torch.no_grad():
            self.motor.bias.copy_(torch.tensor([0.0, 0.7]))

    # -- initialisers ------------------------------------------------------

    def _init_hs(self) -> torch.Tensor:
        """HSN / HSE / HSS: three dorsoventral bands, preferring front-to-back."""
        H, W = self.H, self.W
        n = self.cfg.n_hs
        rows = torch.linspace(0, 1, H).view(H, 1)
        w = torch.zeros(n, 4, H, W)
        centres = torch.linspace(0.78, 0.22, n)          # dorsal -> ventral
        for i, ctr in enumerate(centres):
            band = torch.exp(-0.5 * ((rows - ctr) / 0.22) ** 2).expand(H, W)
            w[i, 0] = band          # T4a  (front-to-back, preferred)
            w[i, 1] = -band         # T4b  (back-to-front, null)
            w[i, 2] = band          # T5a
            w[i, 3] = -band         # T5b
        return w / (H * W) ** 0.5 + 0.01 * torch.randn(n, 4, H, W)

    def _init_vs(self) -> torch.Tensor:
        """VS1..VS10: overlapping azimuth bands, preferring downward motion."""
        H, W = self.H, self.W
        n = self.cfg.n_vs
        cols = torch.linspace(0, 1, W).view(1, W)
        w = torch.zeros(n, 4, H, W)
        centres = torch.linspace(0.05, 0.95, n)          # frontal -> lateral
        for i, ctr in enumerate(centres):
            band = torch.exp(-0.5 * ((cols - ctr) / 0.13) ** 2).expand(H, W)
            w[i, 0] = -band         # T4c  (upward, null)
            w[i, 1] = band          # T4d  (downward, preferred)
            w[i, 2] = -band         # T5c
            w[i, 3] = band          # T5d
        return w / (H * W) ** 0.5 + 0.01 * torch.randn(n, 4, H, W)

    def _init_mexican_hat(self) -> torch.Tensor:
        """Local excitation, global inhibition -- the classic ring-attractor kernel."""
        n = self.cfg.n_epg
        k = torch.arange(n)
        d = torch.minimum(k, n - k).float() * (2 * math.pi / n)
        return 1.6 * torch.exp(-0.5 * (d / 0.55) ** 2) - 0.55

    # -- state -------------------------------------------------------------

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> BrainState:
        device = device or next(self.parameters()).device
        H, W, c = self.H, self.W, self.cfg
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        epg = torch.full((batch, c.n_epg), 1.0 / c.n_epg, device=device, dtype=dtype)
        return BrainState(
            adapt=z(batch, 2, H, W) + 0.5,
            lamina=z(batch, 2, H, W) + 0.5,
            medulla=z(batch, len(self.MEDULLA), 2, H, W),
            epg=epg,
            premotor=z(batch, c.n_hidden),
            motor=z(batch, 2),
        )

    # -- forward -----------------------------------------------------------

    def forward(self, image, proprio, state: BrainState, telemetry: bool = False):
        """Run one 20 ms step of the whole pathway.

        Args:
            image: ``(B, 2, H, W)`` ommatidial luminance in [0, 1].
            proprio: ``(B, 3)`` speed, lateral accel, yaw rate -- the haltere and
                leg-proprioceptor analogue.
            state: previous :class:`BrainState`.
        Returns:
            ``(mu, log_std, value, new_state, telemetry_dict)``.
        """
        B = image.shape[0]
        H, W = self.H, self.W
        tel = {} if telemetry else None

        # -- photoreceptors: slow adaptation, then local contrast -----------
        a_ad = torch.sigmoid(self.adapt_alpha)
        adapt = a_ad * state.adapt + (1 - a_ad) * image
        contrast = (image - adapt) / (adapt + 0.25)

        # -- lamina: L1 (ON), L2 (OFF), L3 (sustained) ----------------------
        a_lam = torch.sigmoid(self.lamina_alpha)
        lam_ref = a_lam * state.lamina + (1 - a_lam) * contrast
        transient = (contrast - lam_ref) * self.lamina_gain
        # Half-wave rectification splits the signal into the ON and OFF
        # channels.  This has to pass through the origin: a softplus would
        # leave a constant pedestal that swamps the motion correlations
        # downstream.
        L1 = F.relu(transient)
        L2 = F.relu(-transient)
        L3 = contrast
        lam = torch.stack([L1, L2, L3], dim=2)               # (B, 2, 3, H, W)

        # -- medulla: one leaky filter per cell type ------------------------
        mix = self.medulla_mix.view(1, 1, len(self.MEDULLA), 3, 1, 1)
        drive = (lam.unsqueeze(2) * mix).sum(3)              # (B, 2, 8, H, W)
        drive = drive.permute(0, 2, 1, 3, 4)                 # (B, 8, 2, H, W)
        a_med = torch.sigmoid(self.medulla_alpha).view(1, -1, 1, 1, 1)
        med = a_med * state.medulla + (1 - a_med) * drive
        Mi1, Tm3, Mi4, Mi9, Tm2, Tm1, CT1, Tm9 = med.unbind(1)

        # -- T4 / T5: three-arm correlators ---------------------------------
        t4_w = F.softplus(self.t4_w)
        t5_w = F.softplus(self.t5_w)
        gain = F.softplus(self.t_gain)
        t4, t5 = [], []
        for i, s in enumerate(SUBTYPES):
            dr, dc = SUBTYPE_OFFSETS[s]
            # "Upstream" is where an edge travelling in the preferred direction
            # was one lattice step ago; "downstream" is where it is heading.
            # Correlating the fast centre signal with the *delayed* upstream
            # signal is the Hassenstein-Reichardt multiplication, and the
            # mirror-image product is subtracted so the null direction cancels.
            # The third, sign-inverting arm suppresses null-direction motion.
            on_pd = Mi1 * shift2d(Mi4, -dr, -dc)
            on_nd = shift2d(Mi1, -dr, -dc) * Mi4
            on_sup = Mi1 * shift2d(Mi9, dr, dc)
            on = (t4_w[i, 0] * on_pd - t4_w[i, 1] * on_nd
                  - t4_w[i, 2] * on_sup - self.t4_bias[i])

            off_pd = Tm2 * shift2d(CT1, -dr, -dc)
            off_nd = shift2d(Tm2, -dr, -dc) * CT1
            off_sup = Tm2 * shift2d(Tm9, dr, dc)
            off = (t5_w[i, 0] * off_pd - t5_w[i, 1] * off_nd
                   - t5_w[i, 2] * off_sup - self.t5_bias[i])

            # Hard rectification.  Anything with a soft floor puts a constant
            # pedestal on every cell, and the pedestal is far larger than the
            # directional part of the signal.
            t4.append(F.relu(on * gain))
            t5.append(F.relu(off * gain))
        T4 = torch.stack(t4, dim=1)                          # (B, 4, 2, H, W)
        T5 = torch.stack(t5, dim=1)

        # -- lobula plate: HS (horizontal) and VS (vertical) ----------------
        horiz = torch.cat([T4[:, HORIZONTAL[0]:HORIZONTAL[1] + 1],
                           T5[:, HORIZONTAL[0]:HORIZONTAL[1] + 1]], dim=1)   # (B,4,2,H,W)
        vert = torch.cat([T4[:, VERTICAL[0]:VERTICAL[1] + 1],
                          T5[:, VERTICAL[0]:VERTICAL[1] + 1]], dim=1)
        hs = torch.einsum("bcehw,ncwh->ben", horiz, self.hs_w.permute(0, 1, 3, 2))
        vs = torch.einsum("bcehw,ncwh->ben", vert, self.vs_w.permute(0, 1, 3, 2))
        lptc = torch.cat([hs, vs], dim=2).reshape(B, -1)     # (B, 2*(n_hs+n_vs))
        lptc = torch.tanh(lptc)

        # -- lobula columnar pool: coarse retinotopic position --------------
        L3_lum = L3 + self.l3_lum_gain * (image - 0.5)
        lam_flat = torch.stack([L1, L2, L3_lum], dim=2).reshape(B * 2, 3, H, W)
        lob = F.relu(self.lobula(lam_flat))
        lob = F.adaptive_avg_pool2d(lob, (1, self.N_AZ_BANDS))
        lc = lob.reshape(B, self.n_lc)

        # -- central complex: EPG ring attractor ----------------------------
        turn_in = torch.cat([lptc, proprio], dim=1)
        omega = torch.tanh(self.omega_proj(turn_in))                     # (B, 1)
        phi = omega * self.dt * self.cfg.ring_gain                       # bins to rotate
        epg_sh = self._circular_shift(state.epg, phi)
        rec = self._ring_conv(epg_sh)
        er = self.er_proj(lc)
        epg = torch.softmax((rec + er) * F.softplus(self.ring_beta), dim=-1)

        bump_cos = (epg * torch.cos(self.epg_angles)).sum(-1, keepdim=True)
        bump_sin = (epg * torch.sin(self.epg_angles)).sum(-1, keepdim=True)

        # -- PFL3: compare the heading bump against a goal direction --------
        goal = self.goal_proj(torch.cat([lptc, lc, proprio], dim=1))
        goal_ang = torch.atan2(goal[:, 1:2], goal[:, 0:1])
        delta = self.epg_angles.view(1, -1) - goal_ang
        pfl3 = torch.stack([
            (epg * torch.cos(delta + self.pfl3_offset[0])).sum(-1),
            (epg * torch.cos(delta + self.pfl3_offset[1])).sum(-1),
        ], dim=1)                                                        # (B, 2)

        # -- premotor pool and descending neurons ---------------------------
        feat = torch.cat([lptc, lc, pfl3, bump_cos, bump_sin, omega, proprio], dim=1)
        pre = torch.tanh(self.pre_norm(self.pre_in(feat) + self.pre_rec(state.premotor)))
        dn = torch.tanh(self.dn(pre))
        mu = torch.tanh(self.motor(dn))
        value = self.value(pre).squeeze(-1)

        new_state = BrainState(
            adapt=adapt, lamina=lam_ref, medulla=med, epg=epg,
            premotor=pre, motor=mu,
        )

        if telemetry:
            tel.update(
                image=image, contrast=contrast, L1=L1, L2=L2,
                T4=T4, T5=T5, hs=hs, vs=vs, lptc=lptc, lc=lc,
                epg=epg, omega=omega, pfl3=pfl3, goal_ang=goal_ang,
                dn=dn, mu=mu, value=value,
            )
        return mu, self.log_std.expand_as(mu), value, new_state, tel

    # -- ring-attractor helpers -------------------------------------------

    def _circular_shift(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Rotate the bump by a fractional number of wedges, via the FFT."""
        n = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        phase = torch.exp(-2j * math.pi * self.ring_k.view(1, -1) * bins / n)
        return torch.fft.irfft(X * phase, n=n, dim=-1)

    def _ring_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Circular convolution with the learned recurrent kernel."""
        n = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        Kf = torch.fft.rfft(self.ring_rec, dim=-1).view(1, -1)
        return torch.fft.irfft(X * Kf, n=n, dim=-1)

    # -- action helpers ----------------------------------------------------

    def act(self, image, proprio, state, deterministic=False, telemetry=False):
        mu, log_std, value, new_state, tel = self(image, proprio, state, telemetry)
        std = log_std.exp()
        if deterministic:
            action = mu
        else:
            action = mu + std * torch.randn_like(mu)
        logp = self._logp(mu, log_std, action)
        return action, logp, value, new_state, tel

    @staticmethod
    def _logp(mu, log_std, action):
        var = torch.exp(2 * log_std)
        return (-0.5 * ((action - mu) ** 2 / var + 2 * log_std + math.log(2 * math.pi))).sum(-1)

    @staticmethod
    def entropy(log_std):
        return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
