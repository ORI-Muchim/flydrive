"""Visualisation: the hexagonal eye, the optic-flow field the fly computes,
the heading ring, and the training dashboard.

The eye panels draw one polygon per ommatidium on the real hexagonal lattice
rather than resampling to a rectangular image, because the lattice *is* the
computation -- T4/T5 read their neighbours along those hex axes.
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

# -- house style ------------------------------------------------------------

BG = "#0d1117"
PANEL = "#161b22"
FG = "#c9d1d9"
MUTED = "#6e7681"
GRID = "#21262d"
ACCENT = "#58a6ff"
WARM = "#f0883e"
GOOD = "#3fb950"
BAD = "#f85149"
VIOLET = "#bc8cff"

FLOW_CMAP = LinearSegmentedColormap.from_list(
    "flow", ["#0d1117", "#15304f", "#1f6feb", "#58a6ff", "#a5d6ff", "#ffffff"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div", ["#f85149", "#8b2c28", "#161b22", "#1a5c33", "#3fb950"])


def style_axes(ax, title=None, fontsize=8):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=fontsize - 1, length=2)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=FG, fontsize=fontsize, pad=4, loc="left")
    return ax


def new_figure(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig


# ---------------------------------------------------------------------------
# Hexagonal eye panel
# ---------------------------------------------------------------------------


def hex_vertices(cx, cy, r):
    """Pointy-top regular hexagon -- the Voronoi cell of a triangular lattice."""
    ang = np.deg2rad(np.arange(30, 390, 60))
    return np.stack([cx[..., None] + r * np.cos(ang),
                     cy[..., None] + r * np.sin(ang)], axis=-1)


class FlyEyePanel:
    """Draws both retinas as one panoramic hexagonal mosaic.

    Azimuth is plotted as ``-az`` so the panorama reads like a first-person
    view: what lies to the fly's left appears on the left of the panel.
    """

    def __init__(self, ax, eye, cmap="gray", vmin=0.0, vmax=1.0, edge=0.0):
        self.ax = ax
        self.eye = eye
        az = eye.az.detach().cpu().numpy()          # (2, H, W)
        el = eye.el.detach().cpu().numpy()
        x = -np.rad2deg(az).reshape(-1)
        y = np.rad2deg(el).reshape(-1)
        r = eye.cfg.delta_phi_deg / math.sqrt(3.0)
        verts = hex_vertices(x, y, r)
        self.coll = PolyCollection(verts, cmap=cmap, edgecolors="none",
                                   linewidths=edge, zorder=2)
        self.coll.set_norm(Normalize(vmin, vmax))
        ax.add_collection(self.coll)
        pad = eye.cfg.delta_phi_deg
        ax.set_xlim(x.max() + pad, x.min() - pad)   # flipped: front stays centred
        ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_aspect("equal")
        ax.set_facecolor(PANEL)
        ax.set_xticks([-150, -90, -30, 0, 30, 90, 150])
        ax.set_xticklabels(["150R", "90R", "30R", "0", "30L", "90L", "150L"], fontsize=6)
        ax.tick_params(colors=MUTED, labelsize=6, length=2)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.axvline(0, color=MUTED, lw=0.5, ls=":", zorder=3)
        ax.axhline(0, color=MUTED, lw=0.5, ls=":", zorder=3)
        self._quiver = None
        # A front-to-back signal (subtype "a", +col on the lattice) points to
        # larger azimuth in the left eye and smaller azimuth in the right, so on
        # this mirrored panel the two eyes need opposite screen-x signs.  Read
        # the sign off the geometry instead of hard-coding it.
        dx = np.gradient(-np.rad2deg(az), axis=2)
        self._u_sign = np.sign(dx)
        self._u_sign[self._u_sign == 0] = 1.0

    def set(self, values):
        """``values``: (2, H, W) array in the eye's own index order."""
        self.coll.set_array(np.asarray(values).reshape(-1))

    def set_clim(self, vmin, vmax):
        self.coll.set_clim(vmin, vmax)

    def quiver(self, vx, vy, stride=2, scale=None, color="#ff7b72"):
        """Overlay the motion vectors T4/T5 report, subsampled for legibility."""
        az = -np.rad2deg(self.eye.az.detach().cpu().numpy())
        el = np.rad2deg(self.eye.el.detach().cpu().numpy())
        sl = (slice(None), slice(None, None, stride), slice(None, None, stride))
        X, Y = az[sl].reshape(-1), el[sl].reshape(-1)
        U = (np.asarray(vx) * self._u_sign)[sl].reshape(-1)
        V = np.asarray(vy)[sl].reshape(-1)
        # Blank out the near-silent detectors: a zero-length arrow still draws
        # its head, and a grid of dots hides the flow field it is meant to show.
        mag = np.hypot(U, V)
        cut = float(np.percentile(mag, 55))
        quiet = mag <= max(cut, 1e-9)
        U = np.where(quiet, np.nan, U)
        V = np.where(quiet, np.nan, V)
        if self._quiver is None:
            self._quiver = self.ax.quiver(
                X, Y, U, V, color=color, angles="xy", scale_units="xy",
                scale=scale or 0.06, width=0.0028, headwidth=3.2,
                headlength=3.6, alpha=0.95, zorder=4)
        else:
            self._quiver.set_UVC(U, V)


# ---------------------------------------------------------------------------
# Ring attractor
# ---------------------------------------------------------------------------


class RingPanel:
    """Polar view of the EPG bump against the true heading and the PFL3 goal."""

    def __init__(self, ax, n_epg):
        self.ax = ax
        self.n = n_epg
        self.theta = np.arange(n_epg) * (2 * np.pi / n_epg)
        width = 2 * np.pi / n_epg * 0.86
        self.bars = ax.bar(self.theta, np.zeros(n_epg), width=width,
                           color=ACCENT, alpha=0.85, zorder=2)
        self.bump = ax.annotate("", xy=(0, 1), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.8), zorder=5)
        self.goal = ax.annotate("", xy=(0, 1), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.4,
                                                linestyle="--"), zorder=5)
        ax.set_facecolor(PANEL)
        ax.set_yticklabels([])
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(["0", "", "90", "", "180", "", "270", ""], fontsize=6)
        ax.tick_params(colors=MUTED, pad=-2)
        ax.grid(color=GRID, lw=0.5)
        ax.spines["polar"].set_color(GRID)

    def set(self, epg, goal_ang=None):
        epg = np.asarray(epg).reshape(-1)
        m = max(float(epg.max()), 1e-6)
        for b, v in zip(self.bars, epg):
            b.set_height(v / m)
        c = float((epg * np.cos(self.theta)).sum())
        s = float((epg * np.sin(self.theta)).sum())
        mag = min(math.hypot(c, s) / max(epg.sum(), 1e-6) * 2.0, 1.0)
        self.bump.xy = (math.atan2(s, c), mag)
        if goal_ang is not None:
            self.goal.xy = (float(goal_ang), 0.9)
        self.ax.set_ylim(0, 1.05)


# ---------------------------------------------------------------------------
# Training dashboard
# ---------------------------------------------------------------------------


PANELS = [
    ("ep_distance", "distance per episode (m)", GOOD, False),
    ("ep_return", "episode return", ACCENT, False),
    ("crash_rate", "crash rate", BAD, False),
    ("speed", "mean speed (m/s)", WARM, False),
    ("abs_lateral", "|lane offset| (m)", VIOLET, False),
    ("ep_length", "episode length (steps)", "#79c0ff", False),
    ("reward", "reward per step", "#d2a8ff", False),
    ("explained_var", "value explained var", "#7ee787", False),
    ("value_loss", "value loss", "#ffa657", True),
    ("entropy", "policy entropy", "#a5d6ff", False),
    ("kl", "approx KL", "#ff7b72", True),
    ("log_std", "action log-std", "#8b949e", False),
    ("t4_act", "T4 mean activity", "#58a6ff", False),
    ("t5_act", "T5 mean activity", "#bc8cff", False),
    ("hs_act", "HS |response|", "#3fb950", False),
    ("ring_conc", "heading-bump sharpness", "#f0883e", False),
]


def smooth(y, k=9):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    k = max(3, min(k, len(y) // 2 * 2 + 1))
    kern = np.ones(k) / k
    pad = np.concatenate([np.full(k // 2, y[0]), y, np.full(k // 2, y[-1])])
    return np.convolve(pad, kern, mode="valid")[:len(y)]


# The last four panels read `probe()`, whose meaning depends on the model.
CNS_LABELS = {
    "t4_act": "mean firing rate (all 162k neurons)",
    "t5_act": "fraction of neurons active",
    "hs_act": "descending-neuron |activity|",
    "ring_conc": "population rate spread (std)",
}


def draw_dashboard(history, path, title="fly brain -- driving", subtitle="", labels=None):
    """Render the whole training history to a single PNG.

    ``labels`` overrides panel titles by history key, e.g. :data:`CNS_LABELS`
    for the whole-connectome model.
    """
    steps = np.asarray(history["step"], dtype=float)
    if len(steps) == 0:
        return
    labels = labels or {}
    ncol, nrow = 4, 4
    fig = new_figure(17, 10.5)
    gs = fig.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.26,
                          left=0.05, right=0.985, top=0.90, bottom=0.06)

    for i, (key, label, colour, logy) in enumerate(PANELS):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        style_axes(ax, labels.get(key, label))
        y = np.asarray(history.get(key, []), dtype=float)
        if len(y) == 0:
            continue
        n = min(len(y), len(steps))
        x, y = steps[:n] / 1e3, y[:n]
        finite = np.isfinite(y)
        if finite.sum() < 2:
            continue
        ax.plot(x[finite], y[finite], color=colour, lw=0.7, alpha=0.30)
        ax.plot(x[finite], smooth(y[finite]), color=colour, lw=1.6)
        # A series that is constant to float precision (crash rate pinned at
        # 1.0) otherwise gets a "1e-8+1" offset axis.
        ax.ticklabel_format(useOffset=False, axis="y")
        if np.ptp(y[finite]) < 1e-6:
            ax.set_ylim(y[finite][0] - 0.05, y[finite][0] + 0.05)
        if logy and np.nanmin(y[finite]) > 0:
            ax.set_yscale("log")
        last = y[finite][-1]
        ax.annotate(f"{last:,.3g}", xy=(0.985, 0.90), xycoords="axes fraction",
                    ha="right", va="top", color=colour, fontsize=8, weight="bold")
        if i // ncol == nrow - 1:
            ax.set_xlabel("env steps (k)", color=MUTED, fontsize=7)

    fig.suptitle(title, color=FG, fontsize=15, x=0.05, ha="left", y=0.975)
    if subtitle:
        fig.text(0.05, 0.935, subtitle, color=MUTED, fontsize=9, ha="left")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
