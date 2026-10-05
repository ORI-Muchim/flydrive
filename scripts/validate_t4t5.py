"""Direction-selectivity check on the T4/T5 array.

The classic experiment: drift a sinusoidal grating across the eye at a range of
directions and temporal frequencies, and measure each subtype's mean response.
A working elementary motion detector is tuned to one direction and inverted-U
tuned in temporal frequency.  This runs the model exactly as the driving task
does -- same timestep, same state carry-over.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flydrive import viz
from flydrive.config import EnvConfig
from flydrive.brain import FlyBrain

DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = EnvConfig()
brain = FlyBrain(cfg).to(DEV).eval()
H, W = cfg.eye.n_row, cfg.eye.n_col
dt = cfg.dt

DIRS = np.arange(0, 360, 30)          # degrees on the lattice, 0 = front-to-back
TFS = [0.5, 1.0, 2.0, 4.0, 8.0]       # Hz
SPATIAL_PERIOD = 6.0                  # lattice steps per cycle
N_WARM, N_MEAS = 40, 60
SUBTYPES = ["a  front->back", "b  back->front", "c  upward", "d  downward"]


@torch.no_grad()
def probe(direction_deg, tf_hz):
    """Drift a grating and return mean T4/T5 response per subtype, shape (2, 4)."""
    th = np.deg2rad(direction_deg)
    col = np.arange(W)[None, :]
    row = np.arange(H)[:, None]
    # Phase advances along the drift direction on the lattice.
    proj = (col * np.cos(th) + row * np.sin(th)) * (2 * np.pi / SPATIAL_PERIOD)
    proj = torch.tensor(proj, dtype=torch.float32, device=DEV)

    state = brain.init_state(1, DEV)
    prop = torch.zeros(1, 3, device=DEV)
    acc = torch.zeros(2, 4, device=DEV)
    for t in range(N_WARM + N_MEAS):
        phase = 2 * np.pi * tf_hz * t * dt
        img = 0.5 + 0.35 * torch.sin(proj - phase)
        img = img.view(1, 1, H, W).expand(1, 2, H, W).contiguous()
        _, _, _, state, tel = brain(img, prop, state, telemetry=True)
        if t >= N_WARM:
            acc[0] += tel["T4"][0].mean(dim=(1, 2, 3))
            acc[1] += tel["T5"][0].mean(dim=(1, 2, 3))
    return (acc / N_MEAS).cpu().numpy()


print("measuring direction tuning ...")
tuning = np.zeros((len(DIRS), 2, 4))          # dir x {T4,T5} x subtype
for i, d in enumerate(DIRS):
    tuning[i] = probe(d, 2.0)
    print(f"  {d:3d} deg  T4={np.round(tuning[i,0],3)}  T5={np.round(tuning[i,1],3)}")

print("measuring temporal-frequency tuning ...")
# Each subtype is probed at its *own* preferred direction, otherwise the curve
# just measures how silent an off-axis cell is.
PREF = [0.0, 180.0, 90.0, 270.0]
tf_curve = np.zeros((len(TFS), 2, 4))
for i, f in enumerate(TFS):
    for j, d in enumerate(PREF):
        r = probe(d, f)
        tf_curve[i, :, j] = r[:, j]

# Direction-selectivity index: (PD - ND) / (PD + ND) at the preferred direction.
def dsi(curve):
    pd = curve.max()
    nd = curve[(curve.argmax() + len(curve) // 2) % len(curve)]
    return (pd - nd) / (pd + nd + 1e-9)


fig = viz.new_figure(15, 7.6)
gs = fig.add_gridspec(2, 4, hspace=0.48, wspace=0.30,
                      left=0.05, right=0.985, top=0.80, bottom=0.09)
th = np.deg2rad(np.append(DIRS, DIRS[0]))
lines = []
for j in range(4):
    ax = fig.add_subplot(gs[0, j], projection="polar")
    for k, (name, colour) in enumerate([("T4 (ON)", viz.ACCENT), ("T5 (OFF)", viz.WARM)]):
        r = tuning[:, k, j]
        r = np.append(r, r[0])
        ax.plot(th, r, color=colour, lw=1.8, label=name)
        ax.fill(th, r, color=colour, alpha=0.15)
    ax.set_facecolor(viz.PANEL)
    ax.set_title(f"T4/T5{SUBTYPES[j]}\nDSI  T4 {dsi(tuning[:,0,j]):.2f}   T5 {dsi(tuning[:,1,j]):.2f}",
                 color=viz.FG, fontsize=9, pad=16)
    ax.tick_params(colors=viz.MUTED, labelsize=6)
    ax.set_yticklabels([])
    ax.grid(color=viz.GRID, lw=0.5)
    ax.spines["polar"].set_color(viz.GRID)
    if j == 0:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.18), fontsize=7,
                  facecolor=viz.PANEL, edgecolor=viz.GRID, labelcolor=viz.FG)

for j in range(4):
    ax = fig.add_subplot(gs[1, j])
    viz.style_axes(ax, f"subtype {SUBTYPES[j].split()[0]} at {PREF[j]:.0f} deg  |  temporal frequency")
    for k, (name, colour) in enumerate([("T4 (ON)", viz.ACCENT), ("T5 (OFF)", viz.WARM)]):
        ax.plot(TFS, tf_curve[:, k, j], "o-", color=colour, lw=1.6, ms=4, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Hz", color=viz.MUTED, fontsize=7)
    if j == 0:
        ax.set_ylabel("mean response", color=viz.MUTED, fontsize=7)

fig.suptitle("T4/T5 direction selectivity  |  drifting sinusoidal grating, "
             f"{SPATIAL_PERIOD:.0f} ommatidia/cycle",
             color=viz.FG, fontsize=13, x=0.05, ha="left", y=0.985)
fig.text(0.05, 0.945, "untrained network -- selectivity comes from the wiring, not from training",
         color=viz.MUTED, fontsize=9, ha="left")
fig.savefig("runs/t4t5_tuning.png", dpi=105, facecolor=viz.BG)
print("\nwrote runs/t4t5_tuning.png")
print("DSI per subtype (T4):", [f"{dsi(tuning[:,0,j]):.3f}" for j in range(4)])
print("DSI per subtype (T5):", [f"{dsi(tuning[:,1,j]):.3f}" for j in range(4)])
