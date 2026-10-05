"""Sanity check: draw the track and what the fly actually sees from it."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from flydrive.config import EnvConfig
from flydrive.eye import CompoundEye
from flydrive.world import TrackSet, Renderer

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = EnvConfig()
tracks = TrackSet(cfg.track, device=dev, seed=1)
eye = CompoundEye(cfg.eye, device=dev)
rend = Renderer(eye, tracks, cfg.render, cfg.car.eye_height, device=dev, seed=1)

T = 0
idxs = [0, 140, 300, 520]
ti = torch.full((len(idxs),), T, device=dev, dtype=torch.long)
si = torch.tensor(idxs, device=dev)
pos = tracks.centre[T][si] + tracks.normal[T][si] * 1.8   # sit off-centre on purpose
head = torch.atan2(tracks.tangent[T][si, 1], tracks.tangent[T][si, 0])
img, dep = rend.render(pos, head, ti)
img = img.cpu().numpy()

fig = plt.figure(figsize=(16, 9), facecolor="#0e1116")
gs = fig.add_gridspec(len(idxs), 3, width_ratios=[1.1, 1.3, 1.3], wspace=0.18, hspace=0.25)

ax = fig.add_subplot(gs[:, 0]); ax.set_facecolor("#0e1116")
c = tracks.centre[T].cpu(); n = tracks.normal[T].cpu(); hw = cfg.track.road_width / 2
ax.fill(*zip(*(torch.cat([c + n * hw, (c - n * hw).flip(0)]).tolist())), color="#2a2f3a", lw=0)
ax.plot(c[:, 0], c[:, 1], color="#4a5568", lw=0.8, ls="--")
p = tracks.post_xy[T].cpu(); pc = tracks.post_col[T].cpu()
ax.scatter(p[:, 0], p[:, 1], c=["#f0f0f0" if v > .5 else "#181818" for v in pc], s=6, zorder=3, edgecolors="#666", linewidths=.3)
pp = pos.cpu()
for k, (x, y) in enumerate(pp.tolist()):
    ax.scatter([x], [y], s=110, marker="o", color=f"C{k}", zorder=5, edgecolors="w", linewidths=1)
    ax.annotate(f"{k}", (x, y), color="w", fontsize=9, ha="center", va="center", zorder=6)
ax.set_aspect("equal"); ax.set_title("track 0  (10 m road, posts every 9 m)", color="w", fontsize=10)
ax.tick_params(colors="#888", labelsize=7)
for s_ in ax.spines.values(): s_.set_color("#333")

for k in range(len(idxs)):
    for e, name in enumerate(["left eye", "right eye"]):
        a = fig.add_subplot(gs[k, 1 + e])
        a.imshow(img[k, e], origin="lower", cmap="gray", vmin=0, vmax=1,
                 aspect="auto", interpolation="nearest")
        a.set_title(f"pos {k} - {name}", color=f"C{k}", fontsize=8)
        a.set_xticks([0, 18, 35]); a.set_xticklabels(["front", "90 deg", "back"], fontsize=6)
        el = eye.el[e, :, 0].cpu() * 57.29578
        a.set_yticks([0, 12, 23]); a.set_yticklabels([f"{el[i]:.0f}" for i in (0, 12, 23)], fontsize=6)
        a.tick_params(colors="#888")
fig.suptitle("Fly compound-eye view of the driving world  (864 ommatidia/eye)",
             color="w", fontsize=13)
fig.savefig("runs/demo_render.png", dpi=110, facecolor="#0e1116", bbox_inches="tight")
print("wrote runs/demo_render.png")
print("luminance per view:", [f"{img[k].mean():.3f}" for k in range(len(idxs))])
