"""Static check of the city: what the fly sees, and the map around it."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from flydrive import viz
from flydrive.config import EnvConfig
from flydrive.env_city import CityEnv
from flydrive.eye import PinholeCamera
from flydrive.world import Renderer

dev = "cuda"
from flydrive.city import CityConfig
V3 = "--v3" in sys.argv
env = CityEnv(EnvConfig(n_envs=4), city_cfg=CityConfig(irregular=V3, buildings=V3), device=dev, seed=3)
a = torch.zeros(env.B, 2, device=dev); a[:, 1] = 0.3
for _ in range(70):
    obs, pro, r, d, info = env.step(a)
sc, state = env.scene()
cam = PinholeCamera(920, 420, fov_deg=104.0, pitch_deg=-5.0, device=dev)
rend = Renderer(cam, env.city, env.renderer.cfg, env.cfg.car.eye_height, device=dev, seed=3)
k = 0
sl = {kk: v[k:k + 1] for kk, v in sc.items()}
_, _, rgb = rend.render(env.pos[k:k+1], env.heading[k:k+1], env.zero_idx[k:k+1], rgb=True, **sl)
fly = obs[k].cpu().numpy()

fig = viz.new_figure(18, 9)
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1.0, 0.9], left=0.03, right=0.985, top=0.9, bottom=0.05, wspace=0.08, hspace=0.25)
ax = fig.add_subplot(gs[0, 0]); ax.imshow(rgb[0, 0].cpu().numpy()); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("driver's-eye camera in the city", color=viz.FG, fontsize=10, loc="left")
ax.text(0.015, 0.97, f"{float(env.speed[k])*3.6:.0f} km/h   next stop line {float(info['d_stop'][k]):.0f} m   signal {['red','yellow','green'][int(info['signal'][k])]}\ncommand L/S/R {np.round(info['cmd'][k].cpu().numpy(),2).tolist()}",
        transform=ax.transAxes, color="w", fontsize=9, va="top", family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="#000000aa", ec="#ffffff33"))
ax2 = fig.add_subplot(gs[1, 0]); p = viz.FlyEyePanel(ax2, env.eye, cmap="gray"); p.set(fly)
ax2.set_title("the same instant through the compound eyes", color=viz.FG, fontsize=10, loc="left")
axm = fig.add_subplot(gs[:, 1]); viz.style_axes(axm, "map  |  route in blue, lamps by state, other cars in white"); axm.set_aspect("equal")
env.draw_map(axm, k)
px, py = float(env.pos[k, 0]), float(env.pos[k, 1]); hd = float(env.heading[k])
axm.plot([px], [py], marker=(3, 0, math.degrees(hd) - 90), ms=12, color=viz.WARM, ls="none", zorder=6)
axm.set_xlim(px - 120, px + 120); axm.set_ylim(py - 120, py + 120)
fig.suptitle("the city v3: irregular blocks, buildings, signals, other cars" if V3 else "the city: 4x4 signalled grid, right-hand lanes, other cars", color=viz.FG, fontsize=14, x=0.03, ha="left", y=0.965)
out = "runs/demo_city_v3.png" if V3 else "runs/demo_city.png"
fig.savefig(out, dpi=100, facecolor=viz.BG); print("wrote", out)
