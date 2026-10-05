"""Where in the fly's brain is the hand-built pathway active, frame by frame?

The hand-built model implements a specific set of real cell types.  This lights
up those types' actual somas in the MaleCNS soma cloud with the model's own
activity while it drives (driver's camera beside it), so the anatomy of what is
being used is visible: retina -> lamina -> medulla -> T4/T5 -> lobula plate ->
central complex -> descending neurons.
"""
import argparse, os, re, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from flydrive.config import EnvConfig
from flydrive.env_city import CityEnv
from flydrive.city import CityConfig
from flydrive.brain import FlyBrain
from flydrive.eye import PinholeCamera
from flydrive.world import Renderer
from flydrive import viz
from flydrive import connectome as cx
from brain_map import soma_cloud, style

DEV = "cuda"
HOT = LinearSegmentedColormap.from_list("darkhot", ["#161b22", "#6e2a0a", "#f85149", "#ffd166", "#ffffff"])

# (label, regex on the MaleCNS type name, how to read the model's activity for it)
STAGES = [
    ("retina R1-R6",            r"^R1-R6$",      lambda tel, st, br: tel["contrast"].abs().mean()),
    ("lamina L1 (ON)",          r"^L1$",         lambda tel, st, br: tel["L1"].mean()),
    ("lamina L2 (OFF)",         r"^L2$",         lambda tel, st, br: tel["L2"].mean()),
    ("lamina L3 (sustained)",   r"^L3$",         lambda tel, st, br: tel["contrast"].abs().mean()),
    ("medulla ON", r"^(Mi1|Tm3|Mi4|Mi9)$", lambda tel, st, br: st.medulla[0, [br.medulla_names.index(n) for n in ("Mi1", "Tm3", "Mi4", "Mi9") if n in br.medulla_names]].abs().mean()),
    ("medulla OFF",  r"^(Tm1|Tm2|CT1)$", lambda tel, st, br: st.medulla[0, [br.medulla_names.index(n) for n in ("Tm1", "Tm2", "CT1") if n in br.medulla_names]].abs().mean()),
    ("T4 (ON motion)",           r"^T4[abcd]$",   lambda tel, st, br: tel["T4"].mean()),
    ("T5 (OFF motion)",          r"^T5[abcd]$",   lambda tel, st, br: tel["T5"].mean()),
    ("lobula plate HS",          r"^HS",          lambda tel, st, br: tel["hs"].abs().mean()),
    ("lobula plate VS",          r"^VS",          lambda tel, st, br: tel["vs"].abs().mean()),
    ("central complex EPG",      r"^EPG",         lambda tel, st, br: tel["epg"].abs().mean()),
    ("central complex PFL3",     r"^PFL3$",       lambda tel, st, br: tel["pfl3"].abs().mean()),
    ("descending DNa01-03",      r"^DNa0[123]$",  lambda tel, st, br: tel["dn"].abs().mean()),
]


@torch.no_grad()
def collect(ckpt, n_steps, seed):
    ck = torch.load(ckpt, map_location=DEV, weights_only=False)
    cc = dict(ck["city_cfg"]); cc["building_height"] = tuple(cc.get("building_height", (6.0, 20.0)))
    env = CityEnv(EnvConfig(n_envs=8), city_cfg=CityConfig(**cc), device=DEV, seed=seed)
    brain = FlyBrain(env.cfg).to(DEV).eval(); brain.load_state_dict(ck["brain"], strict=False)
    cam = PinholeCamera(920, 420, fov_deg=104.0, pitch_deg=-5.0, device=DEV)
    world = env.render_world; world = world() if callable(world) else world
    rend = Renderer(cam, world, env.cfg.render, env.cfg.car.eye_height, device=DEV, seed=seed)
    obs, prop = env.reset_all(), env.proprio(); st = brain.init_state(8, DEV)
    act, frames, steer, lat, speed = [], [], [], [], []
    for t in range(n_steps):
        a, _, _, st, tel = brain.act(obs, prop, st, deterministic=True, telemetry=True)
        act.append([float(f(tel, st, brain)) for _, _, f in STAGES])
        obs, prop, r, d, info = env.step(a)
        if d.any():
            st.reset_(d, brain.init_state(8, DEV))
        _, _, rgb = rend.render(env.pos[0:1], env.heading[0:1], env.zero_idx[0:1], rgb=True, **env.scene_for(0))
        frames.append((rgb[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        steer.append(float(a[0, 0])); lat.append(float(info["lateral"][0])); speed.append(float(env.speed[0]))
        if t % 200 == 0: print(f"  step {t}/{n_steps}", flush=True)
    return dict(act=np.array(act), frames=np.stack(frames), steer=np.array(steer), lat=np.array(lat), speed=np.array(speed))


def video(d, conn, xyz, out, title, fps=20):
    import imageio.v2 as imageio
    has = np.isfinite(xyz[:, 0]); xy = np.stack([xyz[has, 0], -xyz[has, 1]], 1)
    types = np.asarray(conn.type).astype(str)[has]
    masks = [np.array([bool(re.match(pat, x)) for x in types]) for _, pat, _ in STAGES]
    A = d["act"]; T = A.shape[0]
    # each stage normalised by its own 98th percentile over the drive -> 0..1
    norm = A / (np.percentile(A, 98, axis=0) + 1e-9); norm = np.clip(norm, 0, 1)
    lit = np.zeros(len(xy), dtype=bool)
    for m in masks: lit |= m
    idx_lit = np.nonzero(lit)[0]
    stage_of = np.full(len(xy), -1); 
    for k, m in enumerate(masks): stage_of[m] = k

    fig = viz.new_figure(16, 9)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 0.38, 0.9],
                          left=0.10, right=0.985, top=0.90, bottom=0.06, wspace=0.06, hspace=0.42)
    ax_cam = fig.add_subplot(gs[0, 0]); ax_cam.set_xticks([]); ax_cam.set_yticks([])
    for sp in ax_cam.spines.values(): sp.set_color(viz.GRID)
    im = ax_cam.imshow(d["frames"][0], interpolation="bilinear")
    hud = ax_cam.text(0.015, 0.97, "", transform=ax_cam.transAxes, color="w", fontsize=9.5, va="top", family="monospace",
                      bbox=dict(boxstyle="round,pad=0.4", fc="#000000aa", ec="#ffffff33"))
    ax_cam.set_title("driver's-eye camera", color=viz.FG, fontsize=9.5, loc="left", pad=4)
    ax_tr = fig.add_subplot(gs[1, 0]); viz.style_axes(ax_tr, "steering  /  lane offset (m / 5)")
    l_st, = ax_tr.plot([], [], color=viz.BAD, lw=1.4); l_lat, = ax_tr.plot([], [], color=viz.VIOLET, lw=1.2)
    ax_tr.set_ylim(-1.1, 1.1); ax_tr.axhline(0, color=viz.MUTED, lw=0.6)
    ax_bar = fig.add_subplot(gs[2, 0]); viz.style_axes(ax_bar, "activity of each stage the model implements, this frame (0-1, own scale)")
    bars = ax_bar.barh(np.arange(len(STAGES)), np.zeros(len(STAGES)), color=[HOT(0.7)] * len(STAGES))
    ax_bar.set_yticks(np.arange(len(STAGES))); ax_bar.set_yticklabels([s for s, _, _ in STAGES], fontsize=6.8)
    ax_bar.invert_yaxis(); ax_bar.set_xlim(0, 1.0); ax_bar.tick_params(axis="x", labelsize=7)
    ax_br = fig.add_subplot(gs[:, 1]); style(ax_br, "MaleCNS soma positions  |  the cell types this model implements, lit by its activity")
    ax_br.scatter(xy[:, 0], xy[:, 1], s=0.4, c="#262c35", linewidths=0, zorder=1)
    sc = ax_br.scatter(xy[idx_lit, 0], xy[idx_lit, 1], c=np.zeros(len(idx_lit)), cmap=HOT, vmin=0, vmax=1, s=1.6, linewidths=0, zorder=2)
    ax_br.set_aspect("equal")
    ttl = fig.suptitle("", color=viz.FG, fontsize=13, x=0.10, ha="left", y=0.965)
    fig.text(0.10, 0.925, title, color=viz.MUTED, fontsize=8.5, ha="left")
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8, ffmpeg_log_level="error")
    try:
        for t in range(T):
            im.set_data(d["frames"][t])
            hud.set_text(f"{d['speed'][t]*3.6:5.1f} km/h\nlane offset {d['lat'][t]:+5.2f} m\nsteer {d['steer'][t]:+.2f}")
            vals = norm[t]
            sc.set_array(vals[stage_of[idx_lit]])
            for b, v in zip(bars, vals): b.set_width(float(v)); b.set_color(HOT(0.35 + 0.6 * float(v)))
            lo = max(0, t - 200); tt = np.arange(lo, t + 1) * 0.05
            l_st.set_data(tt, d["steer"][lo:t + 1]); l_lat.set_data(tt, d["lat"][lo:t + 1] / 5.0); ax_tr.set_xlim(tt[0], tt[-1] + 1e-3)
            ttl.set_text(f"the hand-built fly pathway driving  |  which parts of the brain it is using  |  t = {t*0.05:5.1f} s")
            fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if t % 200 == 0: print(f"  frame {t}/{T}", flush=True)
    finally:
        writer.close()
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="videos/fly_brain_regions.mp4"); ap.add_argument("--title", default="")
    a = ap.parse_args()
    d = collect(a.ckpt, a.steps, a.seed)
    conn = cx.load(verbose=False); xyz = soma_cloud(conn)
    video(d, conn, xyz, a.out, a.title)


if __name__ == "__main__":
    main()
