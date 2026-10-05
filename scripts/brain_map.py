#!/usr/bin/env python
"""Which parts of the connectome are driving?

Runs the frozen whole-brain policy through one drive, records the firing rate of
every neuron at every step, compares it with an eyes-closed baseline, and paints
the result onto the real soma positions of the MaleCNS reconstruction.

    --figure   one static overview (soma cloud by region, by steering
               correlation, by recruitment; per-region bars; where the steering
               descending neurons get their input)
    --video    the same soma cloud lit up frame by frame next to the driver's
               camera
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flydrive import viz
from matplotlib.colors import LinearSegmentedColormap

# Diverging map whose centre is the panel background, so neurons whose rate
# did not change vanish and the recruited ones stand out on a dark figure.
DARK_DIV = LinearSegmentedColormap.from_list("darkdiv", ["#1f6feb", "#161b22", "#f85149"])
from flydrive.config import EnvConfig
from flydrive.env import FlyDriveEnv
from flydrive.wholebrain import WholeCNS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SUPER_ORDER = ["ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
               "cb_sensory", "cb_intrinsic", "descending_neuron", "ascending_neuron",
               "vnc_sensory", "vnc_intrinsic", "vnc_motor", "sensory_ascending"]
SUPER_LABEL = {"ol_sensory": "photoreceptors", "ol_intrinsic": "optic lobe",
               "visual_projection": "visual projection", "visual_centrifugal": "visual centrifugal",
               "cb_sensory": "central brain sensory", "cb_intrinsic": "central brain",
               "descending_neuron": "descending neurons", "ascending_neuron": "ascending neurons",
               "vnc_sensory": "VNC sensory", "vnc_intrinsic": "VNC (nerve cord)",
               "vnc_motor": "VNC motor", "sensory_ascending": "sensory ascending"}
SUPER_COLOR = {"ol_sensory": "#ffd166", "ol_intrinsic": "#f0883e", "visual_projection": "#ff7b72",
               "visual_centrifugal": "#d2a8ff", "cb_sensory": "#79c0ff", "cb_intrinsic": "#58a6ff",
               "descending_neuron": "#3fb950", "ascending_neuron": "#7ee787",
               "vnc_sensory": "#a5d6ff", "vnc_intrinsic": "#8b949e", "vnc_motor": "#56d364",
               "sensory_ascending": "#c9d1d9"}


# ---------------------------------------------------------------------------


@torch.no_grad()
def collect(ckpt, n_steps=420, settle=20, seed=11, cache=None, city=False):
    if cache and os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=True))
    ck = torch.load(ckpt, map_location=DEV, weights_only=False)
    if city:
        # the signalled grid, in the geometry this checkpoint was trained for
        from flydrive.env_city import CityEnv
        from flydrive.city import CityConfig
        from flydrive.eye import PinholeCamera
        from flydrive.world import Renderer
        cc = dict(ck["city_cfg"]); cc["building_height"] = tuple(cc.get("building_height", (6.0, 20.0)))
        env = CityEnv(EnvConfig(n_envs=8), city_cfg=CityConfig(**cc), device=DEV, seed=seed)
        cam = PinholeCamera(920, 420, fov_deg=104.0, pitch_deg=-5.0, device=DEV)
        world = env.render_world
        world = world() if callable(world) else world
        cam_rend = Renderer(cam, world, env.cfg.render, env.cfg.car.eye_height, device=DEV, seed=seed)
    else:
        env = FlyDriveEnv(EnvConfig(n_envs=8), device=DEV, seed=seed)
    cfg = env.cfg
    from flydrive.wholebrain import readout_kind
    brain = WholeCNS(cfg, device=DEV, readout=readout_kind(ck["brain"])).to(DEV).eval()
    brain.load_state_dict(ck["brain"], strict=False)
    N = brain.N

    # -- eyes closed: uniform grey retina, no proprioception ----------------
    st = brain.init_state(8, DEV)
    grey = torch.full((8, 2, cfg.eye.n_row, cfg.eye.n_col), 0.5, device=DEV)
    zero = torch.zeros(8, cfg.n_proprio, device=DEV)
    acc = torch.zeros(N, device=DEV); n = 0
    for t in range(240):
        _, _, _, st, tel = brain(grey, zero, st, telemetry=True)
        if t >= 120:
            acc += tel["rate"][0]; n += 1
    base = (acc / n).cpu().numpy().astype(np.float32)

    # -- one deterministic drive --------------------------------------------
    obs, prop = env.reset_all(), env.proprio()
    st = brain.init_state(8, DEV)
    rates, steer, thr, lat, head, speed, pos, yaw, track, frames = [], [], [], [], [], [], [], [], [], []
    for t in range(settle + n_steps):
        a, _, _, st, tel = brain.act(obs, prop, st, deterministic=True, telemetry=True)
        obs, prop, r, d, info = env.step(a)
        if d[0]:
            st.reset_(d, brain.init_state(8, DEV))
        if t >= settle:
            rates.append(tel["rate"][0].half().cpu().numpy())
            steer.append(float(a[0, 0])); thr.append(float(a[0, 1]))
            lat.append(float(info["lateral"][0])); head.append(float(info["heading_err"][0]))
            speed.append(float(env.speed[0])); pos.append(env.pos[0].cpu().numpy().copy())
            yaw.append(float(env.heading[0])); track.append(int(env.track_idx[0]) if not city else 0)
            if city:   # the driver's camera with this step's cars, lamps and buildings
                _, _, rgb = cam_rend.render(env.pos[0:1], env.heading[0:1], env.zero_idx[0:1], rgb=True, **env.scene_for(0))
                frames.append((rgb[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    R = np.stack(rates)                                            # (T, N) float16
    dn_idx = brain.dn_idx.cpu().numpy()
    w = brain.motor.weight.detach().cpu().numpy()                  # (2, n_dn)
    var = (brain.dn_norm.sq - brain.dn_norm.mean ** 2).clamp_min(1e-12)
    z_scale = (1.0 / torch.sqrt(var)).cpu().numpy()
    out = dict(R=R, base=base, steer=np.array(steer), throttle=np.array(thr), lat=np.array(lat),
               head=np.array(head), speed=np.array(speed), pos=np.stack(pos), yaw=np.array(yaw),
               track=np.array(track), dn_idx=dn_idx, motor_w=w, z_scale=z_scale,
               step=ck.get("global_step", 0) or 0)
    if frames:
        out["frames"] = np.stack(frames)
    if cache:
        np.savez_compressed(cache, **out)
    return out


def neuron_metrics(d):
    R = d["R"].astype(np.float32)
    T = R.shape[0]
    mean = R.mean(0)
    tstd = R.std(0)
    recruit = mean - d["base"]

    def corr(x):
        x = (x - x.mean()) / (x.std() + 1e-9)
        Rc = R - mean
        num = (Rc * x[:, None]).sum(0) / T
        return np.nan_to_num(num / (tstd + 1e-9))
    return dict(mean=mean, tstd=tstd, recruit=recruit,
                c_steer=corr(d["steer"]), c_lat=corr(d["lat"]), c_head=corr(d["head"]))


def readout_importance(d):
    """How much each descending neuron actually contributes to the command."""
    R = d["R"].astype(np.float32)[:, d["dn_idx"]]
    z = (R - R.mean(0)) * d["z_scale"]
    imp = np.abs(d["motor_w"]) * z.std(0)[None, :]                 # (2, n_dn)
    return imp


# ---------------------------------------------------------------------------


def soma_cloud(conn):
    """Soma xyz (voxels) for every neuron in the model's index order, NaN if unknown."""
    import pandas as pd
    ann = pd.read_feather(os.path.join("data", "body-annotations.feather"),
                          columns=["bodyId", "somaLocation"])
    ann = ann.set_index("bodyId")
    loc = ann["somaLocation"].reindex(conn.body)
    xyz = np.full((conn.n, 3), np.nan, np.float32)
    ok = loc.notna().to_numpy()
    xyz[ok] = np.stack(loc[ok].to_list()).astype(np.float32)
    return xyz


def style(ax, title):
    viz.style_axes(ax, title)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def scatter(ax, xy, c, cmap, vmin, vmax, s=0.6, order=None):
    o = np.argsort(c) if order is None else order
    return ax.scatter(xy[o, 0], xy[o, 1], c=c[o], cmap=cmap, vmin=vmin, vmax=vmax,
                      s=s, linewidths=0, rasterized=True)


def figure(d, conn, xyz, out):
    m = neuron_metrics(d)
    imp = readout_importance(d)
    has = np.isfinite(xyz[:, 0])
    # x is left-right, y dorsal-ventral in the FlyEM frame; flip y so dorsal is up
    xy = np.stack([xyz[:, 0], -xyz[:, 1]], 1)

    fig = viz.new_figure(19.2, 10.8)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.16,
                          left=0.075, right=0.985, top=0.90, bottom=0.07)

    # --- A: anatomy ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0]); style(ax, "MaleCNS v1.0  |  141,781 somata coloured by region")
    for sc in SUPER_ORDER:
        sel = has & (conn.superclass == sc)
        if sel.any():
            ax.scatter(xy[sel, 0], xy[sel, 1], s=0.5, c=SUPER_COLOR[sc], linewidths=0,
                       rasterized=True, label=f"{SUPER_LABEL[sc]} ({sel.sum():,})")
    ax.legend(fontsize=6.5, markerscale=8, facecolor=viz.PANEL, edgecolor=viz.GRID,
              labelcolor=viz.FG, loc="lower left", ncol=2, framealpha=0.9)
    ax.set_aspect("equal")

    # --- B: steering correlation ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); style(ax, "|correlation| of firing rate with the steering command, while driving")
    c = np.abs(m["c_steer"]); v = np.percentile(c[has], 99)
    sc_ = scatter(ax, xy[has], c[has], "magma", 0, v)
    ax.set_aspect("equal")
    cb = fig.colorbar(sc_, ax=ax, fraction=0.03, pad=0.01); cb.ax.tick_params(colors=viz.MUTED, labelsize=7)

    # --- C: recruitment ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); style(ax, "recruitment: mean rate while driving minus eyes-closed baseline")
    c = m["recruit"]; v = np.percentile(np.abs(c[has]), 99)
    sc_ = scatter(ax, xy[has], c[has], DARK_DIV, -v, v, order=np.argsort(np.abs(c[has])))
    ax.set_aspect("equal")
    cb = fig.colorbar(sc_, ax=ax, fraction=0.03, pad=0.01); cb.ax.tick_params(colors=viz.MUTED, labelsize=7)

    # --- D: per-region bars -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0]); viz.style_axes(ax, "per region: mean |steering correlation|  and  share of neurons recruited (|Δrate| > 1e-3)")
    labels, cs, rec = [], [], []
    for sc in SUPER_ORDER:
        sel = conn.superclass == sc
        if sel.sum() < 50:
            continue
        labels.append(SUPER_LABEL[sc]); cs.append(np.abs(m["c_steer"][sel]).mean())
        rec.append((np.abs(m["recruit"][sel]) > 1e-3).mean())
    y = np.arange(len(labels))
    ax.barh(y - 0.2, cs, 0.38, color=viz.ACCENT, label="mean |corr| with steering")
    ax2 = ax.twiny(); ax2.barh(y + 0.2, rec, 0.38, color=viz.WARM, label="fraction recruited")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
    ax.tick_params(axis="x", colors=viz.ACCENT, labelsize=7); ax2.tick_params(axis="x", colors=viz.WARM, labelsize=7)
    for s in ax2.spines.values(): s.set_color(viz.GRID)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, facecolor=viz.PANEL, edgecolor=viz.GRID, labelcolor=viz.FG, loc="lower right")

    # --- E: the descending neurons that carry the command -------------------------
    ax = fig.add_subplot(gs[1, 1]); viz.style_axes(ax, "descending neurons with the largest steering contribution (|w| x std of input)")
    dn_types = conn.type[d["dn_idx"]]
    top = np.argsort(imp[0])[::-1][:18]
    ax.barh(np.arange(len(top)), imp[0][top], color=viz.GOOD)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"{dn_types[i]}  ({'L' if conn.side[d['dn_idx'][i]]=='L' else 'R'})" for i in top], fontsize=7)
    ax.invert_yaxis(); ax.tick_params(axis="x", labelsize=7)

    # --- F: where those neurons get their input ------------------------------------
    ax = fig.add_subplot(gs[1, 2]); viz.style_axes(ax, "presynaptic input to the top-30 steering DNs, by region (sum of |synapse weight|)")
    top30 = set(d["dn_idx"][np.argsort(imp[0])[::-1][:30]].tolist())
    sel = np.isin(conn.post, list(top30))
    pre_sc = conn.superclass[conn.pre[sel]]; wsum = {}
    for sc in np.unique(pre_sc):
        wsum[sc] = np.abs(conn.weight[sel][pre_sc == sc]).sum()
    items = sorted(wsum.items(), key=lambda kv: -kv[1])[:10]
    tot = sum(v for _, v in items) + 1e-9
    ax.barh(np.arange(len(items)), [v / tot for _, v in items],
            color=[SUPER_COLOR.get(k, viz.MUTED) for k, _ in items])
    ax.set_yticks(np.arange(len(items))); ax.set_yticklabels([SUPER_LABEL.get(k, k) for k, _ in items], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("share of input synapses", color=viz.MUTED, fontsize=7); ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("which parts of the fly's nervous system are driving the car", color=viz.FG,
                 fontsize=15, x=0.075, ha="left", y=0.965)
    fig.text(0.075, 0.925, f"whole MaleCNS connectome, brain frozen, readout trained to {int(d['step']):,} steps   |   "
             f"one deterministic drive of {len(d['steer'])} steps   |   eyes-closed baseline = uniform grey retina",
             color=viz.MUTED, fontsize=9, ha="left")
    fig.savefig(out, dpi=100, facecolor=viz.BG)
    print("wrote", out)
    return m, imp


def video(d, conn, xyz, out, fps=20, seed=11):
    """Driver's camera beside the soma cloud, every neuron coloured by its
    firing-rate change against the eyes-closed baseline, frame by frame."""
    import imageio.v2 as imageio
    from flydrive.eye import PinholeCamera
    from flydrive.world import Renderer, TrackSet

    cfg = EnvConfig(n_envs=8)
    tracks = TrackSet(cfg.track, device=DEV, seed=seed)          # same seed as collect()
    cam = PinholeCamera(920, 420, fov_deg=104.0, pitch_deg=-5.0, device=DEV)
    rend = Renderer(cam, tracks, cfg.render, cfg.car.eye_height, device=DEV, seed=seed)

    R = d["R"]; base = d["base"]; T = R.shape[0]
    has = np.isfinite(xyz[:, 0])
    xy = np.stack([xyz[has, 0], -xyz[has, 1]], 1)
    delta_all = R[:, has].astype(np.float32) - base[has][None, :]
    v = float(np.percentile(np.abs(delta_all), 99.5)) + 1e-9
    sup = conn.superclass[has]
    groups = [("optic lobe", np.isin(sup, ["ol_intrinsic", "ol_sensory"])),
              ("visual projection", sup == "visual_projection"),
              ("central brain", np.isin(sup, ["cb_intrinsic", "cb_sensory"])),
              ("descending", sup == "descending_neuron"),
              ("nerve cord", np.isin(sup, ["vnc_intrinsic", "vnc_sensory", "vnc_motor", "ascending_neuron"]))]

    fig = viz.new_figure(16, 9)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.42, 0.42],
                          left=0.075, right=0.985, top=0.90, bottom=0.07, wspace=0.08, hspace=0.35)
    ax_cam = fig.add_subplot(gs[0, 0]); ax_cam.set_xticks([]); ax_cam.set_yticks([])
    for sp in ax_cam.spines.values(): sp.set_color(viz.GRID)
    im = ax_cam.imshow(np.zeros((cam.n_row, cam.n_col, 3)), interpolation="bilinear")
    hud = ax_cam.text(0.015, 0.97, "", transform=ax_cam.transAxes, color="w", fontsize=9.5, va="top",
                      family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="#000000aa", ec="#ffffff33"))
    ax_cam.set_title("driver's-eye camera", color=viz.FG, fontsize=9.5, loc="left", pad=4)

    ax_tr = fig.add_subplot(gs[1, 0]); viz.style_axes(ax_tr, "steering command  /  lane offset (m / 5)")
    l_st, = ax_tr.plot([], [], color=viz.BAD, lw=1.4, label="steering")
    l_lat, = ax_tr.plot([], [], color=viz.VIOLET, lw=1.2, label="lane offset / 5 m")
    ax_tr.set_ylim(-1.1, 1.1); ax_tr.axhline(0, color=viz.MUTED, lw=0.6)
    ax_tr.legend(fontsize=6.5, ncol=2, facecolor=viz.PANEL, edgecolor=viz.GRID, labelcolor=viz.FG, loc="upper left")

    ax_bar = fig.add_subplot(gs[2, 0]); viz.style_axes(ax_bar, "mean |rate change| by region, this frame")
    bars = ax_bar.barh(np.arange(len(groups)), np.zeros(len(groups)),
                       color=[viz.WARM, viz.BAD, viz.ACCENT, viz.GOOD, viz.MUTED])
    ax_bar.set_yticks(np.arange(len(groups))); ax_bar.set_yticklabels([g for g, _ in groups], fontsize=7)
    ax_bar.invert_yaxis(); ax_bar.set_xlim(0, float(np.percentile([np.abs(delta_all[t][m]).mean() for t in range(0, T, 10) for _, m in groups], 99)) * 1.15)
    ax_bar.tick_params(axis="x", labelsize=7)

    ax_br = fig.add_subplot(gs[:, 1]); style(ax_br, "MaleCNS connectome  |  firing-rate change vs eyes closed  (blue down, red up)")
    # Faint underlay of every soma so the outline of the brain and nerve cord
    # stays visible; the coloured layer on top only shows neurons that moved.
    ax_br.scatter(xy[:, 0], xy[:, 1], s=0.5, c="#2d333b", linewidths=0, zorder=1)
    sc = ax_br.scatter(xy[:, 0], xy[:, 1], c=np.zeros(len(xy)), cmap=DARK_DIV, vmin=-v, vmax=v,
                       s=0.7, linewidths=0, zorder=2)
    ax_br.set_aspect("equal")
    title = fig.suptitle("", color=viz.FG, fontsize=13, x=0.075, ha="left", y=0.965)
    trained = (f"readout trained to {int(d['step']):,} steps" if int(d["step"]) > 0
               else "readout fitted by ridge regression on an expert's drive")
    fig.text(0.075, 0.925, f"whole MaleCNS connectome, brain frozen, {trained}   |   "
             "162,432 neurons, 141,781 with a soma position", color=viz.MUTED, fontsize=8.5, ha="left")
    city = "frames" in d

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8, ffmpeg_log_level="error")
    try:
        for t in range(T):
            # the cache stores yaw/steer as float64; the renderer's fields are float32
            pos = torch.tensor(d["pos"][t:t + 1], device=DEV, dtype=torch.float32)
            yaw = torch.tensor(d["yaw"][t:t + 1], device=DEV, dtype=torch.float32)
            ti = torch.tensor(d["track"][t:t + 1], device=DEV, dtype=torch.long)
            if city:
                im.set_data(d["frames"][t])
            else:
                _, _, rgb = rend.render(pos, yaw, ti, rgb=True)
                im.set_data(rgb[0, 0].cpu().numpy())
            hud.set_text(f"{d['speed'][t]*3.6:5.1f} km/h\nlane offset {d['lat'][t]:+5.2f} m\nsteer {d['steer'][t]:+.2f}")
            delta = delta_all[t]
            # draw the strongest changes on top
            o = np.argsort(np.abs(delta)); sc.set_offsets(xy[o]); sc.set_array(delta[o])
            for b, (_, m) in zip(bars, groups): b.set_width(float(np.abs(delta[m]).mean()))
            lo = max(0, t - 200); tt = np.arange(lo, t + 1) * 0.05
            l_st.set_data(tt, d["steer"][lo:t + 1]); l_lat.set_data(tt, d["lat"][lo:t + 1] / 5.0)
            ax_tr.set_xlim(tt[0], tt[-1] + 1e-3)
            title.set_text(f"the whole connectome driving{'   |   city' if city else ''}   |   t = {t*0.05:5.1f} s")
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if t % 60 == 0: print(f"  frame {t}/{T}", flush=True)
    finally:
        writer.close()
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cns_frozen/best.pt")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default="runs/brain_regions.png")
    ap.add_argument("--steps", type=int, default=420)
    ap.add_argument("--video-out", default="runs/brain_activity.mp4")
    ap.add_argument("--city", action="store_true", help="drive the signalled grid instead of the closed track")
    args = ap.parse_args()

    from flydrive import connectome as cx
    conn = cx.load(verbose=False)
    cache = os.path.join(os.path.dirname(args.ckpt), "brain_activity_city.npz" if args.city else "brain_activity.npz")
    d = collect(args.ckpt, n_steps=args.steps, cache=cache, city=args.city)
    print(f"activity cache: {d['R'].shape[0]} steps x {d['R'].shape[1]:,} neurons")
    xyz = soma_cloud(conn)
    if args.figure:
        figure(d, conn, xyz, args.out)
    if args.video:
        video(d, conn, xyz, args.video_out)


if __name__ == "__main__":
    main()
