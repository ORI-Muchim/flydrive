#!/usr/bin/env python
"""Better views of the driving brain.

    --circuit   the steering circuit: the descending neurons that carry the
                command and the neurons that synapse onto them, drawn as edges
                on the soma map
    --export    compact JSON for the interactive 3D page (subsampled neurons,
                quantised activity over the drive, dashcam thumbnails)
    --video     a rotating 3D soma cloud lit by activity, beside the dashcam

All three reuse the activity recorded by brain_map.py.
"""
import argparse
import math
import base64
import io
import json
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
import importlib.util
_spec = importlib.util.spec_from_file_location("brain_map", os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_map.py"))
bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bm)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REGION = {"ol_sensory": 0, "ol_intrinsic": 0, "visual_projection": 1, "visual_centrifugal": 1,
          "cb_sensory": 2, "cb_intrinsic": 2, "descending_neuron": 3,
          "ascending_neuron": 4, "vnc_sensory": 4, "vnc_intrinsic": 4, "vnc_motor": 4, "sensory_ascending": 4}
REGION_NAMES = ["optic lobe", "visual projection", "central brain", "descending", "nerve cord"]


def region_of(superclass):
    """Map any superclass to one of the five display regions (rare ones by prefix)."""
    if superclass in REGION:
        return REGION[superclass]
    return {"ol": 0, "cb": 2, "vn": 4}.get(str(superclass)[:2], 4)
REGION_COLORS = ["#f0883e", "#ff7b72", "#58a6ff", "#3fb950", "#8b949e"]


def load_all(ckpt):
    from flydrive import connectome as cx
    conn = cx.load(verbose=False)
    d = bm.collect(ckpt, cache=os.path.join(os.path.dirname(ckpt), "brain_activity.npz"))
    xyz = bm.soma_cloud(conn)
    return conn, d, xyz


def pick_neurons(conn, xyz, d, n_target=28000, seed=0):
    """Neurons worth showing: everything that moved, plus a spatial sample of the rest."""
    rng = np.random.default_rng(seed)
    has = np.isfinite(xyz[:, 0])
    delta = d["R"].astype(np.float32) - d["base"][None, :]
    act = np.abs(delta).max(0)
    idx_active = np.where(has & (act > 2e-3))[0]
    rest = np.where(has & (act <= 2e-3))[0]
    n_rest = max(0, n_target - len(idx_active))
    rest = rng.choice(rest, size=min(n_rest, len(rest)), replace=False)
    sel = np.sort(np.concatenate([idx_active, rest]))
    return sel, delta


# ---------------------------------------------------------------------------


def circuit(conn, d, xyz, out, n_dn=20, n_pre=8):
    imp = bm.readout_importance(d)
    dn_idx = d["dn_idx"]
    top = dn_idx[np.argsort(imp[0])[::-1][:n_dn]]
    has = np.isfinite(xyz[:, 0])
    xy = np.stack([xyz[:, 0], -xyz[:, 1]], 1)

    edges, partners, type_w = [], set(), {}
    for post in top:
        sel = np.where(conn.post == post)[0]
        if len(sel) == 0:
            continue
        order = sel[np.argsort(-np.abs(conn.weight[sel]))][:n_pre]
        for e in order:
            pre = conn.pre[e]; w = conn.weight[e]
            edges.append((pre, post, w)); partners.add(pre)
            type_w[conn.type[pre]] = type_w.get(conn.type[pre], 0.0) + abs(w)
    partners = np.array(sorted(partners))

    fig = viz.new_figure(19.2, 10.8)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], left=0.03, right=0.985, top=0.88, bottom=0.06, wspace=0.10)
    ax = fig.add_subplot(gs[0, 0]); bm.style(ax, f"the steering circuit  |  top-{n_dn} descending neurons and their {n_pre} strongest inputs each")
    ax.scatter(xy[has, 0], xy[has, 1], s=0.35, c="#262c34", linewidths=0, zorder=1)
    n_drawn = 0
    for pre, post, w in edges:
        if has[pre] and has[post]:
            col = "#f85149" if w > 0 else "#58a6ff"
            ax.plot([xy[pre, 0], xy[post, 0]], [xy[pre, 1], xy[post, 1]], color=col,
                    lw=0.4 + 2.2 * min(abs(w) / 0.5, 1.0), alpha=0.55, zorder=2)
            n_drawn += 1
    for r in range(5):
        m = np.array([has[p] and region_of(conn.superclass[p]) == r for p in partners], bool)
        if m.any():
            ax.scatter(xy[partners[m], 0], xy[partners[m], 1], s=16, c=REGION_COLORS[r], linewidths=0.4,
                       edgecolors="#0d1117", zorder=3, label=f"input from {REGION_NAMES[r]} ({m.sum()})")
    tt = top[has[top]]
    ax.scatter(xy[tt, 0], xy[tt, 1], s=70, c="#3fb950", marker="D", edgecolors="w", linewidths=0.8, zorder=4,
               label="steering descending neurons")
    for i in tt[:10]:
        ax.annotate(conn.type[i], (xy[i, 0], xy[i, 1]), xytext=(6, 4), textcoords="offset points",
                    color="#e6edf3", fontsize=7, zorder=5)
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, facecolor=viz.PANEL, edgecolor=viz.GRID, labelcolor=viz.FG, loc="lower left", markerscale=1.0)
    ax.text(0.99, 0.02, f"{n_drawn} of {len(edges)} synaptic connections drawn (partners without a soma position omitted)\n"
            "red: excitatory (ACh)   blue: inhibitory (GABA / glutamate / histamine)   width ~ synapse count",
            transform=ax.transAxes, color=viz.MUTED, fontsize=7.5, ha="right", va="bottom")

    ax2 = fig.add_subplot(gs[0, 1]); viz.style_axes(ax2, "cell types feeding the steering descending neurons  (sum of |synapse weight|)")
    items = sorted(type_w.items(), key=lambda kv: -kv[1])[:22]
    cols = []
    for t, _ in items:
        sc = conn.superclass[np.where(conn.type == t)[0][0]]
        cols.append(REGION_COLORS[region_of(sc)])
    ax2.barh(np.arange(len(items)), [v for _, v in items], color=cols)
    ax2.set_yticks(np.arange(len(items))); ax2.set_yticklabels([t for t, _ in items], fontsize=8)
    ax2.invert_yaxis(); ax2.tick_params(axis="x", labelsize=7)
    fig.suptitle("how the command gets to the descending neurons", color=viz.FG, fontsize=15, x=0.03, ha="left", y=0.955)
    fig.text(0.03, 0.915, "whole MaleCNS connectome, brain frozen  |  DN importance = |readout weight| x std of the neuron's normalised input while driving",
             color=viz.MUTED, fontsize=9, ha="left")
    fig.savefig(out, dpi=100, facecolor=viz.BG); print("wrote", out)


# ---------------------------------------------------------------------------


def export(conn, d, xyz, out, n_frames=42, thumb=(320, 144), seed=11):
    from PIL import Image
    from flydrive.eye import PinholeCamera
    from flydrive.world import Renderer, TrackSet

    sel, delta = pick_neurons(conn, xyz, d)
    T = d["R"].shape[0]
    frames = np.linspace(0, T - 1, n_frames).round().astype(int)
    v = float(np.percentile(np.abs(delta[:, sel]), 99.7)) + 1e-9
    q = np.clip(np.round(delta[frames][:, sel] / v * 127), -127, 127).astype(np.int8)   # (F, n)

    p = xyz[sel]
    lo, hi = p.min(0), p.max(0)
    pq = np.round((p - lo) / (hi - lo) * 65535).astype(np.uint16)
    region = np.array([region_of(s) for s in conn.superclass[sel]], np.uint8)
    imp = bm.readout_importance(d)[0]
    top = set(d["dn_idx"][np.argsort(imp)[::-1][:30]].tolist())
    is_top = np.array([i in top for i in sel], np.uint8)

    cfg = EnvConfig(n_envs=8)
    tracks = TrackSet(cfg.track, device=DEV, seed=seed)
    cam = PinholeCamera(*thumb, fov_deg=104.0, pitch_deg=-5.0, device=DEV)
    rend = Renderer(cam, tracks, cfg.render, cfg.car.eye_height, device=DEV, seed=seed)
    thumbs = []
    for t in frames:
        pos = torch.tensor(d["pos"][t:t + 1], device=DEV, dtype=torch.float32)
        yaw = torch.tensor(d["yaw"][t:t + 1], device=DEV, dtype=torch.float32)
        ti = torch.tensor(d["track"][t:t + 1], device=DEV, dtype=torch.long)
        _, _, rgb = rend.render(pos, yaw, ti, rgb=True)
        im = Image.fromarray((rgb[0, 0].cpu().numpy() * 255).astype(np.uint8))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=70)
        thumbs.append(base64.b64encode(buf.getvalue()).decode())

    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    payload = dict(
        n=int(len(sel)), n_frames=int(n_frames), dt=0.05 * float(T - 1) / max(n_frames - 1, 1),
        pos_u16=b64(pq), region_u8=b64(region), top_u8=b64(is_top), act_i8=b64(q), act_scale=v,
        region_names=REGION_NAMES, region_colors=REGION_COLORS,
        types=conn.type[sel].tolist(), step=int(d["step"]),
        steer=[float(d["steer"][t]) for t in frames], lat=[float(d["lat"][t]) for t in frames],
        speed=[float(d["speed"][t]) for t in frames], thumbs=thumbs,
        n_total=int(conn.n), n_soma=int(np.isfinite(xyz[:, 0]).sum()),
    )
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {out}: {len(sel):,} neurons x {n_frames} frames, {os.path.getsize(out)/1e6:.1f} MB")


# ---------------------------------------------------------------------------


def video(conn, d, xyz, out, fps=20, seed=11, tilt_deg=22.0):
    """A rotating view of the soma cloud, lit by activity, beside the dashcam.

    Done as an orthographic projection drawn in 2D: the cloud is rotated about
    its vertical axis each frame, painter-sorted by depth, and drawn large and
    centred -- matplotlib's 3D axes squash a 140k-point cloud into an
    unreadable edge-on smear.
    """
    import imageio.v2 as imageio
    from flydrive.eye import PinholeCamera
    from flydrive.world import Renderer, TrackSet

    sel, delta = pick_neurons(conn, xyz, d)
    has = np.where(np.isfinite(xyz[:, 0]))[0]
    P = xyz[has].copy(); P[:, 1] = -P[:, 1]                    # y up
    P -= P.mean(0); P /= np.abs(P).max()
    sel_pos = {i: n for n, i in enumerate(has)}
    sel_rows = np.array([sel_pos[i] for i in sel])
    v = float(np.percentile(np.abs(delta[:, sel]), 99.5)) + 1e-9
    T = d["R"].shape[0]
    ce, se = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))

    cfg = EnvConfig(n_envs=8)
    tracks = TrackSet(cfg.track, device=DEV, seed=seed)
    cam = PinholeCamera(920, 420, fov_deg=104.0, pitch_deg=-5.0, device=DEV)
    rend = Renderer(cam, tracks, cfg.render, cfg.car.eye_height, device=DEV, seed=seed)

    fig = viz.new_figure(16, 9)
    ax_cam = fig.add_axes([0.03, 0.56, 0.40, 0.34]); ax_cam.set_xticks([]); ax_cam.set_yticks([])
    for sp in ax_cam.spines.values(): sp.set_color(viz.GRID)
    im = ax_cam.imshow(np.zeros((cam.n_row, cam.n_col, 3)), interpolation="bilinear")
    hud = ax_cam.text(0.015, 0.97, "", transform=ax_cam.transAxes, color="w", fontsize=9, va="top", family="monospace",
                      bbox=dict(boxstyle="round,pad=0.4", fc="#000000aa", ec="#ffffff33"))
    ax_tr = fig.add_axes([0.05, 0.10, 0.38, 0.34]); viz.style_axes(ax_tr, "steering  /  lane offset (m / 5)")
    l_st, = ax_tr.plot([], [], color=viz.BAD, lw=1.4); l_lat, = ax_tr.plot([], [], color=viz.VIOLET, lw=1.2)
    ax_tr.set_ylim(-1.1, 1.1); ax_tr.axhline(0, color=viz.MUTED, lw=0.6)
    ax = fig.add_axes([0.44, 0.03, 0.55, 0.90]); ax.set_facecolor(viz.BG); ax.set_axis_off()
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
    under = ax.scatter(P[:, 0], P[:, 1], s=0.35, c="#2a3038", linewidths=0, zorder=1)
    act = ax.scatter(P[sel_rows, 0], P[sel_rows, 1], s=1, c=np.zeros(len(sel)), cmap=bm.DARK_DIV, vmin=-v, vmax=v, linewidths=0, zorder=2)
    title = fig.suptitle("", color=viz.FG, fontsize=13, x=0.03, ha="left", y=0.965)
    fig.text(0.03, 0.925, f"whole MaleCNS connectome, brain frozen, readout trained to {int(d['step']):,} steps   |   "
             "colour: firing-rate change vs eyes closed, size: its magnitude", color=viz.MUTED, fontsize=8.5, ha="left")

    def project(angle):
        ca, sa = math.cos(angle), math.sin(angle)
        x = P[:, 0] * ca + P[:, 2] * sa
        depth = -P[:, 0] * sa + P[:, 2] * ca
        y = P[:, 1] * ce + depth * se
        return x, y, depth

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8, ffmpeg_log_level="error")
    try:
        for t in range(T):
            pos = torch.tensor(d["pos"][t:t + 1], device=DEV, dtype=torch.float32)
            yaw = torch.tensor(d["yaw"][t:t + 1], device=DEV, dtype=torch.float32)
            ti = torch.tensor(d["track"][t:t + 1], device=DEV, dtype=torch.long)
            _, _, rgb = rend.render(pos, yaw, ti, rgb=True)
            im.set_data(rgb[0, 0].cpu().numpy())
            hud.set_text(f"{d['speed'][t]*3.6:5.1f} km/h\nlane offset {d['lat'][t]:+5.2f} m\nsteer {d['steer'][t]:+.2f}")
            x, y, depth = project(-1.0 + 2 * math.pi * t / T)
            under.set_offsets(np.c_[x, y])
            dl = delta[t, sel]
            o = np.argsort(depth[sel_rows])                          # far first, near on top
            act.set_offsets(np.c_[x[sel_rows][o], y[sel_rows][o]]); act.set_array(dl[o])
            act.set_sizes(0.8 + 30.0 * np.minimum(np.abs(dl[o]) / v, 1.0) ** 0.8)
            lo = max(0, t - 200); tt = np.arange(lo, t + 1) * 0.05
            l_st.set_data(tt, d["steer"][lo:t + 1]); l_lat.set_data(tt, d["lat"][lo:t + 1] / 5.0); ax_tr.set_xlim(tt[0], tt[-1] + 1e-3)
            title.set_text(f"the whole connectome driving   |   t = {t*0.05:5.1f} s")
            fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if t % 60 == 0: print(f"  frame {t}/{T}", flush=True)
    finally:
        writer.close()
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cns_frozen/best.pt")
    ap.add_argument("--circuit", action="store_true"); ap.add_argument("--export", action="store_true"); ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    conn, d, xyz = load_all(a.ckpt)
    if a.circuit: circuit(conn, d, xyz, "runs/brain_circuit.png")
    if a.export: export(conn, d, xyz, "runs/brain3d_data.json")
    if a.video: video(conn, d, xyz, "runs/brain_3d.mp4")


if __name__ == "__main__":
    main()
