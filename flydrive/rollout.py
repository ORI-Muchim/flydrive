"""Record an episode and render it as a multi-panel neural video.

One frame shows, at the same instant: where the car is, what the retinas see,
the optic-flow field the T4/T5 array computes from it, the heading bump in the
central complex, the lobula-plate wide-field responses, and the motor command
that comes out the other end.
"""

from __future__ import annotations

import math

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import viz
from .viz import (ACCENT, BAD, BG, DIV_CMAP, FG, FLOW_CMAP, GOOD, GRID, MUTED,
                  PANEL, VIOLET, WARM, FlyEyePanel, RingPanel, style_axes)


class RolloutRecorder:
    """Runs a deterministic episode and paints it frame by frame."""

    def __init__(self, env, brain, width=1600, height=900, dpi=100, trail=160):
        self.env = env
        self.brain = brain
        self.trail = trail
        self.fig = viz.new_figure(width / dpi, height / dpi)
        self.dpi = dpi
        self._build()

    # -- figure scaffolding ------------------------------------------------

    def _build(self):
        fig = self.fig
        gs = fig.add_gridspec(
            3, 3, width_ratios=[1.0, 1.25, 1.25], height_ratios=[1.0, 1.0, 0.80],
            left=0.035, right=0.985, top=0.905, bottom=0.06, wspace=0.20, hspace=0.34)

        # top-down world view
        self.ax_map = fig.add_subplot(gs[0:2, 0])
        style_axes(self.ax_map, "world  |  top-down")
        self.ax_map.set_aspect("equal")
        self.road_patch = self.ax_map.fill([], [], color="#2b3240", lw=0, zorder=1)[0]
        self.centre_line, = self.ax_map.plot([], [], color=MUTED, lw=0.6, ls="--", zorder=2)
        self.post_scatter = self.ax_map.scatter([], [], s=4, c="none", zorder=3)
        self.trail_line, = self.ax_map.plot([], [], color=ACCENT, lw=1.4, alpha=0.85, zorder=4)
        self.car_dot, = self.ax_map.plot([], [], marker=(3, 0, 0), ms=11,
                                         color=WARM, zorder=6, ls="none")
        self.map_txt = self.ax_map.text(0.02, 0.975, "", transform=self.ax_map.transAxes,
                                        color=FG, fontsize=7.5, va="top", family="monospace")

        # retina
        self.ax_eye = fig.add_subplot(gs[0, 1:])
        self.eye_panel = FlyEyePanel(self.ax_eye, self.env.eye, cmap="gray")
        self.ax_eye.set_title("R1-R6 photoreceptors  |  1728 ommatidia, hexagonal lattice",
                              color=FG, fontsize=9, loc="left", pad=4)

        # T4/T5 optic flow
        self.ax_flow = fig.add_subplot(gs[1, 1:])
        self.flow_panel = FlyEyePanel(self.ax_flow, self.env.eye, cmap=FLOW_CMAP)
        self.ax_flow.set_title(
            "T4/T5 elementary motion detectors  |  colour = |flow|, arrows = direction",
            color=FG, fontsize=9, loc="left", pad=4)

        # ring attractor
        self.ax_ring = fig.add_subplot(gs[2, 0], projection="polar")
        self.ring_panel = RingPanel(self.ax_ring, self.brain.cfg.n_epg)
        self.ax_ring.set_title("central complex  |  EPG heading bump",
                               color=FG, fontsize=8.5, loc="left", pad=10)

        # lobula plate tangential cells
        self.ax_lptc = fig.add_subplot(gs[2, 1])
        style_axes(self.ax_lptc, "lobula plate  |  HS + VS wide-field cells")
        names = (["HSN", "HSE", "HSS"] + [f"VS{i+1}" for i in range(self.brain.cfg.n_vs)])
        self.lptc_names = names
        y = np.arange(len(names))
        self.bars_l = self.ax_lptc.barh(y - 0.2, np.zeros(len(names)), height=0.38,
                                        color=ACCENT, label="left eye")
        self.bars_r = self.ax_lptc.barh(y + 0.2, np.zeros(len(names)), height=0.38,
                                        color=WARM, label="right eye")
        self.ax_lptc.set_yticks(y)
        self.ax_lptc.set_yticklabels(names, fontsize=6)
        self.ax_lptc.set_xlim(-1.05, 1.05)
        self.ax_lptc.invert_yaxis()
        self.ax_lptc.axvline(0, color=MUTED, lw=0.6)
        self.ax_lptc.legend(loc="lower right", fontsize=6, facecolor=PANEL,
                            edgecolor=GRID, labelcolor=FG, framealpha=0.9)

        # motor output
        self.ax_motor = fig.add_subplot(gs[2, 2])
        style_axes(self.ax_motor, "descending neurons -> motor")
        self.tr_steer, = self.ax_motor.plot([], [], color=BAD, lw=1.4, label="steering")
        self.tr_thr, = self.ax_motor.plot([], [], color=GOOD, lw=1.4, label="throttle")
        self.tr_lat, = self.ax_motor.plot([], [], color=VIOLET, lw=1.1, label="lane offset/5m")
        self.ax_motor.set_ylim(-1.15, 1.15)
        self.ax_motor.axhline(0, color=MUTED, lw=0.6)
        self.ax_motor.legend(loc="upper left", fontsize=6, ncol=3, facecolor=PANEL,
                             edgecolor=GRID, labelcolor=FG, framealpha=0.9)
        self.ax_motor.set_xlabel("time (s)", color=MUTED, fontsize=7)

        self.title = fig.suptitle("", color=FG, fontsize=13, x=0.035, ha="left", y=0.965)
        self.subtitle = fig.text(0.035, 0.925, "", color=MUTED, fontsize=8.5, ha="left")

    # -- per-frame updates -------------------------------------------------

    def _draw_track(self, ti):
        env = self.env
        c = env.tracks.centre[ti].cpu().numpy()
        n = env.tracks.normal[ti].cpu().numpy()
        hw = env.cfg.track.road_width / 2
        poly = np.concatenate([c + n * hw, (c - n * hw)[::-1]])
        self.road_patch.set_xy(poly)
        self.centre_line.set_data(c[:, 0], c[:, 1])
        p = env.tracks.post_xy[ti].cpu().numpy()
        pc = env.tracks.post_col[ti].cpu().numpy()
        self.post_scatter.set_offsets(p)
        self.post_scatter.set_color(["#e6edf3" if v > 0.5 else "#30363d" for v in pc])
        pad = 18
        self.ax_map.set_xlim(c[:, 0].min() - pad, c[:, 0].max() + pad)
        self.ax_map.set_ylim(c[:, 1].min() - pad, c[:, 1].max() + pad)

    @torch.no_grad()
    def record(self, path=None, n_steps=420, env_index=0, fps=20, title="", subtitle="",
               deterministic=True, progress=False, png=None):
        """Drive one episode.

        ``path`` writes an mp4; ``png`` saves the final frame.  Passing only
        ``png`` is the cheap option used between training iterations.
        """
        env, brain = self.env, self.brain
        dev = env.device
        k = env_index
        env.reset_all()
        state = brain.init_state(env.B, dev)
        obs, prop = env.observe(), env.proprio()

        cur_track = -1
        xs, ys, st_hist, th_hist, lat_hist, t_hist = [], [], [], [], [], []
        flow_scale = 1.0
        writer = (imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                     macro_block_size=8, ffmpeg_log_level="error")
                  if path else None)
        crashes = 0
        try:
            for step in range(n_steps):
                action, _, value, state, tel = brain.act(
                    obs, prop, state, deterministic=deterministic, telemetry=True)
                obs, prop, reward, done, info = env.step(action)
                if done.any():
                    state.reset_(done, brain.init_state(env.B, dev))

                if hasattr(env, "draw_map"):
                    env.draw_map(self.ax_map, k)
                    if bool(done[k]): xs.clear(); ys.clear()
                else:
                    ti = int(env.track_idx[k])
                    if ti != cur_track:
                        self._draw_track(ti)
                        cur_track = ti
                        xs.clear(); ys.clear()
                if bool(done[k]):
                    crashes += int(bool(info["off_road"][k]))

                # --- world -------------------------------------------------
                px, py = float(env.pos[k, 0]), float(env.pos[k, 1])
                xs.append(px); ys.append(py)
                if len(xs) > self.trail:
                    xs.pop(0); ys.pop(0)
                self.trail_line.set_data(xs, ys)
                hd = float(env.heading[k])
                self.car_dot.set_data([px], [py])
                self.car_dot.set_marker((3, 0, math.degrees(hd) - 90))
                self.map_txt.set_text(
                    f"speed   {float(env.speed[k]):5.1f} m/s\n"
                    f"offset  {float(info['lateral'][k]):+5.2f} m\n"
                    f"heading {math.degrees(float(info['heading_err'][k])):+5.1f} deg\n"
                    f"lap     {float(env.lap_arc[k]):6.0f} m\n"
                    f"value   {float(value[k]):+6.2f}")

                # --- retina ------------------------------------------------
                self.eye_panel.set(tel["image"][k].cpu().numpy())

                # --- T4/T5 optic flow --------------------------------------
                T4 = tel["T4"][k].cpu().numpy()      # (4, 2, H, W) subtypes a,b,c,d
                T5 = tel["T5"][k].cpu().numpy()
                vx = (T4[0] + T5[0]) - (T4[1] + T5[1])     # back-ward minus forward
                vy = (T4[2] + T5[2]) - (T4[3] + T5[3])     # up minus down
                mag = np.hypot(vx, vy)
                p95 = float(np.percentile(mag, 97)) + 1e-6
                flow_scale = 0.9 * flow_scale + 0.1 * p95
                self.flow_panel.set(mag)
                self.flow_panel.set_clim(0, max(flow_scale, 1e-3) * 1.6)
                self.flow_panel.quiver(vx, vy, stride=2,
                                       scale=max(flow_scale, 1e-3) / 9.0)

                # --- central complex ---------------------------------------
                self.ring_panel.set(tel["epg"][k].cpu().numpy(),
                                    float(tel["goal_ang"][k]))

                # --- lobula plate ------------------------------------------
                hs = tel["hs"][k].cpu().numpy()      # (2, n_hs)
                vs = tel["vs"][k].cpu().numpy()
                left = np.concatenate([hs[0], vs[0]])
                right = np.concatenate([hs[1], vs[1]])
                scale = max(np.abs(np.concatenate([left, right])).max(), 1e-3)
                for b, v in zip(self.bars_l, left / scale):
                    b.set_width(v)
                for b, v in zip(self.bars_r, right / scale):
                    b.set_width(v)

                # --- motor --------------------------------------------------
                t_hist.append(step * env.cfg.dt)
                st_hist.append(float(action[k, 0]))
                th_hist.append(float(action[k, 1]))
                lat_hist.append(float(info["lateral"][k]) / 5.0)
                w = 220
                self.tr_steer.set_data(t_hist[-w:], st_hist[-w:])
                self.tr_thr.set_data(t_hist[-w:], th_hist[-w:])
                self.tr_lat.set_data(t_hist[-w:], lat_hist[-w:])
                self.ax_motor.set_xlim(t_hist[max(0, len(t_hist) - w)], t_hist[-1] + 1e-3)

                self.title.set_text(title or "fly brain driving")
                self.subtitle.set_text(
                    f"{subtitle}    t = {step * env.cfg.dt:5.1f} s    crashes = {crashes}")

                # Drawing is the expensive part, so a snapshot-only run paints
                # just the final frame.
                if writer is not None or step == n_steps - 1:
                    self.fig.canvas.draw()
                if writer is not None:
                    writer.append_data(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])
                if progress and step % 50 == 0:
                    print(f"  frame {step}/{n_steps}", flush=True)
        finally:
            if writer is not None:
                writer.close()
        if png:
            self.fig.savefig(png, dpi=self.dpi, facecolor=viz.BG)
        return {"crashes": crashes, "distance": float(env.lap_arc[k])}

    def close(self):
        plt.close(self.fig)


class DriveCamRecorder:
    """The human-readable view: a colour camera at the same eye position.

    Same world, same renderer, same instant -- only the ray pattern differs.
    The fly's hexagonal mosaic sits underneath so the two can be compared
    directly.
    """

    def __init__(self, env, brain, width=1600, height=900, dpi=100,
                 cam_w=920, cam_h=420, fov=104.0, trail=260):
        from .eye import PinholeCamera
        from .world import Renderer

        self.env = env
        self.brain = brain
        self.trail = trail
        self.dpi = dpi
        self.camera = PinholeCamera(cam_w, cam_h, fov_deg=fov, pitch_deg=-5.0,
                                    device=str(env.device))
        self.cam_rend = Renderer(self.camera, getattr(env, "render_world", env.tracks), env.cfg.render,
                                 env.cfg.car.eye_height, device=str(env.device),
                                 seed=env.seed)
        self.fig = viz.new_figure(width / dpi, height / dpi)
        self._build()

    def _build(self):
        fig = self.fig
        gs = fig.add_gridspec(2, 2, width_ratios=[2.35, 1.0], height_ratios=[1.75, 1.0],
                              left=0.025, right=0.978, top=0.905, bottom=0.065,
                              wspace=0.13, hspace=0.20)

        # --- camera ------------------------------------------------------
        self.ax_cam = fig.add_subplot(gs[0, 0])
        self.im = self.ax_cam.imshow(np.zeros((self.camera.n_row, self.camera.n_col, 3)),
                                     interpolation="bilinear", zorder=1)
        self.ax_cam.set_xticks([]); self.ax_cam.set_yticks([])
        for s in self.ax_cam.spines.values():
            s.set_color(GRID)
        self.ax_cam.set_title("driver's-eye camera  |  the same world, sampled on a "
                              "perspective grid instead of a hex lattice",
                              color=FG, fontsize=9.5, loc="left", pad=5)
        self.hud = self.ax_cam.text(0.015, 0.975, "", transform=self.ax_cam.transAxes,
                                    color="#ffffff", fontsize=10, va="top",
                                    family="monospace", zorder=5,
                                    bbox=dict(boxstyle="round,pad=0.45", fc="#000000aa",
                                              ec="#ffffff33", lw=0.8))
        # steering indicator across the bottom of the camera frame
        self.steer_bar, = self.ax_cam.plot([], [], color=BAD, lw=5, solid_capstyle="round",
                                           transform=self.ax_cam.transAxes, zorder=5)
        self.ax_cam.plot([0.5, 0.5], [0.045, 0.075], color="#ffffff66", lw=1.2,
                         transform=self.ax_cam.transAxes, zorder=4)
        self.ax_cam.plot([0.25, 0.75], [0.06, 0.06], color="#ffffff28", lw=2.5,
                         transform=self.ax_cam.transAxes, zorder=3)

        # --- top-down map -------------------------------------------------
        self.ax_map = fig.add_subplot(gs[0, 1])
        style_axes(self.ax_map, "track")
        self.ax_map.set_aspect("equal")
        self.road_patch = self.ax_map.fill([], [], color="#2b3240", lw=0, zorder=1)[0]
        self.centre_line, = self.ax_map.plot([], [], color=MUTED, lw=0.6, ls="--", zorder=2)
        self.post_scatter = self.ax_map.scatter([], [], s=3, c="none", zorder=3)
        self.trail_line, = self.ax_map.plot([], [], color=ACCENT, lw=1.6, alpha=0.9, zorder=4)
        self.car_dot, = self.ax_map.plot([], [], marker=(3, 0, 0), ms=12,
                                         color=WARM, zorder=6, ls="none")

        # --- fly eye ------------------------------------------------------
        self.ax_eye = fig.add_subplot(gs[1, 0])
        self.eye_panel = FlyEyePanel(self.ax_eye, self.env.eye, cmap="gray")
        self.ax_eye.set_title("what the fly actually gets  |  1728 ommatidia, achromatic, "
                              "5.7 deg acceptance angle",
                              color=FG, fontsize=9.5, loc="left", pad=5)

        # --- telemetry ----------------------------------------------------
        self.ax_tel = fig.add_subplot(gs[1, 1])
        style_axes(self.ax_tel, "speed and control")
        self.tr_speed, = self.ax_tel.plot([], [], color=GOOD, lw=1.5, label="speed / 14 m/s")
        self.tr_steer, = self.ax_tel.plot([], [], color=BAD, lw=1.4, label="steering")
        self.tr_lat, = self.ax_tel.plot([], [], color=VIOLET, lw=1.2, label="lane offset / 5 m")
        self.ax_tel.set_ylim(-1.1, 1.1)
        self.ax_tel.axhline(0, color=MUTED, lw=0.6)
        self.ax_tel.legend(loc="lower left", fontsize=6.5, ncol=1, facecolor=PANEL,
                           edgecolor=GRID, labelcolor=FG, framealpha=0.9)
        self.ax_tel.set_xlabel("time (s)", color=MUTED, fontsize=7)

        self.title = fig.suptitle("", color=FG, fontsize=14, x=0.025, ha="left", y=0.968)
        self.subtitle = fig.text(0.025, 0.928, "", color=MUTED, fontsize=9, ha="left")

    def _draw_track(self, ti):
        env = self.env
        c = env.tracks.centre[ti].cpu().numpy()
        n = env.tracks.normal[ti].cpu().numpy()
        hw = env.cfg.track.road_width / 2
        self.road_patch.set_xy(np.concatenate([c + n * hw, (c - n * hw)[::-1]]))
        self.centre_line.set_data(c[:, 0], c[:, 1])
        p = env.tracks.post_xy[ti].cpu().numpy()
        pc = env.tracks.post_col[ti].cpu().numpy()
        self.post_scatter.set_offsets(p)
        self.post_scatter.set_color(["#e6edf3" if v > 0.5 else "#30363d" for v in pc])
        pad = 18
        self.ax_map.set_xlim(c[:, 0].min() - pad, c[:, 0].max() + pad)
        self.ax_map.set_ylim(c[:, 1].min() - pad, c[:, 1].max() + pad)

    @torch.no_grad()
    def record(self, path=None, n_steps=400, env_index=0, fps=20, title="", subtitle="",
               deterministic=True, progress=False, png=None):
        """``path`` writes an mp4; ``png`` saves the final frame.  Passing only
        ``png`` is the cheap option used between training iterations."""
        env, brain = self.env, self.brain
        k = env_index
        env.reset_all()
        state = brain.init_state(env.B, env.device)
        obs, prop = env.observe(), env.proprio()

        cur_track = -1
        xs, ys, t_h, v_h, s_h, l_h = [], [], [], [], [], []
        writer = (imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                                     macro_block_size=8, ffmpeg_log_level="error")
                  if path else None)
        crashes = 0
        try:
            for step in range(n_steps):
                action, _, value, state, tel = brain.act(
                    obs, prop, state, deterministic=deterministic, telemetry=True)
                obs, prop, reward, done, info = env.step(action)
                if done.any():
                    state.reset_(done, brain.init_state(env.B, env.device))
                    crashes += int(bool(info["off_road"][k] and done[k]))

                if hasattr(env, "draw_map"):
                    env.draw_map(self.ax_map, k)
                    if bool(done[k]): xs.clear(); ys.clear()
                else:
                    ti = int(env.track_idx[k])
                    if ti != cur_track:
                        self._draw_track(ti); cur_track = ti; xs.clear(); ys.clear()

                # --- colour camera ------------------------------------------
                tidx = env.track_idx[k:k + 1] if hasattr(env, "track_idx") else env.zero_idx[k:k + 1]
                _, _, rgb = self.cam_rend.render(env.pos[k:k + 1], env.heading[k:k + 1], tidx, rgb=True,
                                                 **(env.scene_for(k) if hasattr(env, "scene_for") else {}))
                self.im.set_data(rgb[0, 0].cpu().numpy())

                spd = float(env.speed[k]); st = float(action[k, 0]); lat = float(info["lateral"][k])
                self.hud.set_text(f"{spd*3.6:5.1f} km/h   ({spd:4.1f} m/s)\n"
                                  f"lane offset {lat:+5.2f} m\n"
                                  f"lap {float(env.lap_arc[k]):6.0f} m")
                x0 = 0.5
                self.steer_bar.set_data([x0, x0 + 0.25 * st], [0.06, 0.06])

                # --- map -----------------------------------------------------
                px, py = float(env.pos[k, 0]), float(env.pos[k, 1])
                xs.append(px); ys.append(py)
                if len(xs) > self.trail:
                    xs.pop(0); ys.pop(0)
                self.trail_line.set_data(xs, ys)
                self.car_dot.set_data([px], [py])
                self.car_dot.set_marker((3, 0, math.degrees(float(env.heading[k])) - 90))
                if hasattr(env, "draw_map"):
                    self.ax_map.set_xlim(px - 110, px + 110); self.ax_map.set_ylim(py - 110, py + 110)

                # --- fly eye --------------------------------------------------
                self.eye_panel.set(tel["image"][k].cpu().numpy())

                # --- telemetry ------------------------------------------------
                t_h.append(step * env.cfg.dt)
                v_h.append(spd / env.cfg.car.v_max)
                s_h.append(st)
                l_h.append(lat / 5.0)
                w = 240
                self.tr_speed.set_data(t_h[-w:], v_h[-w:])
                self.tr_steer.set_data(t_h[-w:], s_h[-w:])
                self.tr_lat.set_data(t_h[-w:], l_h[-w:])
                self.ax_tel.set_xlim(t_h[max(0, len(t_h) - w)], t_h[-1] + 1e-3)

                self.title.set_text(title or "a fruit fly's visual system driving a car")
                self.subtitle.set_text(f"{subtitle}    t = {step*env.cfg.dt:5.1f} s    "
                                       f"crashes = {crashes}")

                if writer is not None or step == n_steps - 1:
                    self.fig.canvas.draw()
                if writer is not None:
                    writer.append_data(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])
                if progress and step % 50 == 0:
                    print(f"  frame {step}/{n_steps}", flush=True)
        finally:
            if writer is not None:
                writer.close()
        if png:
            self.fig.savefig(png, dpi=self.dpi, facecolor=viz.BG)
        return {"crashes": crashes, "distance": float(env.lap_arc[k])}

    def close(self):
        plt.close(self.fig)
