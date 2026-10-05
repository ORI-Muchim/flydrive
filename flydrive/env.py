"""Vectorised driving environment seen through a pair of compound eyes.

Everything lives on the GPU and every environment steps in lockstep, so a
rollout is a handful of tensor ops rather than a Python loop over environments.
Observations are raw ommatidial luminance -- the agent gets no map, no track
geometry and no lane offset, only what the eyes deliver plus the proprioceptive
signals a fly would have from its halteres and legs.
"""

from __future__ import annotations

import math

import torch

from .config import EnvConfig
from .eye import CompoundEye
from .world import Renderer, TrackSet

TWO_PI = 2.0 * math.pi


class FlyDriveEnv:
    """A batch of cars driving closed-loop tracks under a kinematic bicycle model."""

    def __init__(self, cfg: EnvConfig | None = None, device="cuda", seed: int = 0):
        self.cfg = cfg or EnvConfig()
        self.device = torch.device(device)
        self.B = self.cfg.n_envs
        self.seed = seed
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        self.tracks = TrackSet(self.cfg.track, device=device, seed=seed)
        self.eye = CompoundEye(self.cfg.eye, device=device)
        self.renderer = Renderer(self.eye, self.tracks, self.cfg.render,
                                 self.cfg.car.eye_height, device=device, seed=seed)

        B = self.B
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.pos = z(B, 2)
        self.heading = z(B)
        self.speed = z(B)
        self.steer = z(B)
        self.track_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.lap_arc = z(B)          # cumulative signed progress, metres
        self.prev_arc = z(B)
        self.yaw_rate = z(B)
        self.lat_accel = z(B)
        self.ep_return = z(B)
        self.ep_len = torch.zeros(B, dtype=torch.long, device=self.device)
        self.stall = torch.zeros(B, dtype=torch.long, device=self.device)
        # Rolling statistics of the episodes that have finished.
        self.stats = {"return": [], "length": [], "distance": [], "crash": [], "stall": []}

        self.reset_all()

    # -- properties ------------------------------------------------------

    @property
    def obs_shape(self):
        return (2, self.eye.n_row, self.eye.n_col)

    @property
    def action_dim(self):
        return 2

    # -- reset -----------------------------------------------------------

    def reset_all(self):
        idx = torch.arange(self.B, device=self.device)
        self._respawn(idx)
        self.ep_return.zero_()
        self.ep_len.zero_()
        return self.observe()

    def _respawn(self, idx: torch.Tensor):
        """Place the selected cars at a random point on a random track."""
        n = idx.numel()
        if n == 0:
            return
        cfg = self.cfg
        ti = torch.randint(0, self.tracks.n_tracks, (n,), device=self.device, generator=self.gen)
        si = torch.randint(0, self.tracks.n_points, (n,), device=self.device, generator=self.gen)

        centre = self.tracks.centre[ti, si]
        normal = self.tracks.normal[ti, si]
        tangent = self.tracks.tangent[ti, si]
        # Start off-centre and slightly mis-aligned so the policy has to correct.
        lat = (torch.rand(n, device=self.device, generator=self.gen) - 0.5) * cfg.track.road_width * 0.5
        yaw = (torch.rand(n, device=self.device, generator=self.gen) - 0.5) * 0.4

        self.track_idx[idx] = ti
        self.pos[idx] = centre + normal * lat.unsqueeze(-1)
        self.heading[idx] = torch.atan2(tangent[:, 1], tangent[:, 0]) + yaw
        self.speed[idx] = cfg.car.v_init * (0.6 + 0.6 * torch.rand(n, device=self.device, generator=self.gen))
        self.steer[idx] = 0.0
        self.t[idx] = 0
        self.yaw_rate[idx] = 0.0
        self.lat_accel[idx] = 0.0
        self.stall[idx] = 0
        self.lap_arc[idx] = 0.0
        _, _, arc, _ = self.tracks.nearest_centre(self.track_idx[idx], self.pos[idx])
        self.prev_arc[idx] = arc

    @property
    def render_world(self):
        return self.tracks

    def scene_for(self, k):
        return {}

    # -- observation -----------------------------------------------------

    @torch.no_grad()
    def observe(self):
        image, _ = self.renderer.render(self.pos, self.heading, self.track_idx)
        return image

    @torch.no_grad()
    def proprio(self):
        """Speed, lateral acceleration and yaw rate, each roughly unit-scaled."""
        return torch.stack([
            self.speed / self.cfg.car.v_max,
            self.lat_accel / 9.81,
            self.yaw_rate,
        ], dim=1)

    # -- dynamics --------------------------------------------------------

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """Advance every environment by one control step.

        Args:
            action: ``(B, 2)`` in [-1, 1]: steering and throttle.
        Returns:
            ``(obs, proprio, reward, done, info)``.
        """
        cfg, car = self.cfg, self.cfg.car
        act = action.clamp(-1.0, 1.0)
        steer_cmd = act[:, 0] * car.max_steer
        throttle = act[:, 1]

        prev_steer = self.steer
        # First-order steering actuator: the wheels cannot snap instantly.
        self.steer = prev_steer + (steer_cmd - prev_steer) * 0.45

        accel = torch.where(throttle >= 0, throttle * car.max_accel, throttle * car.max_brake)
        speed = (self.speed + (accel - car.drag * self.speed) * cfg.dt)
        speed = speed.clamp(car.v_min, car.v_max)

        yaw_rate = speed / car.wheelbase * torch.tan(self.steer)
        heading = self.heading + yaw_rate * cfg.dt
        pos = self.pos + torch.stack([torch.cos(heading), torch.sin(heading)], dim=1) * (speed * cfg.dt).unsqueeze(-1)

        self.lat_accel = speed * yaw_rate
        self.yaw_rate = yaw_rate
        self.speed, self.heading, self.pos = speed, torch.remainder(heading + math.pi, TWO_PI) - math.pi, pos

        # -- progress along the centreline --------------------------------
        _, lat, arc, tangent = self.tracks.nearest_centre(self.track_idx, self.pos)
        length = self.tracks.length[self.track_idx]
        d_arc = arc - self.prev_arc
        # Unwrap the lap seam.
        d_arc = torch.where(d_arc > length * 0.5, d_arc - length, d_arc)
        d_arc = torch.where(d_arc < -length * 0.5, d_arc + length, d_arc)
        self.prev_arc = arc
        self.lap_arc = self.lap_arc + d_arc

        heading_err = torch.remainder(
            self.heading - torch.atan2(tangent[:, 1], tangent[:, 0]) + math.pi, TWO_PI) - math.pi

        half = cfg.track.road_width * 0.5
        off_road = lat.abs() > (half + cfg.off_road_margin)
        # Standing still is otherwise a safe local optimum: no progress, but no
        # crash either.  Time it out.
        self.stall = torch.where(self.speed < cfg.stall_speed, self.stall + 1,
                                 torch.zeros_like(self.stall))
        stalled = self.stall >= cfg.stall_patience

        reward = (
            cfg.w_progress * d_arc
            - cfg.w_lateral * (lat / half) ** 2
            - cfg.w_heading * heading_err ** 2
            - cfg.w_steer * self.steer ** 2
            - cfg.w_jerk * (self.steer - prev_steer) ** 2 / cfg.dt
        )
        reward = torch.where(off_road, reward - cfg.crash_penalty, reward)
        reward = torch.where(stalled, reward - cfg.stall_penalty, reward)

        self.t += 1
        timeout = self.t >= cfg.max_steps
        done = off_road | timeout | stalled

        self.ep_return += reward
        self.ep_len += 1

        info = {
            "lateral": lat, "heading_err": heading_err, "off_road": off_road,
            "timeout": timeout, "stalled": stalled,
            "progress": d_arc, "lap_arc": self.lap_arc.clone(),
            "speed": self.speed.clone(),
        }

        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            self.stats["return"].append(self.ep_return[idx].detach().clone())
            self.stats["length"].append(self.ep_len[idx].detach().clone())
            self.stats["distance"].append(self.lap_arc[idx].detach().clone())
            self.stats["crash"].append(off_road[idx].float().detach().clone())
            self.stats["stall"].append(stalled[idx].float().detach().clone())
            self.ep_return[idx] = 0.0
            self.ep_len[idx] = 0
            self._respawn(idx)

        return self.observe(), self.proprio(), reward, done, info

    # -- statistics -------------------------------------------------------

    def pop_stats(self):
        """Drain and summarise the episodes that finished since the last call."""
        if not self.stats["return"]:
            return None
        out = {}
        for k, v in self.stats.items():
            out[k] = torch.cat(v).float().mean().item()
        out["n_episodes"] = int(sum(v.numel() for v in self.stats["return"]))
        for k in self.stats:
            self.stats[k] = []
        return out
