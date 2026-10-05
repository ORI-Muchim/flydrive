"""The city environment: signalled intersections, turn commands, other cars.

Same interface as :class:`flydrive.env.FlyDriveEnv` -- the observation is still
only what the compound eyes deliver -- plus three command channels in the
proprioceptive vector (left / straight / right) that fade in as the next
intersection approaches, the way a goal direction would reach the central
complex.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from .city import CityConfig, CityMap, RouteSet, LEFT, STRAIGHT, RIGHT
from .config import EnvConfig
from .eye import CompoundEye
from .world import Renderer

TWO_PI = 2.0 * math.pi


class CityEnv:
    def __init__(self, cfg: EnvConfig | None = None, city_cfg: CityConfig | None = None,
                 device="cuda", seed: int = 0):
        base = cfg or EnvConfig()
        self.cfg = replace(base, n_proprio=6, max_steps=max(base.max_steps, 3200))
        self.ccfg = city_cfg or CityConfig()
        self.device = torch.device(device)
        self.B = self.cfg.n_envs
        self.seed = seed
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        self.city = CityMap(self.ccfg, device=device, seed=seed)
        self.routes = RouteSet(self.city, device=device, seed=seed)
        self.eye = CompoundEye(self.cfg.eye, device=device)
        rcfg = replace(self.cfg.render, n_posts_visible=6)
        self.renderer = Renderer(self.eye, self.city, rcfg, self.cfg.car.eye_height, device=device, seed=seed)
        self.zero_idx = torch.zeros(self.B, dtype=torch.long, device=self.device)

        B, K = self.B, self.ccfg.n_npc
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.pos = z(B, 2); self.heading = z(B); self.speed = z(B); self.steer = z(B)
        self.route = torch.zeros(B, dtype=torch.long, device=self.device)
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.clock = z(B)
        self.lap_arc = z(B); self.prev_arc = z(B); self.yaw_rate = z(B); self.lat_accel = z(B)
        self.stall = torch.zeros(B, dtype=torch.long, device=self.device)
        self.ep_return = z(B); self.ep_len = torch.zeros(B, dtype=torch.long, device=self.device)
        self.ep_red = z(B)
        self.npc_route = torch.zeros(B, K, dtype=torch.long, device=self.device)
        self.npc_s = z(B, K); self.npc_v = z(B, K); self.npc_lum = z(B, K)
        self.stats = {k: [] for k in ("return", "length", "distance", "crash", "stall", "collision", "red", "wrong_turn")}
        self.reset_all()

    def reseed(self, seed: int):
        """Rebuild the city (routes, signal phases, traffic, buildings) from a new
        seed and respawn every car.  Training in one layout overfits to it —
        94% on the training seed against 70% on a fresh one — so the trainer
        calls this every few iterations across a pool of layouts."""
        self.seed = seed
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.city = CityMap(self.ccfg, device=str(self.device), seed=seed)
        self.routes = RouteSet(self.city, device=str(self.device), seed=seed)
        rcfg = replace(self.cfg.render, n_posts_visible=6)
        self.renderer = Renderer(self.eye, self.city, rcfg, self.cfg.car.eye_height, device=str(self.device), seed=seed)
        return self.reset_all()

    # -- interface -----------------------------------------------------------

    @property
    def obs_shape(self):
        return (2, self.eye.n_row, self.eye.n_col)

    @property
    def action_dim(self):
        return 2

    @property
    def tracks(self):        # recorders reach for env.tracks.centre; give them the route set
        return self.routes

    def reset_all(self):
        idx = torch.arange(self.B, device=self.device)
        self._respawn(idx)
        self.ep_return.zero_(); self.ep_len.zero_(); self.ep_red.zero_()
        return self.observe()

    def _respawn(self, idx):
        n = idx.numel()
        if n == 0:
            return
        g, dev, cfg = self.gen, self.device, self.cfg
        ri = torch.randint(0, self.routes.n_tracks, (n,), device=dev, generator=g)
        s0 = 6.0 + 24.0 * torch.rand(n, device=dev, generator=g)
        if self.ccfg.corner_spawn_frac > 0:
            # Start some episodes just before the first corner, command already
            # on: a turn is then the only way to survive, so the policy sees
            # turning states it would never reach by uncorrelated exploration.
            first = self.routes.corner_arc[ri, 0]
            near = first - (15.0 + 20.0 * torch.rand(n, device=dev, generator=g))
            use = (torch.rand(n, device=dev, generator=g) < self.ccfg.corner_spawn_frac) & (first < 1e8) & (near > 6.0)
            s0 = torch.where(use, near, s0)
        p, tan = self.routes.point_at(ri, s0)
        nrm = torch.stack([-tan[:, 1], tan[:, 0]], -1)
        lat = (torch.rand(n, device=dev, generator=g) - 0.5) * 1.6
        self.route[idx] = ri
        self.pos[idx] = p + nrm * lat.unsqueeze(-1)
        self.heading[idx] = torch.atan2(tan[:, 1], tan[:, 0]) + (torch.rand(n, device=dev, generator=g) - 0.5) * 0.3
        spd = cfg.car.v_init * (0.5 + 0.6 * torch.rand(n, device=dev, generator=g))
        if self.ccfg.corner_spawn_frac > 0:
            spd = torch.where(use, 4.0 + 3.0 * torch.rand(n, device=dev, generator=g), spd)
        self.speed[idx] = spd
        self.steer[idx] = 0.0; self.t[idx] = 0
        self.clock[idx] = torch.rand(n, device=dev, generator=g) * self.ccfg.signal_period
        self.yaw_rate[idx] = 0.0; self.lat_accel[idx] = 0.0; self.stall[idx] = 0; self.lap_arc[idx] = 0.0
        _, _, arc, _ = self.routes.nearest_centre(self.route[idx], self.pos[idx])
        self.prev_arc[idx] = arc
        mask = torch.zeros(self.B, self.ccfg.n_npc, dtype=torch.bool, device=dev); mask[idx] = True
        self._respawn_npc(mask)

    def _respawn_npc(self, mask):
        n = int(mask.sum())
        if n == 0:
            return
        g, dev = self.gen, self.device
        ri = torch.randint(0, self.routes.n_tracks, (n,), device=dev, generator=g)
        L = self.routes.length[ri]
        s = 10.0 + torch.rand(n, device=dev, generator=g) * (L - 80.0)
        # Never drop a car within 30 m of the fly: a car materialising just
        # behind a stopped ego at 8 m/s is a collision nobody can drive around.
        ego = self.pos[torch.nonzero(mask)[:, 0]]
        for _ in range(8):
            p, _ = self.routes.point_at(ri, s)
            bad = torch.linalg.norm(p - ego, dim=-1) < 30.0
            if not bad.any():
                break
            nb = int(bad.sum())
            ri[bad] = torch.randint(0, self.routes.n_tracks, (nb,), device=dev, generator=g)
            s[bad] = 10.0 + torch.rand(nb, device=dev, generator=g) * (self.routes.length[ri[bad]] - 80.0)
        self.npc_route[mask] = ri; self.npc_s[mask] = s
        self.npc_v[mask] = self.ccfg.npc_speed * (0.6 + 0.5 * torch.rand(n, device=dev, generator=g))
        self.npc_lum[mask] = 0.15 + 0.75 * torch.rand(n, device=dev, generator=g)

    # -- scene assembly --------------------------------------------------------

    def _npc_pose(self):
        B, K = self.B, self.ccfg.n_npc
        p, tan = self.routes.point_at(self.npc_route.reshape(-1), self.npc_s.reshape(-1))
        return p.view(B, K, 2), tan.view(B, K, 2)

    def scene(self):
        """Dynamic objects for the renderer: lamp spheres, car boxes, post luminance."""
        B, c = self.B, self.ccfg
        state = self.city.signal_state(self.clock)                              # (B, P)
        lum = CityMap.lamp_luminance(state)
        d = torch.linalg.norm(self.city.lamp_xy.unsqueeze(0) - self.pos.unsqueeze(1), dim=-1)
        _, near = torch.topk(d, 4, dim=1, largest=False)
        b = torch.arange(B, device=self.device).unsqueeze(1)
        lxy = self.city.lamp_xy[near]                                             # (B,4,2)
        spheres = torch.cat([lxy, torch.full((B, 4, 1), c.lamp_height, device=self.device),
                             torch.full((B, 4, 1), c.lamp_radius, device=self.device),
                             lum[b, near].unsqueeze(-1)], -1)
        p, tan = self._npc_pose()
        yaw = torch.atan2(tan[..., 1], tan[..., 0])
        half = torch.tensor([c.car_length / 2, c.car_width / 2, c.car_height / 2], device=self.device)
        boxes = torch.cat([p, yaw.unsqueeze(-1), half.view(1, 1, 3).expand(B, c.n_npc, 3), self.npc_lum.unsqueeze(-1)], -1)
        bld = self.city.nearest_buildings(self.pos, c.n_buildings_visible)
        if bld is not None:
            boxes = torch.cat([boxes, bld], 1)
        post_lum = lum * 0.85 + 0.05          # the pole itself is lit with the signal
        return dict(spheres=spheres, boxes=boxes, post_lum=post_lum), state

    @property
    def render_world(self):
        return self.city

    def scene_for(self, k):
        sc, _ = self.scene()
        return {kk: v[k:k + 1] for kk, v in sc.items()}

    @torch.no_grad()
    def observe(self):
        sc, _ = self.scene()
        image, _ = self.renderer.render(self.pos, self.heading, self.zero_idx, **sc)
        return image

    @torch.no_grad()
    def proprio(self):
        _, _, arc, _ = self.routes.nearest_centre(self.route, self.pos, near=self.prev_arc)
        turn, dist = self.routes.next_corner(self.route, arc)
        # A ramp that reaches 1.0 at the corner itself, so the command carries
        # *when* to turn, not just that a turn is coming -- "turn in 50 m".
        gate = (1.0 - dist / self.ccfg.command_range).clamp(0.0, 1.0)
        # Inside the arc the command becomes the turn still to be made (1 -> 0
        # across the 90 degrees), the way a goal direction in the central
        # complex is compared against the heading: the policy is told how far
        # through the corner it is, not just that it is in one.
        R = torch.where(turn == RIGHT, torch.full_like(dist, self.ccfg.fillet_right), torch.full_like(dist, self.ccfg.fillet_left))
        _, t_exit = self.routes.point_at(self.route, arc + dist.clamp(max=1e6) + R * (math.pi / 2))
        remaining = torch.remainder(torch.atan2(t_exit[:, 1], t_exit[:, 0]) - self.heading + math.pi, TWO_PI) - math.pi
        gate = torch.where(dist <= 0, (remaining.abs() / (math.pi / 2)).clamp(0.0, 1.0), gate)
        cmd = torch.nn.functional.one_hot(turn.long(), 3).float() * gate.unsqueeze(-1)
        return torch.cat([torch.stack([self.speed / self.cfg.car.v_max, self.lat_accel / 9.81, self.yaw_rate], 1), cmd], 1)

    # -- dynamics --------------------------------------------------------------

    @torch.no_grad()
    def step(self, action):
        cfg, car, c = self.cfg, self.cfg.car, self.ccfg
        B, K, dev = self.B, c.n_npc, self.device
        act = action.clamp(-1, 1)
        prev_steer = self.steer
        self.steer = prev_steer + (act[:, 0] * car.max_steer - prev_steer) * 0.45
        accel = torch.where(act[:, 1] >= 0, act[:, 1] * car.max_accel, act[:, 1] * car.max_brake)
        v_cap = c.v_max if c.v_max is not None else car.v_max
        speed = (self.speed + (accel - car.drag * self.speed) * cfg.dt).clamp(car.v_min, v_cap)
        yaw_rate = speed / car.wheelbase * torch.tan(self.steer)
        heading = self.heading + yaw_rate * cfg.dt
        self.pos = self.pos + torch.stack([torch.cos(heading), torch.sin(heading)], 1) * (speed * cfg.dt).unsqueeze(-1)
        self.lat_accel, self.yaw_rate, self.speed = speed * yaw_rate, yaw_rate, speed
        self.heading = torch.remainder(heading + math.pi, TWO_PI) - math.pi
        self.clock = self.clock + cfg.dt

        # -- route progress, lane position ------------------------------------
        _, lat, arc, tangent = self.routes.nearest_centre(self.route, self.pos, near=self.prev_arc)
        d_arc = (arc - self.prev_arc).clamp(-5.0, 5.0)
        heading_err = torch.remainder(self.heading - torch.atan2(tangent[:, 1], tangent[:, 0]) + math.pi, TWO_PI) - math.pi
        lane = c.lane_offset
        off_road = (lat < -(lane + cfg.off_road_margin)) | (lat > (c.road_width - lane + cfg.off_road_margin))
        oncoming = lat > lane
        # Leaving the route while still on asphalt is a wrong turn, not a crash
        # into the verge -- the two say different things about the policy.
        on_asphalt = self.city.sample_field(None, self.pos)[0].abs() < c.road_width / 2 + cfg.off_road_margin
        wrong_turn = off_road & on_asphalt

        # -- v2b shaping: anticipate the route, reward the corner ------------------
        head_ahead = torch.zeros_like(heading_err)
        if c.lookahead > 0:
            _, t_ahead = self.routes.point_at(self.route, arc + c.lookahead)
            head_ahead = torch.remainder(self.heading - torch.atan2(t_ahead[:, 1], t_ahead[:, 0]) + math.pi, TWO_PI) - math.pi
        steer_term = torch.zeros_like(heading_err)
        if c.steer_reward > 0:
            turn_now, dist_now = self.routes.next_corner(self.route, arc)
            gate_now = (1.0 - dist_now / c.command_range).clamp(0.0, 1.0)
            direction = (turn_now == LEFT).float() - (turn_now == RIGHT).float()   # +steer turns left
            s_n = self.steer / car.max_steer
            steer_term = c.steer_reward * gate_now * torch.where(turn_now == STRAIGHT, -s_n.abs(), direction * s_n)
        corner_passed = torch.zeros_like(off_road)
        if c.corner_bonus > 0:
            _, d_c_prev = self.routes.next_corner(self.route, self.prev_arc)
            corner_passed = (d_c_prev >= 0) & (d_c_prev <= d_arc + 1e-3) & (lat.abs() < lane + cfg.off_road_margin)

        # -- signals ---------------------------------------------------------------
        state = self.city.signal_state(self.clock)                                # (B, P)
        d_prev, lamp = self.routes.next_stop(self.route, self.prev_arc)
        my_state = state[torch.arange(B, device=dev), lamp]
        crossed = (d_prev > 1e-3) & (d_prev <= d_arc + 1e-3)   # once: a car parked on the line is not re-crossing it
        red_run = crossed & (my_state < 0.5) & (self.speed > 1.0)   # creeping over the line while stopped is not running it
        d_stop, _ = self.routes.next_stop(self.route, arc)
        # Stopped for a red anywhere in the approach is waiting, not stalling
        # (a cautious driver that brakes 20 m early is slow, not stuck).
        waiting = (my_state < 1.5) & (d_stop < 25.0) & (d_stop > -3.0)

        # -- other cars ------------------------------------------------------------
        p_npc, t_npc = self._npc_pose()
        ds, lamp_n = self.routes.next_stop(self.npc_route.reshape(-1), self.npc_s.reshape(-1))
        ds = ds.view(B, K); st_n = state[torch.arange(B, device=dev).repeat_interleave(K), lamp_n].view(B, K)
        brake = self.npc_v ** 2 / 6.0 + 3.0
        must_stop = (st_n < 1.5) & (ds < brake) & (ds > -0.5)
        rel = self.pos.unsqueeze(1) - p_npc                                        # ego relative to npc
        ahead = (rel * t_npc).sum(-1); side = (rel[..., 0] * t_npc[..., 1] - rel[..., 1] * t_npc[..., 0]).abs()
        # a following distance the car can actually brake inside (4 m/s^2 + one car length)
        gap = (self.npc_v ** 2 / 8.0 + c.npc_gap).clamp_min(c.npc_gap)
        ego_ahead = (ahead > 0) & (ahead < gap) & (side < 2.5)
        relk = p_npc.unsqueeze(1) - p_npc.unsqueeze(2)                              # (B,K,K) other - this
        tk = t_npc.unsqueeze(2)
        ah = (relk * tk).sum(-1); sd = (relk[..., 0] * tk[..., 1] - relk[..., 1] * tk[..., 0]).abs()
        eye_k = torch.eye(K, dtype=torch.bool, device=dev).unsqueeze(0)
        npc_ahead = ((ah > 0) & (ah < gap.unsqueeze(-1)) & (sd < 2.5) & ~eye_k).any(-1)
        # Hold back near a corner while the ego car is close: turning into an
        # occupied lane is the other cars' job to avoid, not the fly's.
        _, dc_n = self.routes.next_corner(self.npc_route.reshape(-1), self.npc_s.reshape(-1)); dc_n = dc_n.view(B, K)
        ego_close = torch.linalg.norm(rel, dim=-1) < 14.0
        hold = ego_close & (dc_n < 12.0) & (dc_n > -2.0) & ~((ahead < -3.0))   # unless the ego is already behind
        # Inside an intersection the other cars give way to a fly that is in
        # front of them, whichever way it is pointing (the fly keeps crawling,
        # so this cannot deadlock).
        npc_in_inter = torch.cdist(p_npc.reshape(-1, 2), self.city.centres).min(dim=1).values.view(B, K) < c.road_width / 2 + 2.0
        give_way = npc_in_inter & (ahead > -1.0) & (ahead < 10.0) & (side < 6.0)
        v_t = torch.where(must_stop | ego_ahead | npc_ahead | hold | give_way, torch.zeros_like(self.npc_v), torch.full_like(self.npc_v, c.npc_speed))
        self.npc_v = torch.where(v_t > self.npc_v, torch.minimum(self.npc_v + 2.0 * cfg.dt, v_t), torch.maximum(self.npc_v - 4.0 * cfg.dt, v_t))
        self.npc_v = torch.where(must_stop & (ds < 0.8), torch.zeros_like(self.npc_v), self.npc_v)
        self.npc_s = self.npc_s + self.npc_v * cfg.dt
        done_npc = self.npc_s > self.routes.length[self.npc_route] - 5.0
        # collision with the ego car, in the ego frame
        fwd = torch.stack([torch.cos(self.heading), torch.sin(self.heading)], -1).unsqueeze(1)
        left = torch.stack([-fwd[..., 1], fwd[..., 0]], -1)
        r2 = p_npc - self.pos.unsqueeze(1)
        fx = (r2 * fwd).sum(-1); fy = (r2 * left).sum(-1)
        collision = ((fx.abs() < c.car_length) & (fy.abs() < c.car_width)).any(-1)
        if done_npc.any():
            self._respawn_npc(done_npc)

        # -- reward ----------------------------------------------------------------
        # Inside an intersection square with another car in it: the expert crawls.
        in_box = torch.cdist(self.pos, self.city.centres).min(dim=1).values < c.road_width / 2 + 2.0
        in_box_shared = (in_box & (torch.linalg.norm(rel, dim=-1) < 12.0).any(-1)).float()
        # Queueing behind a stopped car is waiting too, not stalling.
        in_lane_ahead = (fx > 0) & (fy.abs() < 3.2)
        queued = (in_lane_ahead & (fx < 12.0)).any(-1)
        gap_ahead, k_ahead = torch.where(in_lane_ahead, fx, torch.full_like(fx, 1e6)).min(-1)
        closeness = ((12.0 - gap_ahead) / 12.0).clamp(0.0, 1.0)                  # 0 beyond 12 m, 1 at contact
        v_ahead = torch.gather(self.npc_v, 1, k_ahead.unsqueeze(1)).squeeze(1)
        closing = (self.speed - v_ahead).clamp_min(0.0) * (gap_ahead < 12.0).float()
        self.stall = torch.where((self.speed < cfg.stall_speed) & ~waiting & ~queued, self.stall + 1, torch.zeros_like(self.stall))
        stalled = self.stall >= cfg.stall_patience
        reward = (cfg.w_progress * d_arc
                  - cfg.w_lateral * (lat / lane) ** 2
                  - cfg.w_heading * heading_err ** 2
                  - cfg.w_steer * self.steer ** 2
                  - cfg.w_jerk * (self.steer - prev_steer) ** 2 / cfg.dt
                  - 0.10 * oncoming.float()
                  - c.red_penalty * red_run.float()
                  - c.red_approach_w * self.speed * ((my_state < 0.5) & (d_stop > 0) & (d_stop < 15.0)).float()
                  + c.red_wait_bonus * ((my_state < 0.5) & (d_stop > 0) & (d_stop < 15.0) & (self.speed < 1.0)).float()
                  - c.oncoming_w * (lat - 0.8).clamp_min(0.0)
                  - c.follow_w * closeness * self.speed
                  - c.follow_rel_w * closeness * closing
                  - c.box_speed_w * in_box_shared * (self.speed - 3.0).clamp_min(0.0)
                  - cfg.w_heading * head_ahead ** 2
                  + c.corner_bonus * corner_passed.float()
                  + steer_term)
        crash = off_road | collision
        reward = torch.where(crash & ~wrong_turn, reward - cfg.crash_penalty, reward)
        reward = torch.where(wrong_turn, reward - c.wrong_turn_penalty, reward)
        reward = torch.where(stalled, reward - cfg.stall_penalty, reward)
        self.prev_arc = arc; self.lap_arc = self.lap_arc + d_arc
        self.t += 1
        timeout = self.t >= cfg.max_steps
        finished = arc > self.routes.length[self.route] - 12.0
        done = crash | stalled | timeout | finished
        self.ep_return += reward; self.ep_len += 1; self.ep_red += red_run.float()

        info = {"lateral": lat, "heading_err": heading_err, "off_road": off_road & ~wrong_turn, "wrong_turn": wrong_turn,
                "collision": collision,
                "red_run": red_run, "timeout": timeout, "stalled": stalled, "progress": d_arc,
                "corner_passed": corner_passed,
                "lap_arc": self.lap_arc.clone(), "speed": self.speed.clone(), "signal": my_state,
                "d_stop": d_stop, "cmd": self.proprio()[:, 3:]}
        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            self.stats["return"].append(self.ep_return[idx].clone()); self.stats["length"].append(self.ep_len[idx].clone())
            self.stats["distance"].append(self.lap_arc[idx].clone()); self.stats["crash"].append((off_road & ~wrong_turn)[idx].float())
            self.stats["wrong_turn"].append(wrong_turn[idx].float())
            self.stats["stall"].append(stalled[idx].float()); self.stats["collision"].append(collision[idx].float())
            self.stats["red"].append(self.ep_red[idx].clone())
            self.ep_return[idx] = 0; self.ep_len[idx] = 0; self.ep_red[idx] = 0
            self._respawn(idx)
        return self.observe(), self.proprio(), reward, done, info

    def pop_stats(self):
        if not self.stats["return"]:
            return None
        out = {k: torch.cat(v).float().mean().item() for k, v in self.stats.items()}
        out["n_episodes"] = int(sum(v.numel() for v in self.stats["return"]))
        for k in self.stats:
            self.stats[k] = []
        return out

    # -- drawing ---------------------------------------------------------------

    def draw_map(self, ax, k=0):
        """Roads, lamps, the car's route and the other cars, for environment ``k``."""
        import matplotlib.patches as mp
        c, city = self.ccfg, self.city
        hw = c.road_width / 2
        for a in getattr(ax, "_city_artists", []):
            a.remove()
        arts = []
        xs = city.xs.cpu().tolist(); ys = city.ys.cpu().tolist()
        for bx, by, _, hl, hwid, hh, lum in city.buildings.cpu().tolist():
            arts.append(ax.add_patch(mp.Rectangle((bx - hl, by - hwid), 2 * hl, 2 * hwid, color=(lum * 0.6, lum * 0.6, lum * 0.65), lw=0, zorder=0.5)))
        for y in ys:
            arts.append(ax.add_patch(mp.Rectangle((xs[0] - hw, y - hw), xs[-1] - xs[0] + 2 * hw, 2 * hw, color="#2b3240", lw=0, zorder=1)))
        for x in xs:
            arts.append(ax.add_patch(mp.Rectangle((x - hw, ys[0] - hw), 2 * hw, ys[-1] - ys[0] + 2 * hw, color="#2b3240", lw=0, zorder=1)))
        route = self.routes.centre[int(self.route[k])].cpu().numpy()
        arts.append(ax.plot(route[:, 0], route[:, 1], color="#58a6ff", lw=1.0, alpha=0.7, zorder=2)[0])
        st = self.city.signal_state(self.clock[k:k + 1])[0].cpu().numpy()
        lxy = city.lamp_xy.cpu().numpy()
        cols = ["#f85149" if s < 0.5 else ("#e3b341" if s < 1.5 else "#3fb950") for s in st]
        arts.append(ax.scatter(lxy[:, 0], lxy[:, 1], s=9, c=cols, zorder=4, linewidths=0))
        p, tn = self._npc_pose(); p = p[k].cpu().numpy(); tn = tn[k].cpu().numpy()
        for (px, py), (tx, ty) in zip(p, tn):
            ang = math.degrees(math.atan2(ty, tx))
            r = mp.Rectangle((-c.car_length / 2, -c.car_width / 2), c.car_length, c.car_width, color="#c9d1d9", zorder=5)
            import matplotlib.transforms as mt
            r.set_transform(mt.Affine2D().rotate_deg(ang).translate(px, py) + ax.transData)
            arts.append(ax.add_patch(r))
        ax._city_artists = arts
        e = city.extent
        ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])


class MultiCityEnv:
    """Several city layouts in one batch: cars 0..n-1 drive the first layout,
    n..2n-1 the second, and so on, so every PPO batch mixes layouts instead of
    swapping the single layout every few iterations (which unsettled the
    policy).  Presents the same interface as ``CityEnv`` to the trainer."""

    def __init__(self, cfg: EnvConfig, city_cfg=None, device="cpu", seeds=(0, 1001, 1002, 1003)):
        K = len(seeds); n = cfg.n_envs // K
        assert n * K == cfg.n_envs, "n_envs must be divisible by the number of cities"
        self.envs = [CityEnv(replace(cfg, n_envs=n), city_cfg=city_cfg, device=device, seed=int(s)) for s in seeds]
        e = self.envs[0]
        self.cfg = replace(e.cfg, n_envs=cfg.n_envs); self.ccfg = e.ccfg; self.eye = e.eye; self.device = e.device
        self.seeds = tuple(int(s) for s in seeds); self.n = n; self.B = n * K

    @property
    def obs_shape(self):
        return self.envs[0].obs_shape

    def reset_all(self):
        return torch.cat([e.reset_all() for e in self.envs], 0)

    def proprio(self):
        return torch.cat([e.proprio() for e in self.envs], 0)

    def step(self, action):
        outs = [e.step(action[i * self.n:(i + 1) * self.n]) for i, e in enumerate(self.envs)]
        obs, prop, reward, done = (torch.cat([o[j] for o in outs], 0) for j in range(4))
        info = {k: torch.cat([o[4][k] for o in outs], 0) for k in outs[0][4]}
        return obs, prop, reward, done, info

    def pop_stats(self):
        keys = self.envs[0].stats.keys()
        chunks = {k: [t for e in self.envs for t in e.stats[k]] for k in keys}
        for e in self.envs:
            for k in keys:
                e.stats[k] = []
        if not chunks["return"]:
            return None
        out = {k: torch.cat(v).float().mean().item() for k, v in chunks.items()}
        out["n_episodes"] = int(sum(t.numel() for t in chunks["return"]))
        return out

    def reseed(self, seed: int):
        raise NotImplementedError("MultiCityEnv holds fixed layouts; use CityEnv.reseed for the single-layout trainer")
