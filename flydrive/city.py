"""A small city: a grid of two-way roads, signalled intersections, random
routes with turn commands, and other cars that follow their own routes.

Everything is analytic or precomputed so the existing renderer can draw it for
hundreds of environments at once: the road field is a closed-form function of
position on the grid, routes are arclength-parameterised polylines built from
lane offsets and corner fillets, signals are phase-shifted square waves, and
the other cars are points on routes rendered as oriented boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

TWO_PI = 2.0 * math.pi
LEFT, STRAIGHT, RIGHT = 0, 1, 2
DIRS = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])   # E N W S


@dataclass
class CityConfig:
    nx: int = 4                    # intersections along x
    ny: int = 4
    spacing: float = 100.0
    irregular: bool = False        # v3: block sizes drawn from [spacing_min, spacing_max]
    spacing_min: float = 60.0
    spacing_max: float = 140.0
    buildings: bool = False        # v3: box buildings inside every block
    buildings_per_block: int = 3
    building_setback: float = 3.0  # metres between the road edge and a building
    building_height: tuple = (6.0, 20.0)
    n_buildings_visible: int = 10
    road_width: float = 10.0       # two lanes
    lane_offset: float = 2.5       # right-lane centre from the road centreline
    fillet_right: float = 6.0      # corner radius of the lane path, right turn
    fillet_left: float = 11.0
    n_routes: int = 96
    route_turns: int = 6           # intersections per route (~650 m: finishable in one episode)
    n_points: int = 2048
    dash_period: float = 7.0
    # signals
    signal_period: float = 18.0
    green: float = 8.0
    yellow: float = 3.0            # a realistic amber; 1.5 s let cars enter on amber and cross on red
    stop_line: float = 4.0         # metres before the intersection square
    lamp_height: float = 3.2
    lamp_radius: float = 0.9
    post_radius: float = 0.35      # the lamp's pole, drawn by the post pass; lit like the lamp
    post_height: float = 3.2
    command_range: float = 40.0    # turn command fades in this far before a corner
    command_sharpness: float = 6.0 # metres over which it fades in
    wrong_turn_penalty: float = 8.0
    red_penalty: float = 5.0        # crossing the stop line on red
    red_approach_w: float = 0.0     # per-step penalty x speed while approaching a red within 15 m (dense shaping)
    red_wait_bonus: float = 0.0     # per-step reward for standing still at a red within 15 m (no escape incentive)
    oncoming_w: float = 0.0         # per-step penalty x metres the car sits left of its lane centre beyond 0.8 m (keep right through the box)
    follow_w: float = 0.0           # per-step penalty x speed x closeness to a car ahead within 12 m (keep a following distance)
    follow_rel_w: float = 0.0       # per-step penalty x closeness x closing speed toward the car ahead (approaching a stopped car is what hurts)
    box_speed_w: float = 0.0        # per-step penalty x (speed - 3 m/s)+ inside an intersection shared with another car
    lookahead: float = 0.0         # v2b: also penalise heading error to the route this far ahead (m)
    corner_bonus: float = 0.0      # v2b: reward for passing a corner on the commanded route
    v_max: float | None = None     # v2b: city speed cap (m/s), None = the car's own
    corner_spawn_frac: float = 0.0 # v2c: fraction of spawns placed 15-35 m before a corner
    steer_reward: float = 0.0      # v2c: per-step reward for steering the commanded way
    # other cars
    n_npc: int = 6
    npc_speed: float = 8.0
    npc_gap: float = 12.0
    car_length: float = 4.5
    car_width: float = 1.9
    car_height: float = 1.5


def _right(d):
    return torch.stack([d[..., 1], -d[..., 0]], dim=-1)


def _resample_open(pts: torch.Tensor, n_out: int):
    """Equal-arclength resampling of an open polyline. Returns pts, arclength."""
    seg = torch.linalg.norm(pts[1:] - pts[:-1], dim=-1)
    cum = torch.cat([torch.zeros(1, dtype=pts.dtype), torch.cumsum(seg, 0)])
    target = torch.linspace(0, float(cum[-1]), n_out, dtype=pts.dtype)
    idx = torch.searchsorted(cum, target.contiguous(), right=True).clamp(1, len(cum) - 1)
    lo, hi = idx - 1, idx
    t = ((target - cum[lo]) / (cum[hi] - cum[lo]).clamp_min(1e-9)).unsqueeze(-1)
    return pts[lo] * (1 - t) + pts[hi] * t, target


class CityMap:
    """The fixed geometry: roads, intersections, signal lamps."""

    def __init__(self, cfg: CityConfig, device="cpu", seed: int = 0):
        self.cfg = cfg
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(seed)
        S = cfg.spacing
        if cfg.irregular:
            gx = cfg.spacing_min + (cfg.spacing_max - cfg.spacing_min) * torch.rand(cfg.nx - 1, generator=g)
            gy = cfg.spacing_min + (cfg.spacing_max - cfg.spacing_min) * torch.rand(cfg.ny - 1, generator=g)
        else:
            gx = torch.full((cfg.nx - 1,), S); gy = torch.full((cfg.ny - 1,), S)
        xs = torch.cat([torch.zeros(1), torch.cumsum(gx, 0)]); xs = xs - xs.mean()
        ys = torch.cat([torch.zeros(1), torch.cumsum(gy, 0)]); ys = ys - ys.mean()
        self.xs, self.ys = xs.to(device), ys.to(device)        # road centrelines along each axis
        self.x0, self.y0 = float(xs[0]), float(ys[0])
        # intersection centres, index k = i * ny + j
        ii, jj = torch.meshgrid(torch.arange(cfg.nx), torch.arange(cfg.ny), indexing="ij")
        self.centres = torch.stack([xs[ii.reshape(-1)], ys[jj.reshape(-1)]], 1).to(device)
        self._build_buildings(g)
        self.n_inter = cfg.nx * cfg.ny
        self.phase = (torch.rand(self.n_inter, generator=g) * cfg.signal_period).to(device)

        # one lamp per (intersection, approach direction): at the near-right
        # corner of the approach, just before the stop line
        hw = cfg.road_width / 2
        lamps = []
        for k in range(self.n_inter):
            for a in range(4):
                d = DIRS[a]
                p = self.centres[k].cpu() + _right(d) * (hw + 1.2) - d * (hw + cfg.stop_line)
                lamps.append(p)
        self.lamp_xy = torch.stack(lamps).to(device)                         # (n_inter*4, 2)
        self.lamp_inter = torch.arange(self.n_inter).repeat_interleave(4).to(device)
        self.lamp_dir = torch.arange(4).repeat(self.n_inter).to(device)
        # lamp posts are ordinary posts for the renderer (dark), the lamp itself is a sphere
        self.post_xy = self.lamp_xy.unsqueeze(0)                              # (1, P, 2)
        self.post_col = torch.full((1, self.lamp_xy.shape[0]), 0.35, device=device)
        self.n_posts = self.lamp_xy.shape[0]
        self.n_tracks = 1
        self.extent = (float(xs[0]) - S / 2, float(xs[-1]) + S / 2, float(ys[0]) - S / 2, float(ys[-1]) + S / 2)

    def _build_buildings(self, g):
        """Axis-aligned box buildings inside each block, set back from the roads."""
        cfg = self.cfg
        hw = cfg.road_width / 2 + cfg.building_setback
        rows = []
        if cfg.buildings:
            xs, ys = self.xs.cpu(), self.ys.cpu()
            for i in range(cfg.nx - 1):
                for j in range(cfg.ny - 1):
                    bx0, bx1 = float(xs[i]) + hw, float(xs[i + 1]) - hw
                    by0, by1 = float(ys[j]) + hw, float(ys[j + 1]) - hw
                    if bx1 - bx0 < 12 or by1 - by0 < 12:
                        continue
                    for _ in range(cfg.buildings_per_block):
                        hl = 5.0 + 9.0 * float(torch.rand(1, generator=g)); hwid = 5.0 + 9.0 * float(torch.rand(1, generator=g))
                        hl = min(hl, (bx1 - bx0) / 2 - 0.5); hwid = min(hwid, (by1 - by0) / 2 - 0.5)
                        cx = bx0 + hl + (bx1 - bx0 - 2 * hl) * float(torch.rand(1, generator=g))
                        cy = by0 + hwid + (by1 - by0 - 2 * hwid) * float(torch.rand(1, generator=g))
                        h = cfg.building_height[0] + (cfg.building_height[1] - cfg.building_height[0]) * float(torch.rand(1, generator=g))
                        lum = 0.25 + 0.5 * float(torch.rand(1, generator=g))
                        rows.append([cx, cy, 0.0, hl, hwid, h / 2, lum])
        self.buildings = torch.tensor(rows, device=self.device) if rows else torch.zeros(0, 7, device=self.device)

    def nearest_buildings(self, pos, k):
        """The ``k`` buildings closest to each position, as renderer boxes (B, k, 7)."""
        if self.buildings.shape[0] == 0:
            return None
        k = min(k, self.buildings.shape[0])
        d = torch.linalg.norm(self.buildings[:, :2].unsqueeze(0) - pos.unsqueeze(1), dim=-1)
        _, idx = torch.topk(d, k, dim=1, largest=False)
        return self.buildings[idx]

    # -- signal state -----------------------------------------------------

    def signal_state(self, t: torch.Tensor):
        """State of every lamp at times ``t`` (B,): 0 red, 1 yellow, 2 green -> (B, P)."""
        cfg = self.cfg
        tau = torch.remainder(t.view(-1, 1) + self.phase[self.lamp_inter].view(1, -1), cfg.signal_period)
        ns = (self.lamp_dir % 2 == 1).view(1, -1)               # N/S approaches share a phase
        half = cfg.signal_period / 2
        tau = torch.where(ns, torch.remainder(tau + half, cfg.signal_period), tau)
        state = torch.full_like(tau, 0.0)
        state = torch.where(tau < cfg.green, torch.full_like(tau, 2.0), state)
        state = torch.where((tau >= cfg.green) & (tau < cfg.green + cfg.yellow), torch.full_like(tau, 1.0), state)
        return state

    @staticmethod
    def lamp_luminance(state):
        return torch.where(state > 1.5, 0.98, torch.where(state > 0.5, 0.55, 0.06))

    # -- road field, closed form ------------------------------------------

    def sample_field(self, track_idx, xy):
        """(distance to nearest road centreline, cos/sin of the dash phase, marking mask)."""
        cfg = self.cfg
        S, hw = cfg.spacing, cfg.road_width / 2
        px, py = xy[..., 0], xy[..., 1]
        # nearest vertical / horizontal road centreline (roads need not be evenly spaced)
        xs, ys = self.xs, self.ys
        ix = torch.searchsorted(xs, px.reshape(-1).contiguous()).clamp(1, len(xs) - 1).view(px.shape)
        near_x = torch.where((px - xs[ix - 1]).abs() < (xs[ix] - px).abs(), xs[ix - 1], xs[ix])
        iy = torch.searchsorted(ys, py.reshape(-1).contiguous()).clamp(1, len(ys) - 1).view(py.shape)
        near_y = torch.where((py - ys[iy - 1]).abs() < (ys[iy] - py).abs(), ys[iy - 1], ys[iy])
        dxv = px - near_x                          # signed distance to the nearest vertical road
        dyh = py - near_y
        on_v = dxv.abs() < hw
        on_h = dyh.abs() < hw
        # Roads only span the grid; beyond the last intersection there is verge.
        in_x = (px > xs[0] - hw) & (px < xs[-1] + hw)
        in_y = (py > ys[0] - hw) & (py < ys[-1] + hw)
        on_v = on_v & in_y
        on_h = on_h & in_x
        lat = torch.where(on_v & ~on_h, dxv, torch.where(on_h, dyh, torch.minimum(dxv.abs(), dyh.abs())))
        # dash phase runs along the road the point is on
        along = torch.where(on_v & ~on_h, py, px)
        ph = along * (TWO_PI / cfg.dash_period)
        mark = ((on_v ^ on_h)).float()             # no markings inside the intersection square
        return lat, torch.cos(ph), torch.sin(ph), mark


class RouteSet:
    """Random lane-following routes through the grid, stored like tracks."""

    def __init__(self, city: CityMap, device="cpu", seed: int = 0):
        cfg = city.cfg
        self.cfg = cfg
        self.city = city
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(seed + 7)
        S, o = cfg.spacing, cfg.lane_offset
        centres, tangents, normals, arcs = [], [], [], []
        corner_arc, corner_turn = [], []        # per route: arclength of each corner, its turn type
        inter_arc, inter_lamp = [], []          # per route: arclength of each stop line, its lamp index
        for r in range(cfg.n_routes):
            # random walk on the intersection grid without U-turns, staying inside
            i = int(torch.randint(0, cfg.nx, (1,), generator=g)); j = int(torch.randint(0, cfg.ny, (1,), generator=g))
            a = int(torch.randint(0, 4, (1,), generator=g))
            nodes, dirs, turns = [(i, j)], [], []
            for _ in range(cfg.route_turns + 1):
                # candidate turns: left, straight, right
                opts = []
                for turn, da in ((LEFT, 1), (STRAIGHT, 0), (RIGHT, -1)):
                    na = (a + da) % 4
                    ni, nj = i + int(DIRS[na, 0]), j + int(DIRS[na, 1])
                    if 0 <= ni < cfg.nx and 0 <= nj < cfg.ny:
                        opts.append((turn, na, ni, nj))
                if not opts:
                    break
                turn, na, ni, nj = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                if dirs:
                    turns.append(turn)
                dirs.append(na); i, j, a = ni, nj, na
                nodes.append((i, j))
            # lane path: straights offset to the right lane, corners filleted
            pts = []
            cum_corner, ctype = [], []
            C = [city.centres[n[0] * cfg.ny + n[1]].cpu() for n in nodes]
            d = [DIRS[k] for k in dirs]
            start = C[0] + _right(d[0]) * o
            pts.append(start)
            for k in range(1, len(C) - 1):
                d1, d2 = d[k - 1], d[k]
                P = C[k] + _right(d1) * o + _right(d2) * o if turns[k - 1] != STRAIGHT else C[k] + _right(d1) * o
                if turns[k - 1] == STRAIGHT:
                    pts.append(P); continue
                R = cfg.fillet_right if turns[k - 1] == RIGHT else cfg.fillet_left
                T1, T2 = P - d1 * R, P + d2 * R
                ctr = P - d1 * R + d2 * R
                a1 = math.atan2(float(T1[1] - ctr[1]), float(T1[0] - ctr[0]))
                a2 = math.atan2(float(T2[1] - ctr[1]), float(T2[0] - ctr[0]))
                da = (a2 - a1 + math.pi) % TWO_PI - math.pi
                n_arc = max(6, int(abs(da) * R / 1.0))
                pts.append(T1)
                ctype.append(turns[k - 1]); cum_corner.append(len(pts) - 1)
                for q in range(1, n_arc):
                    ang = a1 + da * q / n_arc
                    pts.append(ctr + torch.tensor([math.cos(ang), math.sin(ang)]) * R)
                pts.append(T2)
            end = C[-1] + _right(d[-1]) * o
            pts.append(end)
            poly = torch.stack(pts)
            c, arc = _resample_open(poly, cfg.n_points)
            # stop line of every intersection the route passes (corner or straight)
            hw = cfg.road_width / 2
            st_arc, st_id, st_dir = [], [], []
            for k in range(1, len(C) - 1):
                d1 = d[k - 1]
                Q = C[k] + _right(d1) * o - d1 * (hw + cfg.stop_line)
                q = int(torch.linalg.norm(c - Q, dim=-1).argmin())
                st_arc.append(float(arc[q])); st_id.append(nodes[k][0] * cfg.ny + nodes[k][1]); st_dir.append(dirs[k - 1])
            inter_arc.append(torch.tensor(st_arc + [1e9])); inter_lamp.append(torch.tensor([i_ * 4 + a_ for i_, a_ in zip(st_id, st_dir)] + [0]))
            tan = torch.zeros_like(c); tan[1:-1] = c[2:] - c[:-2]; tan[0] = c[1] - c[0]; tan[-1] = c[-1] - c[-2]
            tan = tan / torch.linalg.norm(tan, dim=-1, keepdim=True).clamp_min(1e-9)
            nrm = torch.stack([-tan[:, 1], tan[:, 0]], -1)
            # arclength of each corner's entry point on the resampled path
            seg = torch.linalg.norm(poly[1:] - poly[:-1], dim=-1)
            cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
            corner_arc.append(torch.tensor([float(cum[q]) for q in cum_corner] + [1e9]))
            corner_turn.append(torch.tensor(ctype + [STRAIGHT]))
            centres.append(c); tangents.append(tan); normals.append(nrm); arcs.append(arc)
        self.centre = torch.stack(centres).to(device)
        self.tangent = torch.stack(tangents).to(device)
        self.normal = torch.stack(normals).to(device)
        self.arc = torch.stack(arcs).to(device)
        self.length = self.arc[:, -1]
        n_c = max(len(x) for x in corner_arc)
        self.corner_arc = torch.stack([torch.cat([x, torch.full((n_c - len(x),), 1e9)]) for x in corner_arc]).to(device)
        self.corner_turn = torch.stack([torch.cat([x, torch.full((n_c - len(x),), STRAIGHT)]) for x in corner_turn]).to(device)
        n_i = max(len(x) for x in inter_arc)
        self.inter_arc = torch.stack([torch.cat([x, torch.full((n_i - len(x),), 1e9)]) for x in inter_arc]).to(device)
        self.inter_lamp = torch.stack([torch.cat([x, torch.zeros(n_i - len(x), dtype=torch.long)]) for x in inter_lamp]).to(device)
        self.n_tracks = cfg.n_routes
        self.n_points = cfg.n_points

    def nearest_centre(self, route_idx, xy, near=None, window=25.0):
        """Nearest route point to ``xy``: lateral offset, arclength, tangent.

        A route with six turns on a small grid often revisits a street, so the
        globally nearest point can belong to a later visit and the arclength
        flips back and forth by a hundred metres from one step to the next.
        ``near`` (the previous arclength) restricts the search to the stretch
        just behind and ahead of it."""
        c = self.centre[route_idx]
        d = xy.unsqueeze(1) - c
        dist = torch.linalg.norm(d, dim=-1)
        if near is not None:
            a = self.arc[route_idx]
            ok = (a >= near.unsqueeze(1) - 8.0) & (a <= near.unsqueeze(1) + window)
            local = torch.where(ok, dist, torch.full_like(dist, float("inf")))
            dist = torch.where(ok.any(dim=1, keepdim=True), local, dist)
        idx = dist.argmin(dim=1)
        b = torch.arange(xy.shape[0], device=xy.device)
        nrm = self.normal[route_idx][b, idx]; tan = self.tangent[route_idx][b, idx]
        return idx, (d[b, idx] * nrm).sum(-1), self.arc[route_idx][b, idx], tan

    def next_corner(self, route_idx, arc, lookback=25.0):
        """Turn type and distance to the current/next corner from arclength ``arc``.

        A corner stays "current" for ``lookback`` metres past its entry point,
        i.e. through the whole fillet arc (14 m right, 22 m left), so the turn
        command a policy receives does not vanish the moment the turn begins."""
        ca = self.corner_arc[route_idx]                                  # (B, n_c)
        ahead = torch.where(ca > arc.unsqueeze(1) - lookback, ca - arc.unsqueeze(1), torch.full_like(ca, 1e9))
        dist, k = ahead.min(dim=1)
        turn = self.corner_turn[route_idx][torch.arange(len(k), device=k.device), k]
        return turn, dist

    def next_stop(self, route_idx, arc, lookback=1.0):
        """Distance to the next stop line ahead of ``arc`` and that lamp's index."""
        ia = self.inter_arc[route_idx]
        ahead = torch.where(ia > arc.unsqueeze(1) - lookback, ia - arc.unsqueeze(1), torch.full_like(ia, 1e9))
        dist, k = ahead.min(dim=1)
        lamp = self.inter_lamp[route_idx][torch.arange(len(k), device=k.device), k]
        return dist, lamp

    def point_at(self, route_idx, s):
        """Position and tangent at arclength ``s`` (B,) along each route."""
        L = self.length[route_idx]
        u = (s / L).clamp(0, 1) * (self.cfg.n_points - 1)
        i0 = u.floor().long().clamp(0, self.cfg.n_points - 2); f = (u - i0).unsqueeze(-1)
        b = torch.arange(len(s), device=s.device)
        c = self.centre[route_idx]; t = self.tangent[route_idx]
        return c[b, i0] * (1 - f) + c[b, i0 + 1] * f, t[b, i0]
