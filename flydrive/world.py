"""Procedural driving world and the analytic compound-eye renderer.

The world is deliberately simple so that a frame can be rendered in closed form
on the GPU for hundreds of environments at once -- no rasteriser, no OpenGL:

  * a textured ground plane at z = 0 (multi-octave sinusoidal noise, which
    stands in for the random-dot fields used in real fly flight simulators),
  * a brighter road band along a closed-loop centreline,
  * vertical posts along both road edges, the classic "pylon" landmark,
  * a uniform sky above the horizon.

Ray-plane intersection is analytic, so ground luminance is a texture lookup at
the hit point.  Posts are composited by comparing their angular extent against
each ommatidium's viewing direction.  Nothing here needs gradients.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import RenderConfig, TrackConfig
from .eye import CompoundEye

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Track construction
# ---------------------------------------------------------------------------


def _resample_by_arclength(pts: torch.Tensor, n_out: int) -> torch.Tensor:
    """Resample a closed polyline so the samples are equally spaced in arclength."""
    closed = torch.cat([pts, pts[:1]], dim=0)
    seg = torch.linalg.norm(closed[1:] - closed[:-1], dim=-1)
    cum = torch.cat([torch.zeros(1, dtype=seg.dtype, device=seg.device), torch.cumsum(seg, 0)])
    total = cum[-1]
    target = torch.linspace(0, float(total), n_out + 1, device=pts.device, dtype=pts.dtype)[:-1]
    idx = torch.searchsorted(cum, target.contiguous(), right=True).clamp(1, len(cum) - 1)
    lo, hi = idx - 1, idx
    t = ((target - cum[lo]) / (cum[hi] - cum[lo]).clamp_min(1e-9)).unsqueeze(-1)
    return closed[lo] * (1 - t) + closed[hi] * t


class TrackSet:
    """A batch of closed-loop tracks with cached road fields and edge posts."""

    def __init__(self, cfg: TrackConfig, device="cpu", seed: int = 0, dtype=torch.float32):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        g = torch.Generator(device="cpu").manual_seed(seed)

        N, T = cfg.n_points, cfg.n_tracks
        s = torch.linspace(0, TWO_PI, N + 1, dtype=dtype)[:-1]

        centres, tangents, normals, arcs, posts_xy, posts_col = [], [], [], [], [], []
        for _ in range(T):
            r = torch.ones(N, dtype=dtype) * cfg.base_radius
            for m in range(2, 2 + cfg.n_harmonics):
                amp = (cfg.wobble * cfg.base_radius
                       * (0.45 + 0.55 * torch.rand(1, generator=g, dtype=dtype))
                       / float(m) ** 0.7)
                pha = torch.rand(1, generator=g, dtype=dtype) * TWO_PI
                r = r + amp * torch.cos(m * s + pha)
            raw = torch.stack([r * torch.cos(s), r * torch.sin(s)], dim=-1)
            c = _resample_by_arclength(raw, N)

            nxt = torch.roll(c, -1, dims=0)
            prv = torch.roll(c, 1, dims=0)
            tan = nxt - prv
            tan = tan / torch.linalg.norm(tan, dim=-1, keepdim=True).clamp_min(1e-9)
            nrm = torch.stack([-tan[:, 1], tan[:, 0]], dim=-1)  # left-hand normal

            seg = torch.linalg.norm(nxt - c, dim=-1)
            arc = torch.cat([torch.zeros(1, dtype=dtype), torch.cumsum(seg, 0)[:-1]])

            half = cfg.road_width * 0.5
            step = max(1, int(round(cfg.post_spacing / float(seg.mean()))))
            sel = torch.arange(0, N, step)
            left_posts = c[sel] + nrm[sel] * half
            right_posts = c[sel] - nrm[sel] * half
            p_xy = torch.cat([left_posts, right_posts], dim=0)
            # Alternating light/dark posts give the motion detectors strong,
            # unambiguous contrast edges in both the ON and the OFF pathway.
            alt = torch.arange(sel.shape[0]) % 2
            p_col = torch.cat([alt, 1 - alt], dim=0).to(dtype)

            centres.append(c)
            tangents.append(tan)
            normals.append(nrm)
            arcs.append(arc)
            posts_xy.append(p_xy)
            posts_col.append(p_col)

        n_post = min(p.shape[0] for p in posts_xy)
        self.centre = torch.stack(centres).to(device)                 # (T, N, 2)
        self.tangent = torch.stack(tangents).to(device)               # (T, N, 2)
        self.normal = torch.stack(normals).to(device)                 # (T, N, 2)
        self.arc = torch.stack(arcs).to(device)                       # (T, N)
        self.length = self.arc[:, -1] + (self.arc[:, 1] - self.arc[:, 0])
        self.post_xy = torch.stack([p[:n_post] for p in posts_xy]).to(device)    # (T, P, 2)
        self.post_col = torch.stack([p[:n_post] for p in posts_col]).to(device)  # (T, P)
        self.n_tracks, self.n_points = T, N
        self.n_posts = n_post

        self._build_road_field()

    def _build_road_field(self):
        """Cache per-track fields so ray lookups never search the centreline.

        Three channels are stored on a regular grid: the signed distance to the
        centreline (positive to the left), and the cosine/sine of the arclength
        phase.  Storing the phase as a cos/sin pair keeps bilinear interpolation
        well-behaved across the seam where arclength wraps back to zero, and it
        is what drives the dashed centre line -- a strong streaming motion cue.
        """
        cfg = self.cfg
        R = cfg.tex_res
        pad = cfg.road_width * 3.0 + 12.0
        mins = self.centre.amin(dim=1) - pad      # (T, 2)
        maxs = self.centre.amax(dim=1) + pad
        self.tex_min, self.tex_max = mins, maxs

        fields = []
        for t in range(self.n_tracks):
            xs = torch.linspace(float(mins[t, 0]), float(maxs[t, 0]), R, device=self.device)
            ys = torch.linspace(float(mins[t, 1]), float(maxs[t, 1]), R, device=self.device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pts = torch.stack([gx, gy], dim=-1).view(-1, 2)           # (R*R, 2)

            c = self.centre[t]
            arc_phase = self.arc[t] * (TWO_PI / cfg.dash_period)
            best = torch.full((pts.shape[0],), float("inf"), device=self.device)
            lat = torch.zeros_like(best)
            aco = torch.zeros_like(best)
            asi = torch.zeros_like(best)
            rows = torch.arange(pts.shape[0], device=self.device)
            # Chunk over centreline points to bound peak memory.
            for lo in range(0, c.shape[0], 128):
                seg = c[lo:lo + 128]                                   # (S, 2)
                d = pts[:, None, :] - seg[None, :, :]                  # (R*R, S, 2)
                dist = torch.linalg.norm(d, dim=-1)
                m, am = dist.min(dim=1)
                upd = m < best
                nrm = self.normal[t, lo:lo + 128][am]                  # (R*R, 2)
                side = (d[rows, am] * nrm).sum(-1)
                ph = arc_phase[lo:lo + 128][am]
                lat = torch.where(upd, torch.sign(side) * m, lat)
                aco = torch.where(upd, torch.cos(ph), aco)
                asi = torch.where(upd, torch.sin(ph), asi)
                best = torch.where(upd, m, best)
            fields.append(torch.stack([lat, aco, asi]).view(3, R, R))
        self.road_field = torch.stack(fields)                           # (T, 3, R, R)
        self.lateral_field = self.road_field[:, 0]

    # -- sampling --------------------------------------------------------

    def sample_field(self, track_idx: torch.Tensor, xy: torch.Tensor):
        """Bilinear lookup of the cached road fields.

        Args:
            track_idx: ``(B,)`` long tensor.
            xy: ``(B, ..., 2)`` world positions.
        Returns:
            ``(lateral, arc_cos, arc_sin)``, each ``(B, ...)``.
        """
        B = track_idx.shape[0]
        lead = xy.shape[1:-1]
        flat = xy.reshape(B, -1, 2)
        mn = self.tex_min[track_idx].view(B, 1, 2)
        mx = self.tex_max[track_idx].view(B, 1, 2)
        uv = (flat - mn) / (mx - mn).clamp_min(1e-9) * 2.0 - 1.0
        field = self.road_field[track_idx]                              # (B, 3, R, R)
        out = F.grid_sample(
            field, uv.view(B, 1, -1, 2), mode="bilinear",
            padding_mode="border", align_corners=True,
        )                                                               # (B, 3, 1, M)
        out = out.view(B, 3, *lead) if lead else out.view(B, 3, -1)
        return out[:, 0], out[:, 1], out[:, 2]

    def sample_lateral(self, track_idx: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        return self.sample_field(track_idx, xy)[0]

    def nearest_centre(self, track_idx: torch.Tensor, xy: torch.Tensor):
        """Exact nearest-centreline query for the car itself (not for rays).

        Returns ``(index, signed_lateral, arclength, tangent)``.
        """
        c = self.centre[track_idx]                                      # (B, N, 2)
        d = xy.unsqueeze(1) - c
        dist = torch.linalg.norm(d, dim=-1)
        idx = dist.argmin(dim=1)
        b = torch.arange(xy.shape[0], device=xy.device)
        nrm = self.normal[track_idx][b, idx]
        tan = self.tangent[track_idx][b, idx]
        lat = (d[b, idx] * nrm).sum(-1)
        return idx, lat, self.arc[track_idx][b, idx], tan


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """Analytic, batched compound-eye renderer."""

    # Spatial wavelengths (m) of the ground texture octaves.
    OCTAVES = (2.3, 5.7, 13.0, 31.0)

    def __init__(self, eye: CompoundEye, tracks: TrackSet, cfg: RenderConfig,
                 eye_height: float, device="cpu", seed: int = 0):
        self.eye = eye
        self.tracks = tracks
        self.cfg = cfg
        self.eye_height = eye_height
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(seed + 991)
        n_oct = len(self.OCTAVES)
        k = torch.tensor([TWO_PI / w for w in self.OCTAVES])
        ang = torch.rand(n_oct, generator=g) * TWO_PI
        self.kx = (k * torch.cos(ang)).to(device)
        self.ky = (k * torch.sin(ang)).to(device)
        # A second, rotated set per octave breaks up the stripe look.
        ang2 = ang + 1.1 + torch.rand(n_oct, generator=g)
        self.kx2 = (k * torch.cos(ang2)).to(device)
        self.ky2 = (k * torch.sin(ang2)).to(device)
        self.phase = (torch.rand(n_oct, generator=g) * TWO_PI).to(device)
        self.phase2 = (torch.rand(n_oct, generator=g) * TWO_PI).to(device)
        self.amp = (1.0 / torch.arange(1, n_oct + 1, dtype=torch.float32) ** 0.6).to(device)
        self.amp = self.amp / self.amp.sum()
        self.k_mag = k.to(device)

    @torch.no_grad()
    def render(self, pos: torch.Tensor, heading: torch.Tensor, track_idx: torch.Tensor,
               rgb: bool = False, post_lum=None, boxes=None, spheres=None):
        """Render what the ray grid sees.

        Args:
            pos: ``(B, 2)`` world position of the eye.
            heading: ``(B,)`` yaw.
            track_idx: ``(B,)`` which track each env is on.
            rgb: also return a colour image, for the human-readable view.
        Returns:
            ``(image, depth)`` or ``(image, depth, colour)``.  ``image`` is
            ``(B, E, H, W)`` luminance in [0, 1]; ``colour`` adds a trailing
            RGB axis.
        """
        cfg = self.cfg
        B = pos.shape[0]
        H, W = self.eye.n_row, self.eye.n_col

        d = self.eye.rotated_dirs(heading)                    # (B, E, H, W, 3)
        dz = d[..., 2]
        down = dz < -1e-3

        # --- ground: analytic ray/plane intersection at z = 0 ---------------
        t_ground = torch.where(down, self.eye_height / (-dz).clamp_min(1e-3),
                               torch.full_like(dz, float("inf")))
        t_ground = torch.where(t_ground > cfg.max_ground_distance,
                               torch.full_like(t_ground, float("inf")), t_ground)
        hit = torch.isfinite(t_ground)
        t_safe = torch.where(hit, t_ground, torch.zeros_like(t_ground))
        px = pos[:, 0].view(B, 1, 1, 1) + t_safe * d[..., 0]
        py = pos[:, 1].view(B, 1, 1, 1) + t_safe * d[..., 1]

        # Footprint of one sample on the ground; used to pre-filter the texture
        # so distant ground does not alias into the motion detectors.
        foot = t_safe * self.eye.delta_rho / torch.sqrt((-dz).clamp_min(0.05))

        tex = torch.zeros_like(px)
        for i in range(len(self.OCTAVES)):
            atten = torch.exp(-0.5 * (self.k_mag[i] * foot) ** 2)
            tex = tex + self.amp[i] * atten * (
                torch.sin(self.kx[i] * px + self.ky[i] * py + self.phase[i])
                * torch.sin(self.kx2[i] * px + self.ky2[i] * py + self.phase2[i])
            )

        field = self.tracks.sample_field(track_idx, torch.stack([px, py], dim=-1))
        lat, aco, asi = field[0], field[1], field[2]
        mark = field[3] if len(field) > 3 else None       # a city has no markings inside junctions
        half = self.tracks.cfg.road_width * 0.5
        # Soft road edge, widened by the footprint so it stays anti-aliased.
        edge = (half - lat.abs()) / (foot + 0.35)
        on_road = torch.sigmoid(edge * 2.0)

        base = cfg.ground_luminance + (cfg.road_luminance - cfg.ground_luminance) * on_road
        # Tarmac is smoother than the verge, so the texture is a cue in itself.
        tex_gain = cfg.texture_contrast * (1.0 - (1.0 - cfg.road_texture_scale) * on_road)
        textured = base + tex_gain * tex

        # Solid edge lines and a dashed centre line.  The dashes stream past the
        # car and give the motion detectors an unambiguous longitudinal signal.
        line_w = 0.20 + 0.75 * foot
        edge_line = torch.exp(-0.5 * ((lat.abs() - (half - 0.55)) / line_w) ** 2)
        dash_on = torch.sigmoid(aco * 6.0)
        centre_line = torch.exp(-0.5 * (lat / line_w) ** 2) * dash_on
        paint = (edge_line + centre_line).clamp(0.0, 1.0) * on_road
        if mark is not None:
            paint = paint * mark

        albedo_lum = torch.lerp(base, torch.full_like(base, cfg.marking_luminance), paint)
        ground = torch.lerp(textured, torch.full_like(textured, cfg.marking_luminance), paint)
        ground = ground.clamp(0.02, 1.0)

        # Distance haze: the ground fades into the sky value, which removes the
        # hard cut at the far clipping distance and matches what a real horizon
        # looks like through a low-pass optical system.
        sky_grad = cfg.sky_luminance - 0.10 * (dz.clamp_min(0.0))
        haze = torch.exp(-t_safe / cfg.haze_distance)
        image = torch.where(hit, torch.lerp(torch.full_like(ground, cfg.sky_luminance),
                                            ground, haze), sky_grad)
        depth = torch.where(hit, t_ground, torch.full_like(t_ground, 1e6))

        colour = None
        if rgb:
            # The colour image is for people.  It shares the luminance path's
            # geometry, texture and markings (the shading term below), but adds
            # materials the fly never sees: a sidewalk with a kerb, a sky
            # gradient, a warm horizon.  Nothing here touches ``image``.
            dev, dt = image.device, image.dtype
            cc = lambda t: torch.tensor(t, device=dev, dtype=dt)
            absl = lat.abs()
            pavement = torch.sigmoid((half + 2.6 - absl) * 4.0) * torch.sigmoid((absl - half) * 4.0)
            kerb = torch.sigmoid((half + 0.35 - absl) * 12.0) * torch.sigmoid((absl - half) * 12.0)
            alb = torch.lerp(cc(cfg.verge_rgb), cc(cfg.pavement_rgb), pavement.unsqueeze(-1))
            alb = torch.lerp(alb, cc(cfg.kerb_rgb), kerb.unsqueeze(-1))
            alb = torch.lerp(alb, cc(cfg.road_rgb), on_road.unsqueeze(-1))
            alb = torch.lerp(alb, cc(cfg.marking_rgb), paint.unsqueeze(-1))
            shade = (ground / albedo_lum.clamp_min(1e-3)).clamp(0.35, 1.9)
            # the verge's texture is there for the motion detectors; in colour it
            # reads as blotches, so it is shown at a third of its contrast
            shade = torch.lerp(1.0 + 0.33 * (shade - 1.0), 1.0 + 0.6 * (shade - 1.0), on_road)
            lit = alb * shade.unsqueeze(-1)
            up = (dz.clamp(0.0, 1.0) * 2.2).clamp(0.0, 1.0).unsqueeze(-1)
            sky_col = torch.lerp(cc(cfg.sky_horizon_rgb), cc(cfg.sky_zenith_rgb), up)
            # a low sun behind and to the left of the driver's start heading: a
            # warm glow on the sky, purely cosmetic
            sun = cc(cfg.sun_dir); sun = sun / sun.norm()
            cosang = (d * sun).sum(-1).clamp(-1, 1)
            glow = torch.exp(-((1.0 - cosang) / 0.05) ** 2) * 0.22 + torch.exp(-((1.0 - cosang) / 0.006) ** 2) * 0.7
            sky_col = (sky_col + glow.unsqueeze(-1) * cc((1.0, 0.92, 0.70))).clamp(0, 1)
            # a thin cloud layer (cosmetic): three octaves of value noise on the
            # ray's intersection with a plane high above, thinning at the horizon
            u = 8.0 * d[..., 0] / (dz.clamp_min(0.0) + 0.25); v = 8.0 * d[..., 1] / (dz.clamp_min(0.0) + 0.25)
            n = (torch.sin(0.35 * u + 0.21 * v) * torch.sin(0.17 * u - 0.29 * v + 1.3)
                 + 0.5 * torch.sin(0.71 * u - 0.43 * v + 0.7) * torch.sin(0.53 * u + 0.62 * v + 2.1)
                 + 0.25 * torch.sin(1.37 * u + 0.91 * v + 3.0) * torch.sin(1.13 * u - 1.21 * v))
            cloud = torch.sigmoid((n - 0.35) * 5.0) * (dz.clamp(0.0, 0.5) * 2.0).sqrt() * (1.0 - glow.clamp(0, 1))
            sky_col = torch.lerp(sky_col, cc((0.97, 0.97, 0.99)), (0.8 * cloud).unsqueeze(-1))
            lit = torch.lerp(cc(cfg.sky_horizon_rgb), lit, haze.unsqueeze(-1))
            colour = torch.where(hit.unsqueeze(-1), lit, sky_col)

        image, depth, colour = self._composite_posts(image, depth, colour, pos, heading, track_idx, post_lum)
        if spheres is not None:
            image, depth, colour = self._composite_spheres(image, depth, colour, pos, d, spheres)
        if boxes is not None:
            image, depth, colour = self._composite_boxes(image, depth, colour, pos, d, boxes)
        image = image.clamp(0.0, 1.0)
        if rgb:
            return image, depth, colour.clamp(0.0, 1.0)
        return image, depth

    @torch.no_grad()
    def _composite_posts(self, image, depth, colour, pos, heading, track_idx, post_lum=None):
        cfg = self.cfg
        tcfg = self.tracks.cfg
        B = pos.shape[0]
        E, H, W = self.eye.az.shape
        K = min(cfg.n_posts_visible, self.tracks.n_posts)

        p_xy = self.tracks.post_xy[track_idx]                  # (B, P, 2)
        p_col = self.tracks.post_col[track_idx]                # (B, P)
        rel = p_xy - pos.unsqueeze(1)
        dist = torch.linalg.norm(rel, dim=-1)                  # (B, P)
        near_d, near_i = torch.topk(dist, K, dim=1, largest=False)
        b = torch.arange(B, device=pos.device).unsqueeze(1)
        rel = rel[b, near_i]                                   # (B, K, 2)
        col = p_col[b, near_i]                                 # (B, K)

        c, s = torch.cos(heading).view(B, 1), torch.sin(heading).view(B, 1)
        fwd = rel[..., 0] * c + rel[..., 1] * s                # (B, K) body +x
        lft = -rel[..., 0] * s + rel[..., 1] * c               # (B, K) body +y
        az_p = torch.atan2(lft, fwd)
        D = near_d.clamp_min(0.35)

        az = self.eye.az.view(1, 1, E, H, W)
        el = self.eye.el.view(1, 1, E, H, W)
        Dv = D.view(B, K, 1, 1, 1)
        az_p = az_p.view(B, K, 1, 1, 1)

        # Exact ray/cylinder intersection in the horizontal plane.  Using the
        # post's angular half-width and its centre distance instead makes a
        # nearby post render as a leaning slab, because the top and bottom
        # edges then ignore how the surface distance varies across its width.
        # Radius is inflated by half an acceptance angle so the silhouette stays
        # soft on the coarse ommatidial lattice.
        r_eff = tcfg.post_radius + Dv * self.eye.sigma_rho * 0.5
        daz = torch.remainder(az - az_p + math.pi, TWO_PI) - math.pi
        perp = Dv * torch.sin(daz)
        disc = r_eff ** 2 - perp ** 2
        along = Dv * torch.cos(daz)
        d_surf = along - torch.sqrt(disc.clamp_min(0.0))

        el_top = torch.atan2(torch.full_like(d_surf, tcfg.post_height - self.eye_height),
                             d_surf.clamp_min(0.05))
        el_bot = torch.atan2(torch.full_like(d_surf, -self.eye_height),
                             d_surf.clamp_min(0.05))
        cover = (disc >= 0) & (d_surf > 0.05) & (el >= el_bot) & (el <= el_top)

        big = torch.full_like(d_surf, 1e6)
        d_post = torch.where(cover, d_surf, big)
        best_d, best_k = d_post.min(dim=1)                     # (B, E, H, W)

        if post_lum is not None:
            lum = post_lum[b, near_i]                                       # (B, K) dynamic
        else:
            lum = torch.where(col > 0.5, cfg.post_luminance_b, cfg.post_luminance_a)  # (B, K)
        lum2d = lum
        lum = lum.view(B, K, 1, 1, 1).expand(-1, -1, E, H, W)
        post_lum = torch.gather(lum, 1, best_k.unsqueeze(1)).squeeze(1)

        visible = best_d < depth
        image = torch.where(visible, post_lum, image)
        if colour is not None:
            dev, dt = colour.device, colour.dtype
            dark = torch.tensor(cfg.post_dark_rgb, device=dev, dtype=dt)
            light = torch.tensor(cfg.post_light_rgb, device=dev, dtype=dt)
            if post_lum is not None:
                post_rgb = lum2d.unsqueeze(-1) * torch.ones(3, device=dev, dtype=dt)   # (B, K, 3)
            else:
                post_rgb = torch.where((col > 0.5).view(B, K, 1), light, dark)   # (B, K, 3)
            post_rgb = post_rgb.view(B, K, 1, 1, 1, 3).expand(-1, -1, E, H, W, 3)
            chosen = torch.gather(post_rgb, 1,
                                  best_k.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, -1, 3))
            colour = torch.where(visible.unsqueeze(-1), chosen.squeeze(1), colour)
        depth = torch.where(visible, best_d, depth)
        return image, depth, colour


    # -- dynamic scene objects ----------------------------------------------

    @torch.no_grad()
    def _composite_spheres(self, image, depth, colour, pos, d, spheres):
        """Signal lamps: ``spheres`` is (B, K, 5) = x, y, z, radius, luminance."""
        B = pos.shape[0]; K = spheres.shape[1]
        E, H, W = self.eye.az.shape
        eye = torch.stack([pos[:, 0], pos[:, 1], torch.full_like(pos[:, 0], self.eye_height)], -1)
        o = eye.view(B, 1, 1, 1, 1, 3) - spheres[:, :, :3].view(B, K, 1, 1, 1, 3)   # (B,K,1,1,1,3)
        w = d.unsqueeze(1)                                                            # (B,1,E,H,W,3)
        bq = (o * w).sum(-1)                                                          # (B,K,E,H,W)
        cq = (o * o).sum(-1) - spheres[:, :, 3].view(B, K, 1, 1, 1) ** 2
        disc = bq * bq - cq
        t = -bq - torch.sqrt(disc.clamp_min(0.0))
        hit = (disc >= 0) & (t > 0.05)
        t = torch.where(hit, t, torch.full_like(t, 1e6))
        best, k = t.min(dim=1)                                                        # (B,E,H,W)
        lum = torch.gather(spheres[:, :, 4].view(B, K, 1, 1, 1).expand(-1, -1, E, H, W), 1, k.unsqueeze(1)).squeeze(1)
        vis = best < depth
        image = torch.where(vis, lum, image)
        if colour is not None:
            # Signal state is luminance for the fly; for people it is the colour
            # a traffic lamp actually has.
            dev, dt = colour.device, colour.dtype
            red = torch.tensor((0.95, 0.16, 0.10), device=dev, dtype=dt)
            amber = torch.tensor((1.00, 0.72, 0.12), device=dev, dtype=dt)
            green = torch.tensor((0.20, 0.92, 0.38), device=dev, dtype=dt)
            l = lum.unsqueeze(-1)
            rgb = torch.where(l > 0.8, green, torch.where(l > 0.3, amber, red))
            colour = torch.where(vis.unsqueeze(-1), rgb, colour)
        depth = torch.where(vis, best, depth)
        return image, depth, colour

    @torch.no_grad()
    def _composite_boxes(self, image, depth, colour, pos, d, boxes):
        """Other cars: ``boxes`` is (B, K, 7) = x, y, yaw, half_len, half_wid, half_hgt, luminance.

        Exact ray / oriented-box intersection (slab test in the box frame); the
        face the ray enters through sets the shading so the box reads as solid.
        """
        B = pos.shape[0]; K = boxes.shape[1]
        E, H, W = self.eye.az.shape
        eye = torch.stack([pos[:, 0], pos[:, 1], torch.full_like(pos[:, 0], self.eye_height)], -1)
        c = torch.cat([boxes[:, :, :2], boxes[:, :, 5:6]], -1)                       # centre (B,K,3)
        yaw = boxes[:, :, 2]; cy, sy = torch.cos(yaw), torch.sin(yaw)
        half = boxes[:, :, 3:6]                                                        # (B,K,3)

        o = eye.view(B, 1, 3) - c                                                     # (B,K,3)
        ox = cy * o[..., 0] + sy * o[..., 1]; oy = -sy * o[..., 0] + cy * o[..., 1]; oz = o[..., 2]
        w = d.unsqueeze(1)                                                            # (B,1,E,H,W,3)
        cy5, sy5 = cy.view(B, K, 1, 1, 1), sy.view(B, K, 1, 1, 1)
        wx = cy5 * w[..., 0] + sy5 * w[..., 1]; wy = -sy5 * w[..., 0] + cy5 * w[..., 1]; wz = w[..., 2].expand(B, K, E, H, W)

        def slab(oo, ww, hh):
            ww = torch.where(ww.abs() < 1e-6, torch.full_like(ww, 1e-6), ww)
            t1 = (-hh - oo) / ww; t2 = (hh - oo) / ww
            return torch.minimum(t1, t2), torch.maximum(t1, t2)
        ex, xx = slab(ox.view(B, K, 1, 1, 1), wx, half[..., 0].view(B, K, 1, 1, 1))
        ey, xy_ = slab(oy.view(B, K, 1, 1, 1), wy, half[..., 1].view(B, K, 1, 1, 1))
        ez, xz = slab(oz.view(B, K, 1, 1, 1), wz, half[..., 2].view(B, K, 1, 1, 1))
        enter = torch.maximum(torch.maximum(ex, ey), ez); exit_ = torch.minimum(torch.minimum(xx, xy_), xz)
        hit = (enter <= exit_) & (enter > 0.05)
        face = torch.stack([ex, ey, ez], -1).argmax(-1)                                # which slab we enter
        shade = torch.where(face == 2, 1.18, torch.where(face == 0, 0.80, 1.0))
        t = torch.where(hit, enter, torch.full_like(enter, 1e6))
        best, k = t.min(dim=1)                                                        # (B,E,H,W)
        lum_k = boxes[:, :, 6].view(B, K, 1, 1, 1).expand(-1, -1, E, H, W)
        lum = torch.gather(lum_k * shade, 1, k.unsqueeze(1)).squeeze(1).clamp(0.02, 1.0)
        vis = best < depth
        image = torch.where(vis, lum, image)
        if colour is not None:
            colour = self._box_colour(colour, vis, best, k, face, boxes, eye, d, shade, lum)
            colour = self._contact_shadow(colour, depth, vis, boxes, eye, d)
        depth = torch.where(vis, best, depth)
        return image, depth, colour

    BUILDING_PALETTE = ((0.66, 0.40, 0.33), (0.82, 0.74, 0.58), (0.47, 0.52, 0.60),
                        (0.87, 0.83, 0.73), (0.44, 0.58, 0.58), (0.62, 0.60, 0.56))
    CAR_PALETTE = ((0.80, 0.18, 0.16), (0.92, 0.92, 0.90), (0.20, 0.36, 0.72),
                   (0.90, 0.72, 0.15), (0.70, 0.72, 0.76), (0.16, 0.17, 0.20))

    def _contact_shadow(self, colour, depth, vis, boxes, eye, d):
        """Darken the ground just around each box's footprint (colour only): the
        cheapest cue that a building or car stands on the road rather than
        floating over it."""
        B = colour.shape[0]; K = boxes.shape[1]
        ground = (~vis) & (depth < 1e5)
        p = eye.view(B, 1, 1, 1, 3) + depth.unsqueeze(-1) * d                     # ground hit (B,E,H,W,3)
        px, py = p[..., 0].unsqueeze(1), p[..., 1].unsqueeze(1)                    # (B,1,E,H,W)
        cx_, cy_ = boxes[:, :, 0].view(B, K, 1, 1, 1), boxes[:, :, 1].view(B, K, 1, 1, 1)
        yaw = boxes[:, :, 2].view(B, K, 1, 1, 1); cy, sy = torch.cos(yaw), torch.sin(yaw)
        ox, oy = px - cx_, py - cy_
        ux = (cy * ox + sy * oy).abs(); uy = (-sy * ox + cy * oy).abs()
        hl, hw = boxes[:, :, 3].view(B, K, 1, 1, 1), boxes[:, :, 4].view(B, K, 1, 1, 1)
        reach = 0.6 + 0.12 * boxes[:, :, 5].view(B, K, 1, 1, 1)                     # taller boxes cast a wider shade
        dx = (ux - hl).clamp_min(0.0); dy = (uy - hw).clamp_min(0.0)
        dist = torch.sqrt(dx * dx + dy * dy)
        shade = (1.0 - torch.exp(-dist / reach)).prod(dim=1)                         # (B,E,H,W): 0 at the wall, 1 far away
        dark = torch.lerp(torch.full_like(shade, 0.55), torch.ones_like(shade), shade)
        dark = torch.where(ground, dark, torch.ones_like(dark))
        return colour * dark.unsqueeze(-1)

    def _box_colour(self, colour, vis, best, k, face, boxes, eye, d, shade, lum):
        """Colour for the nearest box under each ray (colour image only).

        A box taller than a car is a building: a facade colour hashed from its
        position, a window grid on the walls, a dark roof.  A car gets a paint
        colour from its (fixed) luminance and a dark glass band.
        """
        B = colour.shape[0]; K = boxes.shape[1]
        E, H, W = self.eye.az.shape
        dev, dt = colour.device, colour.dtype
        kk = k.unsqueeze(1)
        g = lambda v: torch.gather(v.view(B, K, 1, 1, 1).expand(-1, -1, E, H, W), 1, kk).squeeze(1)
        cx_, cy_, yaw, hl, hw, hh = (g(boxes[:, :, i]) for i in range(6))
        face_k = torch.gather(face, 1, kk).squeeze(1)
        shade_k = torch.gather(shade, 1, kk).squeeze(1)
        p = eye.view(B, 1, 1, 1, 3) + best.unsqueeze(-1) * d                      # hit point (B,E,H,W,3)
        ox, oy, z = p[..., 0] - cx_, p[..., 1] - cy_, p[..., 2]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        ux = cy * ox + sy * oy; uy = -sy * ox + cy * oy
        tangent = torch.where(face_k == 0, uy, ux)
        building = hh > 2.0
        # facade colour: hashed from the footprint so it never flickers
        idx_b = (torch.floor(cx_ * 0.37 + cy_ * 0.61).long() % len(self.BUILDING_PALETTE))
        idx_c = (torch.floor(boxes[:, :, 6].clamp(0, 0.999) * len(self.CAR_PALETTE)).long())
        idx_c = torch.gather(idx_c.view(B, K, 1, 1, 1).expand(-1, -1, E, H, W), 1, kk).squeeze(1)
        pal_b = torch.tensor(self.BUILDING_PALETTE, device=dev, dtype=dt)[idx_b]
        pal_c = torch.tensor(self.CAR_PALETTE, device=dev, dtype=dt)[idx_c]
        wu = torch.frac(tangent / 3.2); wz = torch.frac((z - 1.0) / 3.0)
        window = (wu > 0.18) & (wu < 0.82) & (wz > 0.22) & (wz < 0.82) & (z > 1.0) & (z < 2 * hh - 0.9) & (face_k != 2)
        # some panes catch the sky, some are dark: a per-window hash keeps it fixed
        cell = torch.floor(tangent / 3.2) * 7.0 + torch.floor((z - 1.0) / 3.0) * 13.0 + cx_ * 0.11 + cy_ * 0.07
        hsh = torch.frac(torch.sin(cell) * 4375.85)
        glass = torch.where((hsh > 0.62).unsqueeze(-1), torch.tensor((0.64, 0.74, 0.84), device=dev, dtype=dt),
                            torch.tensor((0.30, 0.40, 0.52), device=dev, dtype=dt))
        roof = torch.tensor((0.30, 0.30, 0.33), device=dev, dtype=dt)
        wall = torch.where(window.unsqueeze(-1), glass, pal_b)
        # a storey ledge line, a darker plinth and a lighter parapet give the walls scale
        ledge = (wz < 0.06) & (z > 1.0) & (z < 2 * hh - 0.9) & (face_k != 2)
        wall = torch.where(ledge.unsqueeze(-1), wall * 0.78, wall)
        wall = wall * (0.80 + 0.20 * (z / 2.5).clamp(0.0, 1.0)).unsqueeze(-1)
        wall = torch.where(((z > 2 * hh - 0.5) & (face_k != 2)).unsqueeze(-1), wall * 1.10, wall)
        wall = torch.where((face_k == 2).unsqueeze(-1), roof, wall)
        car_glass = (z > 2 * hh * 0.58) & (face_k != 2)
        car = torch.where(car_glass.unsqueeze(-1), torch.tensor((0.16, 0.20, 0.26), device=dev, dtype=dt), pal_c)
        base = torch.where(building.unsqueeze(-1), wall, car)
        tint = (base * shade_k.clamp(0.6, 1.25).unsqueeze(-1)).clamp(0, 1)
        return torch.where(vis.unsqueeze(-1), tint, colour)
