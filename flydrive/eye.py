"""Compound-eye geometry.

The eye is a hexagonal lattice of ommatidia on a sphere, but every downstream
stage keeps it as a dense ``(row, col)`` grid so the whole optic lobe can be
written with plain 2-D tensor shifts and convolutions:

    col (W) -> azimuth, increasing from front to back
    row (H) -> elevation, increasing upwards

The hexagonal packing survives as a half-column offset on odd rows, which is
what the ray directions and the plotting code actually use.

Both eyes share the index convention: the right eye is the mirror image of the
left, so ``col + 1`` always means "one step in the front-to-back (regressive)
direction".  A T4/T5 subtype therefore has the same lattice offset in both
eyes, which is exactly how the real optic lobes are organised.
"""

from __future__ import annotations

import math

import torch

from .config import EyeConfig

DEG = math.pi / 180.0
LEFT, RIGHT = 0, 1


class RayGrid:
    """Common interface the renderer needs: per-sample ray directions in the
    body frame, their azimuth/elevation, and an angular sample spacing.

    Both the compound eye and the rectilinear camera below satisfy it, so the
    same renderer draws the fly's view and the human view of the same instant.
    """

    dirs: torch.Tensor      # (E, H, W, 3), body frame: +x forward, +y left, +z up
    az: torch.Tensor        # (E, H, W)
    el: torch.Tensor        # (E, H, W)
    delta_rho: float        # angular width of one sample, radians
    sigma_rho: float

    def to(self, device):
        self.device = torch.device(device)
        self.az = self.az.to(device)
        self.el = self.el.to(device)
        self.dirs = self.dirs.to(device)
        return self

    def rotated_dirs(self, heading: torch.Tensor) -> torch.Tensor:
        """Rotate the body-frame rays into world coordinates.

        Args:
            heading: ``(B,)`` yaw angles, world frame, counter-clockwise.
        Returns:
            ``(B, E, H, W, 3)`` unit vectors.
        """
        c = torch.cos(heading).view(-1, 1, 1, 1)
        s = torch.sin(heading).view(-1, 1, 1, 1)
        dx, dy, dz = self.dirs[..., 0], self.dirs[..., 1], self.dirs[..., 2]
        dx = dx.unsqueeze(0)
        dy = dy.unsqueeze(0)
        dz = dz.unsqueeze(0).expand(c.shape[0], -1, -1, -1)
        return torch.stack([dx * c - dy * s, dx * s + dy * c, dz], dim=-1)


class CompoundEye(RayGrid):
    """Ray directions and lattice bookkeeping for a pair of compound eyes."""

    def __init__(self, cfg: EyeConfig | None = None, device="cpu", dtype=torch.float32):
        self.cfg = cfg or EyeConfig()
        self.device = torch.device(device)
        self.dtype = dtype

        H, W = self.cfg.n_row, self.cfg.n_col
        self.n_row, self.n_col = H, W
        self.n_omma = H * W

        row = torch.arange(H, dtype=dtype, device=device).view(H, 1).expand(H, W)
        col = torch.arange(W, dtype=dtype, device=device).view(1, W).expand(H, W)

        # Hexagonal packing: odd rows are offset by half an interommatidial angle
        # and rows are spaced by dphi * sin(60 deg).
        dphi = self.cfg.delta_phi_deg
        hex_shift = 0.5 * (row % 2)
        az_left = (self.cfg.az_front_deg + (col + hex_shift) * dphi) * DEG

        row_span = self.cfg.el_span_deg / max(H - 1, 1)
        el = (self.cfg.el_center_deg + (row - (H - 1) / 2) * row_span) * DEG

        # Right eye mirrors the left across the sagittal plane.
        az = torch.stack([az_left, -az_left], dim=0)          # (2, H, W)
        el = torch.stack([el, el], dim=0)                      # (2, H, W)

        self.az = az
        self.el = el

        # Body frame: +x forward, +y left, +z up.  Azimuth is positive to the left.
        cos_el = torch.cos(el)
        self.dirs = torch.stack(
            [cos_el * torch.cos(az), cos_el * torch.sin(az), torch.sin(el)], dim=-1
        )                                                      # (2, H, W, 3)

        self.delta_rho = self.cfg.delta_rho_deg * DEG
        # Convert the FWHM-style acceptance angle into a Gaussian sigma.
        self.sigma_rho = self.delta_rho / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    # -- helpers ---------------------------------------------------------

    def hex_xy(self, eye: int = LEFT):
        """Plot coordinates (degrees) for drawing the lattice as hexagons."""
        return self.az[eye] / DEG, self.el[eye] / DEG

    def __repr__(self):
        return (
            f"CompoundEye({self.n_row}x{self.n_col} = {self.n_omma} ommatidia/eye, "
            f"{2 * self.n_omma} total, dphi={self.cfg.delta_phi_deg}deg, "
            f"drho={self.cfg.delta_rho_deg}deg)"
        )


class PinholeCamera(RayGrid):
    """An ordinary rectilinear camera looking out of the same eye position.

    This exists purely so a human can see what the car is doing.  It samples the
    identical world with the identical renderer -- only the ray pattern differs:
    a perspective grid instead of a hexagonal lattice, at a far finer angular
    spacing and with no Gaussian acceptance blur.
    """

    def __init__(self, width=640, height=360, fov_deg=92.0, pitch_deg=-4.0,
                 device="cpu", dtype=torch.float32):
        self.n_col, self.n_row = width, height
        self.n_samples = width * height
        self.fov_deg = fov_deg

        half_x = math.tan(fov_deg * DEG / 2)
        half_y = half_x * height / width
        xs = torch.linspace(-half_x, half_x, width, device=device, dtype=dtype)
        ys = torch.linspace(half_y, -half_y, height, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        # Screen +x is to the right, which is -y in the body frame; row 0 is the
        # top of the image, so ys runs downwards.
        d = torch.stack([torch.ones_like(gx), -gx, gy], dim=-1)
        if pitch_deg:
            p = pitch_deg * DEG
            cp, sp = math.cos(p), math.sin(p)
            dx, dy, dz = d.unbind(-1)
            d = torch.stack([dx * cp - dz * sp, dy, dx * sp + dz * cp], dim=-1)
        d = d / d.norm(dim=-1, keepdim=True)

        self.dirs = d.unsqueeze(0)                                    # (1, H, W, 3)
        self.az = torch.atan2(d[..., 1], d[..., 0]).unsqueeze(0)
        self.el = torch.asin(d[..., 2].clamp(-1, 1)).unsqueeze(0)
        self.device = torch.device(device)
        self.delta_rho = fov_deg * DEG / width
        self.sigma_rho = self.delta_rho / 2.355

    def __repr__(self):
        return f"PinholeCamera({self.n_col}x{self.n_row}, fov {self.fov_deg:.0f} deg)"
