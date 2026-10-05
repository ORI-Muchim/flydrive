"""Central configuration for the fly-brain driving project.

Every magnitude here is in SI units (metres, seconds, radians) unless the name
says otherwise.  Angles that describe eye geometry are kept in degrees because
that is how the Drosophila literature reports them.
"""

from dataclasses import dataclass, field


@dataclass
class EyeConfig:
    """Geometry of one compound eye.

    Drosophila has ~750 ommatidia per eye with an interommatidial angle of
    ~5 deg and a Gaussian acceptance angle (half-width) of ~5.7 deg.  A 36x24
    lattice gives 864 ommatidia per eye, which lands in the right ballpark.
    """

    n_col: int = 36          # azimuth samples, front -> back
    n_row: int = 24          # elevation samples, ventral -> dorsal
    delta_phi_deg: float = 5.0    # interommatidial angle
    delta_rho_deg: float = 5.7    # acceptance angle (FWHM of the Gaussian)
    az_front_deg: float = -10.0   # most frontal column (negative = across midline)
    el_span_deg: float = 100.0    # total vertical extent of the lattice
    el_center_deg: float = -16.0  # tilt the field down: a driver looks at the road


@dataclass
class CarConfig:
    """Kinematic bicycle model."""

    wheelbase: float = 2.6
    eye_height: float = 1.2
    max_steer: float = 0.5        # rad at the front wheel
    max_accel: float = 4.0        # m/s^2 at full throttle
    max_brake: float = 5.0        # m/s^2 at full brake
    drag: float = 0.05            # linear drag coefficient
    v_max: float = 14.0
    v_min: float = 0.0
    v_init: float = 6.0


@dataclass
class TrackConfig:
    n_tracks: int = 8             # how many procedural tracks to pre-build
    n_points: int = 1024          # centreline resolution
    base_radius: float = 120.0
    wobble: float = 0.30          # how strongly the radius is modulated
    n_harmonics: int = 4
    road_width: float = 10.0
    post_spacing: float = 9.0     # arclength between edge posts
    dash_period: float = 7.0      # centre-line dash repeat (m)
    post_radius: float = 0.28
    post_height: float = 2.2
    tex_res: int = 640            # resolution of the cached road field


@dataclass
class RenderConfig:
    n_posts_visible: int = 24     # nearest posts composited per frame
    sky_luminance: float = 0.82
    ground_luminance: float = 0.20
    road_luminance: float = 0.58
    marking_luminance: float = 0.95
    post_luminance_a: float = 0.05
    post_luminance_b: float = 0.98
    texture_contrast: float = 0.30
    road_texture_scale: float = 0.25   # tarmac is smoother than the verge
    # Colours are only used for the human-readable camera view; the fly's own
    # pathway is achromatic, as R1-R6 effectively are.
    sky_rgb: tuple = (0.52, 0.70, 0.93)          # haze colour
    sky_zenith_rgb: tuple = (0.27, 0.50, 0.90)
    sky_horizon_rgb: tuple = (0.80, 0.88, 0.96)
    sun_dir: tuple = (-0.55, 0.75, 0.50)          # where the (cosmetic) sun sits: high, behind-left
    verge_rgb: tuple = (0.30, 0.47, 0.21)
    pavement_rgb: tuple = (0.66, 0.65, 0.61)     # a sidewalk band beside the kerb (colour only)
    kerb_rgb: tuple = (0.48, 0.48, 0.46)
    road_rgb: tuple = (0.25, 0.26, 0.29)
    marking_rgb: tuple = (0.95, 0.93, 0.80)
    post_dark_rgb: tuple = (0.11, 0.12, 0.14)
    post_light_rgb: tuple = (0.95, 0.95, 0.95)
    max_ground_distance: float = 400.0
    haze_distance: float = 130.0   # e-folding distance of the distance haze


@dataclass
class EnvConfig:
    n_envs: int = 256
    n_proprio: int = 3            # proprioceptive channels the brain receives
    dt: float = 0.05
    max_steps: int = 900
    off_road_margin: float = 0.6   # extra slack beyond the road edge
    w_progress: float = 1.0
    w_lateral: float = 0.30
    w_heading: float = 0.30
    w_steer: float = 0.02
    w_jerk: float = 0.06
    crash_penalty: float = 8.0
    stall_speed: float = 1.0      # below this the car counts as stalled
    stall_patience: int = 40      # steps of stalling before the episode ends
    stall_penalty: float = 8.0
    car: CarConfig = field(default_factory=CarConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)


@dataclass
class BrainConfig:
    n_hs: int = 3                 # HS cells per eye (HSN, HSE, HSS)
    n_vs: int = 10                # VS cells per eye (VS1..VS10)
    n_epg: int = 16               # ring-attractor wedges
    n_dn: int = 24                # descending neurons
    n_hidden: int = 96            # premotor pool feeding the DNs
    ring_gain: float = 6.0
    log_std_init: float = -0.9


@dataclass
class TrainConfig:
    total_steps: int = 6_000_000
    rollout: int = 64
    epochs: int = 2
    n_minibatch: int = 2          # minibatches over the *environment* axis
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.25
    ent_coef: float = 3e-3
    max_grad_norm: float = 5.0
    target_kl: float = 0.03
    critic_warmup: int = 0        # iterations that update only the value head (fine-tuning from imitation)
    bc_anchor: float = 0.0        # weight of ||mu - mu_reference||^2, holding a fine-tuned policy near its clone
    bc_anchor_final: float | None = None   # anchor weight at the end of training (linear schedule); None = constant
    bc_anchor_throttle_scale: float = 1.0  # anchor weight on the throttle channel relative to steering
    seed: int = 0
    anneal_lr: bool = True
    dash_every: int = 5           # iterations between dashboard refreshes
    snap_every: int = 15          # iterations between neural snapshots
    video_every: int = 25         # iterations between rollout videos
    ckpt_every: int = 40
