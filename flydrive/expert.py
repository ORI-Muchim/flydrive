"""A scripted reference driver for the city.

Not a policy: it reads the route, the signals and the other cars directly, and
exists to answer two questions -- is the environment solvable to a clean drive
at all, and what does "perfect" score in it -- and to serve as a teacher for a
warm start if exploration alone cannot discover turning.
"""

from __future__ import annotations

import math

import torch

from .city import LEFT, RIGHT, STRAIGHT


@torch.no_grad()
def expert_action(env, lookahead=7.0, v_cruise=9.0, v_corner=4.5, brake_decel=3.0, crawl_shared=True):
    """(B, 2) steering / throttle actions in [-1, 1] for every environment."""
    cfg, car, c = env.cfg, env.cfg.car, env.ccfg
    B, dev = env.B, env.device
    _, lat, arc, tan = env.routes.nearest_centre(env.route, env.pos, near=env.prev_arc)

    # --- pure pursuit on the lane path ---------------------------------------
    target, _ = env.routes.point_at(env.route, arc + lookahead)
    to_t = target - env.pos
    alpha = torch.atan2(to_t[:, 1], to_t[:, 0]) - env.heading
    alpha = torch.remainder(alpha + math.pi, 2 * math.pi) - math.pi
    Ld = torch.linalg.norm(to_t, dim=-1).clamp_min(1.0)
    steer = torch.atan(2.0 * car.wheelbase * torch.sin(alpha) / Ld)
    a_steer = (steer / car.max_steer).clamp(-1, 1)

    # --- speed target: corners, signals, cars ahead -------------------------------
    turn, d_corner = env.routes.next_corner(env.route, arc)
    v_t = torch.full((B,), v_cruise, device=dev)
    v_t = torch.where((turn != STRAIGHT) & (d_corner < 25.0), torch.full_like(v_t, v_corner), v_t)
    v_t = torch.where((turn != STRAIGHT) & (d_corner < 0.0) & (d_corner > -20.0), torch.full_like(v_t, v_corner), v_t)

    state = env.city.signal_state(env.clock)
    d_stop, lamp = env.routes.next_stop(env.route, arc)
    my = state[torch.arange(B, device=dev), lamp]
    brake_dist = env.speed ** 2 / (2 * brake_decel) + 3.0
    # hold the stop until well past the line, so a small overshoot never releases it
    must_stop = (my < 1.5) & (d_stop < brake_dist) & (d_stop > -3.0)
    # an amber is driven through only when the car physically cannot stop for it
    cannot_stop = d_stop < env.speed ** 2 / (2 * 5.0) + 0.5
    must_stop = must_stop & ~((my > 0.5) & cannot_stop & (d_stop > 0))
    # A braking curve the car can actually follow (3 m/s^2, under its 5 m/s^2
    # limit), coming to rest 2.5 m short of the line.
    v_creep = torch.sqrt(2.0 * 3.0 * (d_stop - 2.5).clamp_min(0.0))
    v_t = torch.where(must_stop, torch.minimum(v_t, v_creep), v_t)

    p_npc, _ = env._npc_pose()
    fwd = torch.stack([torch.cos(env.heading), torch.sin(env.heading)], -1).unsqueeze(1)
    left = torch.stack([-fwd[..., 1], fwd[..., 0]], -1)
    rel = p_npc - env.pos.unsqueeze(1)
    fx = (rel * fwd).sum(-1); fy = (rel * left).sum(-1)
    ahead = ((fx > 0) & (fx < 20.0) & (fy.abs() < 3.2)).any(-1)
    gap = torch.where((fx > 0) & (fy.abs() < 3.2), fx, torch.full_like(fx, 1e6)).min(-1).values
    v_follow = ((gap - 6.0) / 12.0).clamp(0.0, 1.0) * v_cruise
    v_t = torch.where(ahead, torch.minimum(v_t, v_follow), v_t)

    # yield on a left turn while an oncoming car is close
    oncoming = ((fx > 0) & (fx < 28.0) & (fy > 2.0) & (fy < 8.0)).any(-1)
    yield_left = (turn == LEFT) & (d_corner < 14.0) & (d_corner > -6.0) & oncoming
    v_t = torch.where(yield_left, torch.minimum(v_t, torch.full_like(v_t, 1.5)), v_t)   # creep, never a dead stop
    # inside an intersection with another car in it: crawl until it is clear
    near_c = torch.cdist(env.pos, env.city.centres).min(dim=1).values < c.road_width / 2 + 2.0
    shared = near_c & ((rel.norm(dim=-1) < 12.0) & (fx > -2.0)).any(-1)
    if crawl_shared:
        v_t = torch.where(shared, torch.minimum(v_t, torch.full_like(v_t, 2.5)), v_t)

    a_thr = ((v_t - env.speed) / 3.0).clamp(-1, 1)
    return torch.stack([a_steer, a_thr], 1)
