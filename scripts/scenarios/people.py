"""Plausible shopper trajectories through the store (pure Python).

Figures walk entrance -> aisle -> dwell at a shelf -> checkout queue -> exit, as
simple upright mannequin boxes. IDs are **session-scoped random tokens** (no
cross-visit identity, no biometrics) to respect the task's privacy-by-design rule.
Mannequins are procedural primitives (own geometry, no external asset), so there is
no third-party licence to record for the default set.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

from state import Entity

Vec3 = Tuple[float, float, float]

PERSON_SIZE: Vec3 = (0.5, 0.42, 1.75)
PERSON_Z = PERSON_SIZE[2] / 2


def _heading(a: Vec3, b: Vec3) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _resample(waypoints: List[Tuple[Vec3, float]], fps: int, n_frames: int,
              speed: float) -> List[Tuple[int, Vec3, float]]:
    """Turn (point, dwell_s) waypoints into per-frame keyframes at a walking speed."""
    kf: List[Tuple[int, Vec3, float]] = []
    t = 0.0
    prev = waypoints[0][0]
    for i, (pt, dwell) in enumerate(waypoints):
        if i > 0:
            dist = math.dist(prev[:2], pt[:2])
            t += dist / speed
        frame = int(round(t * fps))
        heading = _heading(prev, pt) if i > 0 else _heading(pt, waypoints[1][0])
        kf.append((frame, pt, heading))
        if dwell > 0:
            t += dwell
            kf.append((int(round(t * fps)), pt, heading))
        prev = pt
    # clamp to sequence length
    return [(min(f, n_frames - 1), c, r) for f, c, r in kf]


def generate_people(plan: dict, n_people: int, n_frames: int, fps: int,
                    seed: int) -> List[Entity]:
    rng = random.Random(seed)
    store = plan["store"]
    w2, d2 = store["width"] / 2, store["depth"] / 2
    entrance = (w2 - 3.5, -d2 + 1.0, PERSON_Z)
    exit_pt = (w2 - 1.5, -d2 + 0.2, PERSON_Z)
    # a few aisle x-lanes the shopper can walk down (gondola gaps)
    aisle_x = [g["x"] + plan["bay_w"] for g in plan["gondolas"][:5]]
    checkouts = [(x, -d2 + 2.6, PERSON_Z) for x in (1.0, 2.2, 3.4)]
    people: List[Entity] = []
    for i in range(n_people):
        token = "s_" + "".join(rng.choice("0123456789abcdef") for _ in range(10))
        lane = rng.choice(aisle_x)
        dwell_y = rng.uniform(-2.0, 4.0)
        checkout = rng.choice(checkouts)
        speed = rng.uniform(0.9, 1.4)
        wp: List[Tuple[Vec3, float]] = [
            (entrance, 0.0),
            ((lane, -d2 + 3.5, PERSON_Z), 0.0),
            ((lane, dwell_y, PERSON_Z), rng.uniform(1.5, 4.0)),   # dwell at a shelf
            ((lane, dwell_y + rng.uniform(1.5, 4.0), PERSON_Z), rng.uniform(0.0, 2.0)),
            ((checkout[0], checkout[1] + 1.5, PERSON_Z), 0.0),
            (checkout, rng.uniform(2.0, 5.0)),                     # checkout queue dwell
            (exit_pt, 0.0),
        ]
        kf = _resample(wp, fps, n_frames, speed)
        start = int(rng.uniform(0, max(1, n_frames * 0.3)))
        kf = [(min(f + start, n_frames - 1), c, r) for f, c, r in kf]
        people.append(Entity(
            id=f"person_{i}", kind="person", cls="person", pass_index=-1,  # set by caller
            center=kf[0][1], size=PERSON_SIZE, rot_z=kf[0][2], annotate=True,
            present_from=kf[0][0], present_to=kf[-1][0], path=kf,
            meta=dict(session_id=token, speed=round(speed, 3))))
    return people


def trajectories_gt(people: List[Entity], n_frames: int) -> dict:
    """Store-floor (x, y) per frame per session — the people-movement ground truth."""
    out = {}
    for p in people:
        track = []
        for f in range(n_frames):
            if not p.present_at(f):
                continue
            (cx, cy, _), heading = p.pose_at(f)
            track.append(dict(frame=f, x=round(cx, 3), y=round(cy, 3),
                              heading=round(heading, 3)))
        out[p.meta["session_id"]] = dict(entity_id=p.id, track=track)
    return out
