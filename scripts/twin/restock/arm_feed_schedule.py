"""Per-phase arm depth-feed schedule (pure, CPU) -- the phase time-sharing contract.

The nvblox ESDF must be live wherever an arm leg moves, but the two depth feeds cannot both run
in every stage: the wrist feed (camera_0) always runs during arm phases, while the mast feed
(camera_1) is the only live feed while a carried carton fills the wrist frame -- yet the mast does
NOT fit beside the wrist during the ``live_pose`` transfer peak (26 MiB free), whereas the
collect/carry/set-down stages have ~2.9 GB free (sim-perception-architecture-s1 s7). So the feeds
are phase-time-shared: enable the mast for collect/carry/set-down, keep it OFF for transfer.

This module states that schedule once (which feeds are live in a given phase) so the Isaac-side
render lifecycle (``restock.native_cameras.set_render_products_enabled`` /
``destroy_camera_render_product``) and the pre-boot VRAM budget (:mod:`gpu_budget`) agree. It is
pure data -- it toggles nothing by itself; the integration render-lifecycle call consumes it.
"""
from __future__ import annotations

from typing import Iterable

FEED_WRIST = 'wrist'      # nvblox camera_0, eye-in-hand on ur_tool0
FEED_MAST = 'mast'        # nvblox camera_1, body-mounted on base_link

#: Phases where the WRIST feeds the ESDF (gripper near the mapped workspace, carton not yet held).
WRIST_PHASES = frozenset({'observe', 'approach', 'contact', 'grasp', 'lift'})
#: Phases where the carried carton fills the wrist frame, so the MAST is the live ESDF feed.
CARRY_PHASES = frozenset({'lift', 'tuck', 'pan', 'transit', 'preplace', 'place', 'retreat'})


def feeds_active_in_phase(phase, *, mast_enabled=False, live_pose_transfer=False) -> frozenset:
    """Which depth feeds should be rendering in ``phase``.

    :param mast_enabled: the mast feed is configured on for this run.
    :param live_pose_transfer: this run's transfer uses ``live_pose`` (the VRAM peak stage) -- the
        mast is force-dropped there regardless of ``mast_enabled`` (it does not fit beside the
        wrist at the transfer peak).
    """
    active = set()
    if phase in WRIST_PHASES or phase in CARRY_PHASES:
        active.add(FEED_WRIST)
    if mast_enabled and phase in CARRY_PHASES and not (live_pose_transfer and phase in
                                                       {'preplace', 'place', 'retreat'}):
        active.add(FEED_MAST)
    return frozenset(active)


def peak_feeds_over_phases(phases: Iterable, *, mast_enabled=False, live_pose_transfer=False):
    """The union of feeds live across ``phases`` whose count is largest -- the VRAM-sizing set.

    Returns the single phase's feed set with the most feeds (the binding peak), so the budget is
    sized for the worst concurrent feed load the schedule ever reaches in those phases.
    """
    worst = frozenset()
    for phase in phases:
        active = feeds_active_in_phase(phase, mast_enabled=mast_enabled,
                                       live_pose_transfer=live_pose_transfer)
        if len(active) > len(worst):
            worst = active
    return worst
