"""sim-gripper-status-relay-v1: the opt-in switch that makes the LIVE host-facing /sensors capture carry
the gripper contact-feedback (gripper_enabled / seal_gap_mm / seal_pad_angle_deg / seal_hit_target +
a sim-clock sample time) to the host every tick during the pick.

Why (coldwin65): PR 356 added those fields to the ``Sensors`` packet and populated them in the
non-host-facing ``isaac_sensors.Sensors.capture`` helper, but the mission's host-facing /sensors come
from ``isaac_factory.CollectTargetBridge`` (``capture`` / ``_capture_ground_truth_collect`` /
``_capture_transfer`` / ``_survival_packet``), which NEVER populated them. So every /sensors packet that
reached the host carried the all-default feedback (``gripper_enabled=False``, ``seal_gap_mm=None``,
``seal_hit_target=False``) regardless of the true Isaac state -- in coldwin65 the Isaac plugin was
``Closing``/``Closed`` with the pad gap closing 9.25->4.87 mm, yet the host's ``contact_require_suction_
enabled`` guard read the default ``False`` and e-stopped a pick that would have sealed (coldwin64 sealed
the identical pick only because nothing read that stale view). The last-value /manipulation/status relay
PR 356 blamed is NOT implicated here: the enable landed (reissue-each-tick on; SEAL_DIAG ``command:close``
``enabled=True`` throughout), and /sensors has no heartbeat / last-value drop -- the fields were simply
never set.

What: with ``RESTOCK_CONTACT_STATUS_FEEDBACK=1`` the host-facing capture paths add the contact-feedback
kwargs (read best-effort from the ``NativePanelContact`` probe, same source as the SEAL_DIAG log) plus
``gripper_status_time=now`` so the host can tell a fresh report from no-report. Unset/0 => no kwargs added
=> the /sensors packet (and the whole run) is byte-identical.

One switch, read by the Isaac process (the capture that fills the packet). The host-side guards that
CONSUME the feedback stay their own opt-in HOST keys (``contact_require_suction_enabled`` /
``contact_accept_seal_diag``); for a trustworthy guard the next run sets this env AND those keys (see
docs/examples/gripper-status-relay-fragment.json).

Pure (no Isaac/ROS/pxr imports) so the gate is CPU-tested.
"""
import os

ENV = 'RESTOCK_CONTACT_STATUS_FEEDBACK'


def enabled(env=None):
    """True only for an explicit truthy ``RESTOCK_CONTACT_STATUS_FEEDBACK``; unset/0/garbage => False."""
    e = os.environ if env is None else env
    return str(e.get(ENV, '')).strip().lower() in ('1', 'true', 'yes', 'on')
