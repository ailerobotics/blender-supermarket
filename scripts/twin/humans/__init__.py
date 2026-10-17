"""Optional simulated people + human-safety layer for the mobile restock mission.

Everything in this package is **opt-in and default off**: with no ``humans`` block
in the runtime config the mission composes, steps and records byte-identically.
Three independent pieces, each with its own config and offline (CPU) tests:

* ``paths`` / ``config`` / ``people`` / ``isaac_people`` -- scripted shoppers and
  staff authored into the Isaac scene on deterministic paths (piece 1);
* ``perception`` / ``detection`` -- the fleet human-safety decision logic (vendored
  verbatim from ``ur10-setup/cams/human_safety``) fed either by the real detector on
  the mission camera topics or, behind an explicit ground-truth option, by boxes
  projected from the known people poses (piece 2);
* ``reaction`` -- a configurable slow/stop layer for the base and a pause/resume for
  the arm, driven by the published safety zone (piece 3).

See ``scripts/twin/humans/README.md`` and
``/home/manishsaini/firstmate/data/sim-restock-humans-v1/report.md``.
"""
