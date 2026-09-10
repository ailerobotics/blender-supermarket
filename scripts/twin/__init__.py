"""Tooling to stand the fleet's UR10 (shelfbot_newgripper) inside the supermarket
twin under Isaac Sim.

Everything that needs a GPU/Isaac-Sim boot lives behind an explicit guard in
``compose_robot_in_store.py``; the pure logic here (pose maths, collider policy,
GPU refusal, CLI/dry-run) imports neither ``pxr`` nor ``isaacsim`` and is unit
tested without a simulator. ``usd_probe`` reads the store USD with the pxr
bindings bundled inside Isaac Sim WITHOUT booting the simulator (no RTX, no GPU
session) and is the source of the measured facts in ``docs/twin_usd_facts.json``.
"""
