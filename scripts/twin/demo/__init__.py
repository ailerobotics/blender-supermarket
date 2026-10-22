"""Choreographed seven-stage restock DEMO (captain 2026-10-20).

A deliberately choreographed walkthrough of the full restock cycle in the mission's
Isaac scene, paired with a live decision/reason tree panel. This is a SEPARATE demo
mode: it never changes the autonomous mission's defaults or behaviour.

What is choreographed vs measured (see ``data/sim-demo-video-v1/report.md``):

* CHOREOGRAPHED (scripted, kinematic): the base motion along the real mission poses
  and the PR154 tow-lane centrelines, the arm joint keyframes, the carton weld/parent
  and the cart coupling. Physics is partly off (gravity zeroed) so the scripted poses
  hold for the shot.
* MEASURED (read from the live USD stage each frame): every value shown beside a
  decision node in the tree panel -- base distance/yaw to a pose, hitch gap, lane
  progress/remaining metres, forward-capsule obstacle clearance, cage/slot free space,
  carton-vs-target error, articulation angle. These are read from the prim world
  transforms at that frame by :mod:`twin.demo.scene_reader`, never typed constants.

Modules:

* :mod:`twin.demo.tree_model`  -- the PR153 behaviour-tree structure as a renderable tree.
* :mod:`twin.demo.measures`    -- pure geometry turning raw scene reads into (value, reason).
* :mod:`twin.demo.choreography`-- the keyframed stage timeline (scripted motion).
* :mod:`twin.demo.panel`       -- PIL render of the decision/reason tree panel per frame.
* :mod:`twin.demo.compositor`  -- composite scene view + panel and encode the mp4.
* :mod:`twin.demo.scene_reader`-- read prim world poses from the live Isaac stage (GPU only).
* :mod:`twin.demo.run_demo_gpu`-- the guarded GPU capture entry (flock + VRAM guard).

The REAL mission's demo cut (not choreographed; CPU, run dir read-only):

* :mod:`twin.demo.run_evidence`   -- read a mission run dir (events, nav-trace, frames, labels).
* :mod:`twin.demo.run_check`      -- pass/fail checks (stages, success, carton on cart / on bay).
* :mod:`twin.demo.walkthrough_cut`-- title/stage/closing cards + captions + time-lapse -> mp4.
"""
