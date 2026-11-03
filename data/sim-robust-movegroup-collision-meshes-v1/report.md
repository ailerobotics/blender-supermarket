# sim-robust-movegroup-collision-meshes-v1

Make the resolved URDF's mesh references loadable by `move_group`, behind an opt-in HOST
switch, so MoveIt's FCL `ValidateSolution` runs against real arm link geometry instead of
empty geometry. CPU evaluation of what changes, plus one bounded GPU proof config for the
integration lane. **No GPU run was performed by this lane** (the validator holds the GPU).

## The defect

The host stack writes `move_group`'s `robot_description` from the generated
`mobile.urdf` (`restock_host.launch`, `description()`). `restock_host.generate_assets`
resolves every `package://` mesh to a **bare absolute filesystem path**, e.g.

```
<mesh filename="/opt/ros/jazzy/share/ur_description/meshes/ur10/collision/base.stl"/>
```

MoveIt loads every link visual/collision mesh through `resource_retriever`, which treats the
`filename` as a **URI**. A bare path has no scheme, so the retriever rejects it:

- move_group's C++/libcurl retriever logs **`URL rejected: No host part in the URL`** (the
  line PR 229 surfaced in every coldwin7 `host-stack.log` at move_group start);
- the Python `resource_retriever.get` raises `ValueError: unknown url type: '/opt/ros/...'`
  — same root cause (no URI scheme).

Net effect: move_group loads **no** arm-link collision or visual geometry. The cuMotion XRDF
spheres (read separately from `urdf_file_path`) still do the real self/scene collision check,
and MoveIt's FCL `ValidateSolution` runs against **empty link geometry**.

## CPU proof that bare paths fail and `file://` resolves

`scripts/check_movegroup_mesh_uris.py` on the deployed
`~/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf` (16 meshes: 7 collision links,
9 visual):

| robot_description | collision loadable | visual loadable |
|---|---|---|
| **OFF** (shipped, bare absolute paths) | 0 | 0 |
| **ON** (`moveit_mesh_uris`, `file://` URIs) | 7 | 9 |

Cross-checked against the **real** `resource_retriever` (ROS-sourced shell): OFF **16/16
FAIL** (`ValueError: unknown url type`), ON **16/16 LOAD** (e.g. `base.dae` → 147 300 bytes).
The model in `restock_host.mesh_uri.retriever_loadable` agrees with the real loader
mesh-for-mesh. Unit proof: `scripts/twin/tests/test_mesh_uri.py` (retriever classification,
the rewrite, idempotency, and the loader-level empty-off / non-empty-on evaluation).

## The fix (opt-in, default off, byte-identical when off)

- `scripts/twin/restock_host/mesh_uri.py` — `normalize_mesh_uris(urdf_text)` rewrites each
  **bare absolute** `<mesh filename>` to a `file://` URI; `package://` and existing `file://`
  are untouched; a relative bare path is left as-is (the host never ships one, and inventing a
  cwd could hide a genuinely unresolved mesh). Textual only — no geometry/link/joint/inertial/
  transform changes, no mesh file read or moved, idempotent.
- `scripts/twin/restock_host/launch.py` — reads the HOST config
  (`restock_manipulation/ros_host_config.json`, the same file the manip bridge reads) once and,
  when `moveit_mesh_uris` is true, applies the rewrite to `move_group`'s `robot_description`.
  Off (default) ⇒ `robot_description` is the file verbatim (byte-identical, the historical
  empty-geometry path). `cuMotion`'s `urdf_file_path` and `trajectory_host`'s `--urdf` are
  untouched (both read the file directly and already handle bare absolute paths; cuMotion
  works today).
- `scripts/twin/restock_manipulation/config_preflight.py` — `moveit_mesh_uris` registered as a
  **HOST** key (the switch-key registry the other robustness keys use), so a run can
  `--require moveit_mesh_uris` and prove it loads from the host config before any GPU boot.
- `scripts/twin/restock_manipulation/ros_host_config.json` — ships `"moveit_mesh_uris": false`.

## FCL risk assessment (which previously-accepted plans could now be rejected)

With real link geometry loaded, MoveIt's FCL `ValidateSolution` checks the linearly
interpolated path for (a) link-vs-link self-collision and (b) link-vs-attached-object. World
obstacles stay in cuMotion's nvblox ESDF, not move_group's FCL scene, so this change does **not**
add link-vs-world checks.

- **Self-collision (all four families).** The XRDF spheres are a **conservative cover that
  encloses each link's collision mesh** (`generate_assets` / `coverage.py`), and the generated
  SRDF disables exactly the same pairs the XRDF excludes — including
  `ur_forearm_link`↔`ur_wrist_2_link` (verified in the deployed `mobile.srdf`). Loading the
  real, **tighter** meshes therefore makes self-collision checking *more* permissive, never
  less. **No self-collision-only plan is newly at risk.**
- **collect pick / transfer pick — NOT at risk.** Both plan *before* the carton is attached, so
  only self-collision applies (now more permissive). The reach target (carton on shelf/cart)
  lives in the ESDF, not the FCL scene.
- **collect place / transfer preplace & place — AT RISK (the families to watch).** These carry
  the carton as a MoveIt **attached object** while the arm is folded. With empty link geometry
  the carton-vs-arm check was vacuous; with real link meshes, carton-vs-(**non-touch**) arm link
  is now a real check (carton-vs-gripper stays allowed via SRDF `touch_links`). A fold that
  brings the carton near the forearm could produce a new `collision-detected` /
  `INVALID_MOTION_PLAN`. If it appears it is a **true clearance finding to report**, not a
  regression of this switch. The known `ur_wrist_2`-vs-forearm *carry* jam is arm-vs-arm and
  stays disabled in the SRDF, so it is not caught by FCL either (handled at the IK layer).

Loader-level evidence for the empty-off / non-empty-on claim is pinned in
`test_mesh_uri.py::LoaderLevelEvaluationTests`.

## Bounded GPU proof (for the integration lane, not run here)

`docs/examples/movegroup-mesh-uris-mission-fragment.json`. Staged **transfer-only**
(`--start-at transfer --start-staging recorded_bay_park --stop-after verify`) with
`moveit_mesh_uris` on first — PASS = no `URL rejected` lines in `host-stack.log`, collision
geometry loaded, and the staged transfer still completes with no new plan rejection vs the
key-off baseline. Only then the full capstone (coldwin7 shape) with the key on. Scheduled
through `/tmp/aile-gpu.lock`, never forced, never touching whispr.

## Scope

Out of scope and untouched: the XRDF sphere model, cuMotion settings, any planner tolerance.
The fix only changes the URI *scheme* of mesh references in move_group's `robot_description`
when the opt-in key is on.
