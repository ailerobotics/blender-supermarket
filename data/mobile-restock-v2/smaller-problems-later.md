# Smaller problems, deferred

Deferred items that do not block the first full cycle; revisit after the cycle completes.

- **Uncouple and leave the cart on return** (fm413, 2026-10-20): `return` currently tows the cart home coupled (goal_articulation 0.2); there is no uncouple primitive (only `release_cart_brake`). Captain accepted towing-home for now. Later: add an uncouple step so `return` leaves the cart parked and the robot drives home empty.
- **Native SurfaceGripper suction non-engagement** (gripped_n=0 always; the SIM_ATTACH_FALLBACK weld is the working attach) -- deferred per fm405.
