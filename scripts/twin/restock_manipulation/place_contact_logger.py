"""PhysX contact-pair logger for the collect place DESCENT (sim-pad-down-descent-jam-physics-v1).

coldwin98-100 jammed the pad-down set-down ~0.11 m above the shelf while the CPU cage-clearance proof said
every link cleared by >= 0.10 m -- the model and the physics disagreed, and the contacting pair could only
be found by a CPU FK replay against the composed USD colliders (``scripts/check_pad_down_shelf_contact.py``:
the gripper suction HOUSING vs ``/Rollwagen/chassis/shelf``). So that a disagreement like this can NEVER
again be argued from the model alone, this module logs the REAL PhysX contact pair during the place descent:
when the arm / gripper / carton first touches a cart prim, and whenever that touch deepens, it prints

    CONTACT_PAIR a=<robot-or-carton prim> b=<cart prim> depth=<mm> sep=<m> phase=<phase> event=first|deepen

directly from the ``omni.physx`` contact-report callback (the SAME API the cart diagnostic uses --
``PhysxContactReportAPI`` + ``subscribe_contact_report_events``), so the next GPU run NAMES the contact
instead of leaving it to a model argument.

Opt-in + byte-identical off: nothing constructs or attaches a logger unless the ``place_contact_logger``
HOST/runtime block is present. The boot wiring is VIEW-SAFE (sim-place-contact-logger-view-safe-v1): the
PhysxContactReportAPI is authored BEFORE ``world.reset()`` creates the tensor-backed simulation view
(:func:`wire_place_contact_logger_pre_reset`, called from ``physics_setup.assemble``), and only the
callback is subscribed AFTER reset (:func:`wire_place_contact_logger`, called from ``isaac_factory.create``)
-- so no prim the view covers is ever re-authored after the view exists. Re-authoring such a prim post-reset
was the coldwin101 crash (the view was invalidated and every arm DOF/transform read failed). The PURE core
(path classification + the per-pair first/deepen log decision + the line format) needs no Isaac and is
CPU-tested; only :meth:`PlaceContactLogger.author_contact_reports` / :meth:`subscribe` touch ``omni.physx``
/ ``pxr`` (lazily, with injectable hooks so the boot-path wiring is testable without a GPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

CONTACT_PAIR = 'CONTACT_PAIR'
PLACE_CONTACT_LOGGER_KEY = 'place_contact_logger'

# Prim-path prefixes that classify a contact side. A relevant pair is one side ROBOT/GRIPPER or CARTON and
# the other side a CART prim -- the place-descent contact the model must never again be blind to.
DEFAULT_ARM_PREFIXES = ('/MobileRobot',)                       # arm links + the gripper housing
DEFAULT_CARTON_PREFIXES = ('/RestockScenario/HandlingUnit',)   # the welded handling unit
DEFAULT_CART_PREFIXES = ('/Rollwagen',)                        # deck / shelf / posts / bars / drawbar

# The gripper-housing link basenames (the jam contact is one of these vs the shelf); used only to label the
# contact KIND in the log, never to filter (any robot link vs a cart prim is logged).
GRIPPER_HOUSING_LINKS = ('suction_cups', 'cube_without_surface_plates',
                         'surface_plate_1', 'surface_plate_2', 'surface_plate_3',
                         'surface_plate_4', 'surface_plate_5', 'surface_plate_6')


def _starts_any(path: str, prefixes: Iterable[str]) -> bool:
    p = str(path)
    return any(p == pre or p.startswith(pre.rstrip('/') + '/') or p.startswith(pre) for pre in prefixes)


def classify_contact(paths: Iterable[str], *, arm_prefixes=DEFAULT_ARM_PREFIXES,
                     carton_prefixes=DEFAULT_CARTON_PREFIXES,
                     cart_prefixes=DEFAULT_CART_PREFIXES) -> Optional[Tuple[str, str, str]]:
    """Classify a contact's prim paths into ``(robot_or_carton_prim, cart_prim, kind)`` or ``None``.

    ``paths`` are the prim paths of a contact header (collider0/collider1, optionally actor0/actor1). A
    RELEVANT contact has exactly one CART side (``/Rollwagen/...``) and at least one ROBOT/GRIPPER
    (``/MobileRobot/...``) or CARTON (``/RestockScenario/HandlingUnit``) side -- the place-descent contact.
    ``kind`` is ``'carton'`` when the non-cart side is the handling unit, ``'gripper'`` when it is a known
    gripper-housing link, else ``'arm'``. A contact with NO cart side but a CARTON side and a ROBOT side is
    also relevant (sim-place-release-residual-bind-v1): ``'carton-gripper'`` when the robot side is a gripper
    housing link (the residual-bind panel contact), else ``'carton-arm'``. Returns ``None`` for any other
    contact (cart-vs-floor, arm-vs-arm, carton-vs-floor). Pure/CPU."""
    paths = [str(p) for p in paths if p]
    cart = [p for p in paths if _starts_any(p, cart_prefixes)]
    carton = [p for p in paths if _starts_any(p, carton_prefixes)]
    arm = [p for p in paths if _starts_any(p, arm_prefixes) and p not in carton]
    if not cart:
        # sim-place-release-residual-bind-v1: a carton-vs-gripper/arm contact has NO cart side and is
        # INVISIBLE to the cart-only filter above -- yet it is exactly the coldwin106 residual bind (a
        # still-live gripper panel collider coincident with the carton face lifts it on the clearance
        # lift). Log it so the next run surfaces the panel contact the model was blind to.
        if carton and arm:
            g = arm[0]
            # match a gripper-housing link name ANYWHERE in the path (contact headers report the collider
            # prim, e.g. .../surface_plate_3/collisions/mesh_0, so the last component is not the link name).
            kind = ('carton-gripper' if any(('/' + link) in g for link in GRIPPER_HOUSING_LINKS)
                    else 'carton-arm')
            return (carton[0], g, kind)
        return None
    if carton:
        return (carton[0], cart[0], 'carton')
    if arm:
        a = arm[0]
        base = a.rsplit('/', 1)[-1]
        kind = 'gripper' if base in GRIPPER_HOUSING_LINKS else 'arm'
        return (a, cart[0], kind)
    return None


def format_contact_line(a: str, b: str, separation_m: float, *, phase: str, event: str, kind: str) -> str:
    """The :data:`CONTACT_PAIR` log line. ``separation_m`` is the PhysX contact separation (NEGATIVE while
    penetrating); ``depth`` is the penetration in mm (``max(0, -sep)``)."""
    depth_mm = max(0.0, -float(separation_m)) * 1000.0
    return ('%s a=%s b=%s kind=%s depth=%.1fmm sep=%.4fm phase=%s event=%s'
            % (CONTACT_PAIR, a, b, kind, depth_mm, float(separation_m), phase, event))


@dataclass
class PlaceContactLogger:
    """Observe PhysX contacts between the arm/gripper/carton and the cart during the place descent and log
    each pair the first time it touches and whenever it deepens (sim-pad-down-descent-jam-physics-v1).

    PURE except :meth:`attach` (``omni.physx``/``pxr``). ``phase`` labels the log; set it to the live stage
    (``collect``/``place``) via :meth:`set_phase`. ``deepen_step_m`` suppresses log spam: a pair re-logs only
    when its penetration grows by at least this much past its last logged depth."""
    arm_prefixes: Tuple[str, ...] = DEFAULT_ARM_PREFIXES
    carton_prefixes: Tuple[str, ...] = DEFAULT_CARTON_PREFIXES
    cart_prefixes: Tuple[str, ...] = DEFAULT_CART_PREFIXES
    deepen_step_m: float = 0.005
    log: Callable[[str], None] = print
    phase: str = 'place'
    # sim-place-release-residual-bind-v1: substrings that mark a CART prim as the placement SUPPORT (deck /
    # shelf) rather than an obstruction (post / bar). A carton-vs-support contact is the POSITIVE freedom
    # signal the release_free_positive gate consumes (``support_contact_seen``); a carton-vs-post contact is
    # NOT support. Lower-cased substring match on the cart-side prim path.
    support_tokens: Tuple[str, ...] = ('deck', 'shelf', 'support', 'bay')
    _support_seen: bool = False
    _seen: Dict[Tuple[str, str], float] = field(default_factory=dict)
    _subscription: object = None
    _armed_bodies: int = 0          # rigid bodies the contact-report API was authored onto
    _attach_phase: str = ''         # WHEN the authoring ran: 'pre-reset' on the view-safe boot path

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)

    def classify(self, paths) -> Optional[Tuple[str, str, str]]:
        return classify_contact(paths, arm_prefixes=self.arm_prefixes,
                                carton_prefixes=self.carton_prefixes, cart_prefixes=self.cart_prefixes)

    def on_contact(self, a: str, b: str, separation_m: float, *, kind: str = 'arm') -> Optional[str]:
        """Record a classified contact; return the :data:`CONTACT_PAIR` line to log (and log it via
        ``self.log``) on the FIRST touch of a pair or when its penetration DEEPENS by ``deepen_step_m``,
        else ``None``. ``separation_m`` negative = penetrating. Pure (no Isaac)."""
        key = (a, b)
        pen = max(0.0, -float(separation_m))
        prev = self._seen.get(key)
        if prev is None:
            event = 'first'
        elif pen >= prev + float(self.deepen_step_m):
            event = 'deepen'
        else:
            return None
        self._seen[key] = pen
        # sim-place-release-residual-bind-v1: latch a carton-vs-SUPPORT contact (the free-release positive
        # signal). Only a CARTON side touching a cart prim whose path names a support (deck/shelf) counts --
        # a carton-vs-post contact is an obstruction, not support, and an arm/gripper contact is not the
        # carton resting. Monotonic until clear_support_contact() (reset at the detach by the bridge).
        if kind == 'carton':
            bl = str(b).lower()
            if any(tok in bl for tok in self.support_tokens):
                self._support_seen = True
        line = format_contact_line(a, b, separation_m, phase=self.phase, event=event, kind=kind)
        self.log(line)
        return line

    def support_contact_seen(self) -> bool:
        """sim-place-release-residual-bind-v1: True when a carton-vs-SUPPORT contact pair has been observed
        since the last :meth:`clear_support_contact`. The POSITIVE freedom signal the release_free_positive
        gate reads (a welded carton held OFF the deck reports none). PURE (no Isaac)."""
        return bool(self._support_seen)

    def clear_support_contact(self) -> None:
        """Reset the carton-vs-support latch (call at the weld detach so the release window starts clean)."""
        self._support_seen = False

    def handle_header_paths(self, paths, separation_m) -> Optional[str]:
        """Classify ``paths`` and, if a robot/carton-vs-cart contact, log it. Returns the line or ``None``.
        The single seam the Isaac callback and the CPU wiring test both drive. Pure (no Isaac)."""
        hit = self.classify(paths)
        if hit is None:
            return None
        a, b, kind = hit
        return self.on_contact(a, b, separation_m, kind=kind)

    # ------------------------------------------------- Isaac attach (lazy; injectable hooks; VIEW-SAFE split)
    # coldwin101 crashed because the ORIGINAL single :meth:`attach` applied the PhysxContactReportAPI to the
    # 46 rigid bodies AFTER ``world.reset()`` had built the tensor-backed simulation view over the SAME prims
    # -- re-authoring a prim the view covers invalidates the view, so every arm DOF/transform read failed
    # from the first mission step. The fix splits the two responsibilities: :meth:`author_contact_reports`
    # (PURE USD authoring, MUST run BEFORE the view is created, i.e. before ``world.reset()``) and
    # :meth:`subscribe` (registering a Python callback on the omni.physx simulation interface, which
    # re-authors NO prim, so it is view-safe to call AFTER reset -- the same subscribe timing PR 383 used).
    def author_contact_reports(self, stage, *, apply_contact_report=None, is_rigid_body=None,
                               threshold_m: float = 0.0, phase: str = 'pre-reset') -> int:
        """Apply a PhysX contact-report API to every rigid body under the robot/carton/cart prefixes on
        ``stage``. PURE USD authoring -- it touches no tensor-backed simulation view, so it MUST be called
        BEFORE ``world.reset()`` (and before any ArticulationView / RigidPrimView initialisation) so that no
        prim the view covers is re-authored after the view exists. Returns the number of bodies instrumented
        and records it (``_armed_bodies``) with the attach ``phase`` (``_attach_phase``) for the armed line.

        ``apply_contact_report(prim, threshold_m)`` and ``is_rigid_body(prim)`` default to the real
        ``PhysxSchema.PhysxContactReportAPI`` / ``UsdPhysics.RigidBodyAPI`` calls but are INJECTABLE so the
        boot-path wiring (including the view-ordering test) is testable with NO pxr/Isaac. Idempotent-safe."""
        if apply_contact_report is None or is_rigid_body is None:
            from pxr import UsdPhysics                               # pxr only; usd-core has it
            if apply_contact_report is None:
                from pxr import PhysxSchema
                def apply_contact_report(prim, thr):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(float(thr))
            if is_rigid_body is None:
                is_rigid_body = lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)

        prefixes = tuple(self.arm_prefixes) + tuple(self.carton_prefixes) + tuple(self.cart_prefixes)
        n = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not _starts_any(path, prefixes):
                continue
            if is_rigid_body(prim):
                apply_contact_report(prim, threshold_m)
                n += 1
        self._armed_bodies = n
        self._attach_phase = str(phase)
        return n

    def subscribe(self, sim_interface, *, decode_path=None):
        """Subscribe this logger's callback to ``sim_interface`` (``omni.physx``'s
        ``get_physx_simulation_interface()``). Registers a Python callback only -- it re-authors NO prim, so
        it is VIEW-SAFE to call AFTER ``world.reset()``. Returns the subscription token.

        ``decode_path(int)`` defaults to the real ``PhysicsSchemaTools.intToSdfPath`` call but is INJECTABLE,
        and when left ``None`` it is resolved LAZILY on the first contact -- so ``subscribe`` itself imports
        no ``pxr`` and the post-reset wiring is testable without usd-core."""
        resolver = {'fn': decode_path}

        def _callback(headers, data):
            try:
                fn = resolver['fn']
                if fn is None:
                    from pxr import PhysicsSchemaTools
                    fn = resolver['fn'] = lambda i: str(PhysicsSchemaTools.intToSdfPath(i))
                for header in headers:
                    paths = [fn(getattr(header, k))
                             for k in ('collider0', 'collider1', 'actor0', 'actor1')]
                    pts = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
                    sep = min((float(p.separation) for p in pts), default=0.0)
                    self.handle_header_paths(paths, sep)
            except Exception as exc:                                 # a logger must never crash the sim
                self.log('%s_ERROR %s' % (CONTACT_PAIR, exc))

        self._subscription = sim_interface.subscribe_contact_report_events(_callback)
        return self._subscription

    def attach(self, stage, sim_interface, *, apply_contact_report=None, decode_path=None,
               threshold_m: float = 0.0):
        """Author the contact-report API then subscribe, in one call. Convenience for a caller that holds the
        stage BEFORE the view exists (and for the CPU attach test). Returns the number of bodies instrumented.
        The VIEW-SAFE boot path does NOT use this -- it splits :meth:`author_contact_reports` (pre-reset, in
        ``physics_setup.assemble``) from :meth:`subscribe` (post-reset, in ``isaac_factory.create``)."""
        n = self.author_contact_reports(stage, apply_contact_report=apply_contact_report,
                                        threshold_m=threshold_m, phase='attach')
        self.subscribe(sim_interface, decode_path=decode_path)
        return n

    def detach(self):
        self._subscription = None


def _build_logger_from_config(config, *, log=print) -> Optional[PlaceContactLogger]:
    """Construct a :class:`PlaceContactLogger` from the ``place_contact_logger`` HOST/runtime block, or
    ``None`` when the block is absent/false (byte-identical off -- nothing constructed). No Isaac import."""
    block = config.get(PLACE_CONTACT_LOGGER_KEY)
    if not block:
        return None
    opts = block if isinstance(block, dict) else {}
    return PlaceContactLogger(
        arm_prefixes=tuple(opts.get('arm_prefixes', DEFAULT_ARM_PREFIXES)),
        carton_prefixes=tuple(opts.get('carton_prefixes', DEFAULT_CARTON_PREFIXES)),
        cart_prefixes=tuple(opts.get('cart_prefixes', DEFAULT_CART_PREFIXES)),
        deepen_step_m=float(opts.get('deepen_step_m', 0.005)), log=log)


def wire_place_contact_logger_pre_reset(stage, config, *, log=print,
                                        apply_contact_report=None,
                                        is_rigid_body=None) -> Optional[PlaceContactLogger]:
    """PRE-RESET half of the boot wiring (sim-place-contact-logger-view-safe-v1). Construct the logger and
    AUTHOR the PhysX contact-report API onto the stage's rigid bodies BEFORE ``world.reset()`` creates the
    tensor-backed simulation view -- so no prim the view covers is ever re-authored after the view exists
    (the coldwin101 crash). Called from ``physics_setup.assemble`` right before ``world.reset()`` with the
    live stage. Returns the AUTHORED-but-not-yet-subscribed logger (the caller stashes it on the assembly so
    :func:`wire_place_contact_logger` can subscribe it post-reset), or ``None`` when the block is absent
    (byte-identical -- nothing constructed, applied or imported).

    Block schema (all optional)::

        "place_contact_logger": true | {
            "deepen_step_m": 0.005,          # re-log a pair only when its penetration grows this much
            "threshold_m": 0.0,              # PhysX contact-report threshold
            "arm_prefixes":   ["/MobileRobot"],
            "carton_prefixes":["/RestockScenario/HandlingUnit"],
            "cart_prefixes":  ["/Rollwagen"]
        }

    A bare ``true`` uses all defaults. ``apply_contact_report`` / ``is_rigid_body`` are injection seams for
    the CPU ordering test; production leaves them ``None`` (the real ``PhysxSchema`` / ``UsdPhysics`` calls)."""
    logger = _build_logger_from_config(config, log=log)
    if logger is None:
        return None
    block = config.get(PLACE_CONTACT_LOGGER_KEY)
    opts = block if isinstance(block, dict) else {}
    n = logger.author_contact_reports(stage, apply_contact_report=apply_contact_report,
                                      is_rigid_body=is_rigid_body,
                                      threshold_m=float(opts.get('threshold_m', 0.0)), phase='pre-reset')
    log('PLACE_CONTACT_LOGGER armed on %d rigid bodies (arm/gripper/carton vs cart) phase=%s bodies=%d -- '
        'the contact-report API is authored BEFORE world.reset()/view creation (view-safe: arm DOF reads '
        'keep working); the place-descent PhysX contact pair is logged as %s a=.. b=.. depth=.. at first '
        'contact and on deepen, so the run NAMES the contact instead of leaving it to the model '
        '(sim-place-contact-logger-view-safe-v1)'
        % (n, logger._attach_phase, n, CONTACT_PAIR))
    return logger


def wire_place_contact_logger(assembly, config, *, log=print) -> Optional[PlaceContactLogger]:
    """POST-RESET half of the boot wiring (sim-place-contact-logger-view-safe-v1): SUBSCRIBE the logger that
    :func:`wire_place_contact_logger_pre_reset` already authored pre-reset. Called from
    ``isaac_factory.create`` with the live ``assembly``. Subscribing registers a Python callback on the
    omni.physx simulation interface -- it re-authors NO prim, so it is view-safe after ``world.reset()``.

    Returns the subscribed logger (so the caller can ``set_phase`` / ``detach``), or ``None`` when no logger
    was pre-authored (block absent => byte-identical, or the assembly did not run the pre-reset step). It
    NEVER authors prims here -- doing so after the view exists is exactly the coldwin101 crash."""
    logger = getattr(assembly, 'place_contact_logger', None)
    if logger is None:
        return None
    from omni.physx import get_physx_simulation_interface
    sim = get_physx_simulation_interface()
    logger.subscribe(sim)
    log('PLACE_CONTACT_LOGGER subscribed on %d bodies (attach phase=%s, view-safe post-reset) -- %s lines '
        'will print at the place descent (sim-place-contact-logger-view-safe-v1)'
        % (logger._armed_bodies, logger._attach_phase or 'pre-reset', CONTACT_PAIR))
    return logger
