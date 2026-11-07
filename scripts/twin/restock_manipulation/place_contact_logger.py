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
HOST/runtime block is present (:func:`wire_place_contact_logger`, called from ``isaac_factory.create``).
The PURE core (path classification + the per-pair first/deepen log decision + the line format) needs no
Isaac and is CPU-tested; only :meth:`PlaceContactLogger.attach` touches ``omni.physx`` / ``pxr`` (lazily,
with injectable hooks so the boot-path wiring is testable without a GPU).
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
    gripper-housing link, else ``'arm'``. Returns ``None`` for any contact not between the robot/carton and
    the cart (so a cart-vs-floor or arm-vs-arm contact is ignored). Pure/CPU."""
    paths = [str(p) for p in paths if p]
    cart = [p for p in paths if _starts_any(p, cart_prefixes)]
    carton = [p for p in paths if _starts_any(p, carton_prefixes)]
    arm = [p for p in paths if _starts_any(p, arm_prefixes) and p not in carton]
    if not cart:
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
    _seen: Dict[Tuple[str, str], float] = field(default_factory=dict)
    _subscription: object = None

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
        line = format_contact_line(a, b, separation_m, phase=self.phase, event=event, kind=kind)
        self.log(line)
        return line

    def handle_header_paths(self, paths, separation_m) -> Optional[str]:
        """Classify ``paths`` and, if a robot/carton-vs-cart contact, log it. Returns the line or ``None``.
        The single seam the Isaac callback and the CPU wiring test both drive. Pure (no Isaac)."""
        hit = self.classify(paths)
        if hit is None:
            return None
        a, b, kind = hit
        return self.on_contact(a, b, separation_m, kind=kind)

    # ----------------------------------------------------------------- Isaac attach (lazy; injectable hooks)
    def attach(self, stage, sim_interface, *, apply_contact_report=None, decode_path=None,
               threshold_m: float = 0.0):
        """Apply a PhysX contact-report API to every rigid body under the robot/carton/cart prefixes on
        ``stage`` and subscribe this logger's callback to ``sim_interface`` (``omni.physx``'s
        ``get_physx_simulation_interface()``). Returns the number of bodies instrumented.

        ``apply_contact_report(prim, threshold_m)`` and ``decode_path(int)`` default to the real
        ``PhysxSchema.PhysxContactReportAPI``/``PhysicsSchemaTools`` calls, but are INJECTABLE so the
        boot-path wiring is testable without Isaac. Idempotent-safe to call once at boot."""
        from pxr import UsdPhysics                                   # pxr only; usd-core has it
        if apply_contact_report is None:
            from pxr import PhysxSchema
            def apply_contact_report(prim, thr):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(float(thr))
        if decode_path is None:
            from pxr import PhysicsSchemaTools
            decode_path = lambda i: str(PhysicsSchemaTools.intToSdfPath(i))

        prefixes = tuple(self.arm_prefixes) + tuple(self.carton_prefixes) + tuple(self.cart_prefixes)
        n = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not _starts_any(path, prefixes):
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                apply_contact_report(prim, threshold_m)
                n += 1

        def _callback(headers, data):
            try:
                for header in headers:
                    paths = [decode_path(getattr(header, k))
                             for k in ('collider0', 'collider1', 'actor0', 'actor1')]
                    pts = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
                    sep = min((float(p.separation) for p in pts), default=0.0)
                    self.handle_header_paths(paths, sep)
            except Exception as exc:                                 # a logger must never crash the sim
                self.log('%s_ERROR %s' % (CONTACT_PAIR, exc))

        self._subscription = sim_interface.subscribe_contact_report_events(_callback)
        return n

    def detach(self):
        self._subscription = None


def wire_place_contact_logger(assembly, config, *, log=print) -> Optional[PlaceContactLogger]:
    """Boot-layer wiring (sim-pad-down-descent-jam-physics-v1): construct + attach a :class:`PlaceContactLogger`
    when the ``place_contact_logger`` HOST/runtime block is present, else return ``None`` (byte-identical --
    nothing is constructed, applied or subscribed). Called from ``isaac_factory.create`` with the live
    ``assembly`` (its ``.stage`` + ``omni.physx`` simulation interface).

    Block schema (all optional)::

        "place_contact_logger": true | {
            "deepen_step_m": 0.005,          # re-log a pair only when its penetration grows this much
            "threshold_m": 0.0,              # PhysX contact-report threshold
            "arm_prefixes":   ["/MobileRobot"],
            "carton_prefixes":["/RestockScenario/HandlingUnit"],
            "cart_prefixes":  ["/Rollwagen"]
        }

    A bare ``true`` uses all defaults. Returns the attached logger (so the caller can ``set_phase`` /
    ``detach``)."""
    block = config.get(PLACE_CONTACT_LOGGER_KEY)
    if not block:
        return None
    opts = block if isinstance(block, dict) else {}
    logger = PlaceContactLogger(
        arm_prefixes=tuple(opts.get('arm_prefixes', DEFAULT_ARM_PREFIXES)),
        carton_prefixes=tuple(opts.get('carton_prefixes', DEFAULT_CARTON_PREFIXES)),
        cart_prefixes=tuple(opts.get('cart_prefixes', DEFAULT_CART_PREFIXES)),
        deepen_step_m=float(opts.get('deepen_step_m', 0.005)), log=log)
    stage = getattr(assembly, 'stage', None)
    from omni.physx import get_physx_simulation_interface
    sim = get_physx_simulation_interface()
    n = logger.attach(stage, sim, threshold_m=float(opts.get('threshold_m', 0.0)))
    log('PLACE_CONTACT_LOGGER armed on %d rigid bodies (arm/gripper/carton vs cart) -- the place-descent '
        'PhysX contact pair is now logged as %s a=.. b=.. depth=.. at first contact and on deepen, so the '
        'run NAMES the contact instead of leaving it to the model (sim-pad-down-descent-jam-physics-v1)'
        % (n, CONTACT_PAIR))
    return logger
