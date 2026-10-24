"""Forward hybrid A* for the actual base, rear hitch and independently yawing cart.

The trailer model is a quasi-static no-side-slip centre-axle approximation, not
a PhysX substitute. Runtime starts every collision rollout from measured hitch
articulation. Slip/model error must stay within the configured tracking margin.
"""
from dataclasses import dataclass
import heapq
import math
import os
import time

# fm505: default-off controller-internal nav-plan diagnostics (every point where the coupled drive
# can end up route=None), so a LIVE return/outbound stall names its cause on the first bench run.
_NAV_PLAN_DIAG = os.environ.get('RESTOCK_NAV_PLAN_DIAG') == '1'

from twin.mobile import MobileSpec, Pose2, occupied_envelope, swept_envelope, wrap
from .core import Controller, Footprint, Result


@dataclass(frozen=True)
class TrainState:
    x: float
    y: float
    yaw: float
    cart_yaw: float

    @property
    def pose(self):
        return self.x, self.y, self.yaw

    @property
    def articulation(self):
        return wrap(self.yaw-self.cart_yaw)

    def bodies(self, spec):
        robot = Pose2(self.x, self.y, self.yaw)
        hx, hy = robot.point(spec.hitch_x)
        cart = Pose2(hx-spec.cart_hitch_x*math.cos(self.cart_yaw),
                     hy-spec.cart_hitch_x*math.sin(self.cart_yaw), self.cart_yaw)
        return robot, cart


def advance(state, distance, curvature, spec, locked=False):
    """Midpoint integration of an off-axle rear hitch trailer, in path length.

    fm455 (option c): with ``locked`` the drawbar hitch is rigidly PINNED (a real lockable hitch, see
    `MobileController`/`physics_setup`), so the cart yaws rigidly WITH the base -- cart_yaw tracks the new
    base_yaw exactly (articulation stays 0). The off-axle trailer integration is bypassed for that step;
    the rigid train is one body. ``locked=False`` (default) is the unchanged off-axle model, byte-identical."""
    half_yaw = state.yaw+curvature*distance/2
    new_yaw = wrap(state.yaw+curvature*distance)
    if locked:
        return TrainState(state.x+distance*math.cos(half_yaw),
                          state.y+distance*math.sin(half_yaw),
                          new_yaw, new_yaw)
    def cart_rate(yaw, phi):
        beta = yaw-phi
        return (math.sin(beta)+spec.hitch_x*curvature*math.cos(beta))/spec.cart_hitch_x
    half_phi = state.cart_yaw+cart_rate(state.yaw,state.cart_yaw)*distance/2
    return TrainState(state.x+distance*math.cos(half_yaw),
                      state.y+distance*math.sin(half_yaw),
                      new_yaw,
                      wrap(state.cart_yaw+cart_rate(half_yaw,half_phi)*distance))


def dubins_connections(start, goal, radius=1.5):
    """Four CSC forward Dubins families, used only as collision-checked shots.

    Hybrid search supplies additional turns; omitting CCC shots affects search
    efficiency/completeness under its budget, never collision acceptance.
    """
    mod=lambda a:a%(2*math.pi)
    dx,dy=goal[0]-start.x,goal[1]-start.y
    d=math.hypot(dx,dy)/radius
    direction=math.atan2(dy,dx)
    a,b=mod(start.yaw-direction),mod(goal[2]-direction)
    sa,sb,ca,cb=math.sin(a),math.sin(b),math.cos(a),math.cos(b)
    solutions=[]
    p2=2+d*d-2*math.cos(a-b)+2*d*(sa-sb)
    if p2>=0:
        t=math.atan2(cb-ca,d+sa-sb)
        solutions.append([(mod(-a+t)*radius,1/radius),(math.sqrt(p2)*radius,0),
                          (mod(b-t)*radius,1/radius)])
    p2=2+d*d-2*math.cos(a-b)+2*d*(sb-sa)
    if p2>=0:
        t=math.atan2(ca-cb,d-sa+sb)
        solutions.append([(mod(a-t)*radius,-1/radius),(math.sqrt(p2)*radius,0),
                          (mod(-b+t)*radius,-1/radius)])
    p2=-2+d*d+2*math.cos(a-b)+2*d*(sa+sb)
    if p2>=0:
        p=math.sqrt(p2)
        t=math.atan2(-ca-cb,d+sa+sb)-math.atan2(-2,p)
        solutions.append([(mod(-a+t)*radius,1/radius),(p*radius,0),
                          (mod(-b+t)*radius,-1/radius)])
    p2=d*d-2+2*math.cos(a-b)-2*d*(sa+sb)
    if p2>=0:
        p=math.sqrt(p2)
        t=math.atan2(ca+cb,d-sa-sb)-math.atan2(2,p)
        solutions.append([(mod(a-t)*radius,-1/radius),(p*radius,0),
                          (mod(b-t)*radius,1/radius)])
    return sorted(solutions,key=lambda s:sum(p[0] for p in s))


def polygon_hits_box(poly, box):
    """Separating-axis test, contact counts as occupied."""
    a,b,c,d = box
    corners = ((a,b),(a,d),(c,b),(c,d))
    axes = [(1,0),(0,1)]
    axes.extend((q[1]-p[1],p[0]-q[0]) for p,q in zip(poly,poly[1:]+poly[:1]))
    for nx,ny in axes:
        pp = [x*nx+y*ny for x,y in poly]
        qq = [x*nx+y*ny for x,y in corners]
        if max(pp)<min(qq) or max(qq)<min(pp):
            return False
    return True


class ArticulatedMap:
    """Spatially indexed static AABBs; polygon sweeps retain MobileSpec margins."""
    def __init__(self, grid, spec=None, margin=0.05):
        self.grid, self.spec = grid, spec or MobileSpec()
        if not math.isfinite(margin) or margin < 0.05:
            raise ValueError('mobile geometry requires at least the 0.05 m margin')
        self.margin = margin
        self.index = {}
        for i,box in enumerate(grid.obstacles):
            for cell in self.cells(box):
                self.index.setdefault(cell,[]).append(i)

    @staticmethod
    def cells(box):
        a,b,c,d = box
        return ((i,j) for i in range(math.floor(a),math.floor(c)+1)
                for j in range(math.floor(b),math.floor(d)+1))

    def clear_polygon(self, poly, dynamic=()):
        xs,ys = zip(*poly)
        box = min(xs),min(ys),max(xs),max(ys)
        a,b,c,d = self.grid.bounds
        if box[0]<=a or box[1]<=b or box[2]>=c or box[3]>=d:
            return False
        candidates = set(i for cell in self.cells(box) for i in self.index.get(cell,()))
        for i in candidates:
            r = self.grid.obstacles[i]
            if r[2]<box[0] or r[0]>box[2] or r[3]<box[1] or r[1]>box[3]:
                continue
            if polygon_hits_box(poly,r):
                return False
        for x,y in dynamic:
            if box[0]<=x<=box[2] and box[1]<=y<=box[3] and polygon_hits_box(poly,(x,y,x,y)):
                return False
        return True

    def valid(self,state,dynamic=(),extra_margin=0.0):
        if abs(state.articulation)>=self.spec.max_articulation:
            return False
        return self.clear_polygon(occupied_envelope(*state.bodies(self.spec),self.spec,
                                  margin=self.margin+extra_margin),dynamic)

    def rollout(self,state,distance,curvature,dynamic=(),extra_margin=0.0,step=0.08,locked=False):
        """Check each local swept hull, never one convex hull of the whole turn.

        fm455 (option c): ``locked`` rolls out the RIGID train (cart pinned to the base, articulation 0)
        so the swept hull is the single-rigid-body footprint rotating through the arc, not the trailing
        off-axle cart. ``locked=False`` (default) is the unchanged off-axle rollout."""
        if distance<0 or abs(curvature)>1/1.5+1e-9:
            return []
        count=max(1,math.ceil(distance/step))
        ds=distance/count
        samples=[state]
        for _ in range(count):
            nxt=advance(samples[-1],ds,curvature,self.spec,locked)
            if abs(nxt.articulation)>=self.spec.max_articulation:
                return []
            # Centre-trajectory chord error plus RK2 integration allowance.
            pad=ds*ds*(abs(curvature)+1/self.spec.cart_hitch_x)/8+0.001
            hull=swept_envelope([samples[-1].bodies(self.spec),nxt.bodies(self.spec)],
                               self.spec,margin=self.margin+extra_margin+pad)
            if not self.clear_polygon(hull,dynamic):
                return []
            samples.append(nxt)
        return samples

    def near_static(self,x,y,tol):
        """True iff the point (x,y) lies within `tol` of any static-map obstacle AABB.

        A measured lidar/ESDF return sitting a few cm PROUD of known static geometry (the
        documented couple-corridor effect: live returns register a few cm toward free space of
        the true AABB) is classified here as a known-static return, NOT a new obstacle. Uses the
        same integer-cell index as clear_polygon; `tol` must be < 1 cell (one grid unit), which
        the caller (the tow-lane proud tolerance) enforces. Only returns within `tol` of static
        geometry are matched -- a genuinely new obstacle in open space is never near_static."""
        cx,cy=math.floor(x),math.floor(y)
        seen=set()
        for i in range(cx-1,cx+2):
            for j in range(cy-1,cy+2):
                for oi in self.index.get((i,j),()):
                    if oi in seen:
                        continue
                    seen.add(oi)
                    xmin,ymin,xmax,ymax=self.grid.obstacles[oi]
                    dx=max(xmin-x,0.0,x-xmax); dy=max(ymin-y,0.0,y-ymax)
                    if math.hypot(dx,dy)<=tol:
                        return True
        return False

    def nearest_static(self,x,y,max_cells=4):
        """Distance (m) from (x,y) to the nearest static-map obstacle AABB, searching cells within
        +-max_cells. DIAGNOSTIC ONLY (LANE_BLOCKED_DIAG classification): a small distance means the
        return is proud-of-mapped static, a distance at/over the search cap means it sits off finer
        sim geometry than the AABB map carries (or genuinely open space). Capped so the per-tick diag
        stays cheap; returns the cap when nothing is indexed within it."""
        cx,cy=math.floor(x),math.floor(y)
        best=float(max_cells)
        seen=set()
        for i in range(cx-max_cells,cx+max_cells+1):
            for j in range(cy-max_cells,cy+max_cells+1):
                for oi in self.index.get((i,j),()):
                    if oi in seen:
                        continue
                    seen.add(oi)
                    xmin,ymin,xmax,ymax=self.grid.obstacles[oi]
                    dx=max(xmin-x,0.0,x-xmax); dy=max(ymin-y,0.0,y-ymax)
                    d=math.hypot(dx,dy)
                    if d<best:
                        best=d
        return best


@dataclass
class Route:
    states: list
    curvatures: list
    reason: str
    expanded: int = 0
    # fm455 (option c): per-step rigid-hitch-lock flags (parallel to `curvatures`; locked[i] drives
    # states[i]->states[i+1]). Empty () => no step is locked => off-axle trailer everywhere, byte-identical.
    # Keyword-only in practice (every positional Route(...) passes at most `expanded`).
    locked: tuple = ()


class ArticulatedPlanner:
    """Bounded hybrid A*: (x,y,base yaw,cart articulation), forward arcs only."""
    def __init__(self, world, *, xy_resolution=0.20, yaw_bins=72,
                 primitive_length=0.4, max_expansions=60000, timeout=30.0,
                 min_turn_radius=1.5):
        self.world=world
        self.xy_resolution=xy_resolution
        self.yaw_step=2*math.pi/yaw_bins
        self.length=primitive_length
        self.max_expansions,self.timeout=max_expansions,timeout
        # fm447 (a): the planned min turn radius for the hybrid-A* arc successors + the Dubins shots.
        # 1.5 m (default) = the coupled physical minimum (byte-identical). A LARGER value plans GENTLER
        # turns so a coupled base that over-rotates the physical-min turn by ~24% (run-330b) still executes
        # an arc >= the physical min -- IF the aisle/portal can fit the wider turn (proven in check_tow_lanes).
        self.min_turn_radius=float(min_turn_radius)
        self.seeds=[]

    def key(self,s):
        return (round(s.x/self.xy_resolution),round(s.y/self.xy_resolution),
                round(wrap(s.yaw)/self.yaw_step),round(s.articulation/self.yaw_step))

    def validate_seed(self, seed, start, goal, dynamic=(), extra_margin=0.02,
                      goal_articulation=None):
        """Replay controls from measured start; reject stale/colliding/invalid seeds.

        A cached route is a planning hint, never a replacement for feedback. Its
        world poses are not sent to actuators. Every replayed arc is collision
        checked against the current static map and current obstacle observations.
        """
        limit=self.world.spec.max_articulation if goal_articulation is None else goal_articulation
        if not seed.states or len(seed.curvatures)!=len(seed.states)-1:
            return Route([],[],'invalid_seed')
        initial=seed.states[0]
        if (math.dist(initial.pose[:2],start.pose[:2])>0.10
                or abs(wrap(initial.yaw-start.yaw))>0.03
                or abs(wrap(initial.cart_yaw-start.cart_yaw))>0.10):
            return Route([],[],'seed_start_mismatch')
        if not self.world.valid(start,dynamic,extra_margin):
            return Route([],[],'invalid_start')
        states=[start]; curvatures=[]; out_locked=[]
        for i,(a,b,k) in enumerate(zip(seed.states,seed.states[1:],seed.curvatures)):
            lk=bool(seed.locked[i]) if seed.locked else False
            ds=math.dist(a.pose[:2],b.pose[:2])
            if not math.isfinite(k) or not 0<ds<=0.10:
                return Route([],[],'invalid_seed_primitive')
            expected=advance(a,ds,k,self.world.spec,lk)
            if (math.dist(expected.pose[:2],b.pose[:2])>1e-5
                    or abs(wrap(expected.yaw-b.yaw))>1e-5
                    or abs(wrap(expected.cart_yaw-b.cart_yaw))>1e-5):
                return Route([],[],'seed_kinematics_mismatch')
            samples=self.world.rollout(states[-1],ds,k,dynamic,extra_margin,step=0.10,locked=lk)
            if not samples:
                return Route([],[],'seed_collision')
            states.extend(samples[1:]); curvatures.extend([k]*(len(samples)-1))
            out_locked.extend([lk]*(len(samples)-1))
        end=states[-1]
        if (math.dist(end.pose[:2],goal[:2])>=0.07 or abs(wrap(end.yaw-goal[2]))>=0.07
                or abs(end.articulation)>=limit):
            return Route([],[],'seed_goal_mismatch')
        return Route(states,curvatures,'validated_seed',locked=tuple(out_locked) if seed.locked else ())

    def plan(self,start,goal,dynamic=(),extra_margin=0.02,goal_articulation=None):
        """Try verified cache, an independently checked proposal, then full search.

        The proposal search keeps the mandatory body margin but omits localization
        inflation only while finding a candidate. It cannot return that candidate
        until validate_seed replays every segment at the requested full allowance.
        This changes search order without changing trajectory acceptance criteria.
        """
        if len(goal)!=3 or not all(math.isfinite(v) for v in goal):
            raise ValueError('goal must be finite [x,y,chassis_yaw]')
        if not self.world.valid(start,dynamic,extra_margin):
            return Route([],[],'invalid_start')
        for seed in self.seeds:
            checked=self.validate_seed(seed,start,goal,dynamic,extra_margin,goal_articulation)
            if checked.states:
                return checked
        corridor=self._corridor_route(start,goal,dynamic,extra_margin,goal_articulation)
        if corridor.states:
            return corridor
        if extra_margin>=0.02:
            proposer=ArticulatedPlanner(self.world,xy_resolution=self.xy_resolution,
                yaw_bins=round(2*math.pi/self.yaw_step),primitive_length=self.length,
                max_expansions=self.max_expansions,timeout=min(5.0,self.timeout/4))
            candidate=proposer._search(start,goal,dynamic,0.0,goal_articulation)
            if candidate.states:
                checked=self.validate_seed(candidate,start,goal,dynamic,extra_margin,goal_articulation)
                if checked.states:
                    checked.reason='validated_proposal'
                    checked.expanded=candidate.expanded
                    return checked
        return self._search(start,goal,dynamic,extra_margin,goal_articulation)

    def _corridor_route(self,start,goal,dynamic,extra_margin,goal_articulation):
        """Try map passage alignment before expensive unconstrained expansion.

        Hints never declare free space: all Dubins arcs, the connecting straight,
        and independent trailer states are swept at the requested full margin.
        A blocked hint falls back to search, not relaxed collision acceptance.
        """
        deadline=time.monotonic()+min(5.0,self.timeout/4)
        def extend(state,shot):
            states=[];curves=[]
            for length,k in shot:
                if length<1e-6:
                    continue
                samples=self.world.rollout(state,length,k,dynamic,extra_margin)
                if not samples:
                    return None
                states.extend(samples[1:]);curves.extend([k]*(len(samples)-1))
                state=samples[-1]
            return states,curves
        for x,y,heading in self.world.grid.corridors:
            axis=(math.cos(heading),math.sin(heading))
            a=(start.x-x)*axis[0]+(start.y-y)*axis[1]
            b=(goal[0]-x)*axis[0]+(goal[1]-y)*axis[1]
            if a*b>=0:
                continue
            if b<0:
                heading=wrap(heading+math.pi)
            dx,dy=math.cos(heading),math.sin(heading)
            for approach in (1.5,2.0,2.5,3.0):
                entry=(x-approach*dx,y-approach*dy,heading)
                for shot in dubins_connections(start,entry,self.min_turn_radius):
                    if time.monotonic()>deadline:
                        return Route([],[],'corridor_budget')
                    first=extend(start,shot)
                    if first is None or not first[0]:
                        continue
                    for departure in (.5,.625,.75,1.0,1.5,2.0,2.5,3.0):
                        middle=extend(first[0][-1],[(approach+departure,0.0)])
                        if middle is None:
                            continue
                        for final in dubins_connections(middle[0][-1],goal,self.min_turn_radius):
                            if time.monotonic()>deadline:
                                return Route([],[],'corridor_budget')
                            last=extend(middle[0][-1],final)
                            if last is None:
                                continue
                            candidate=Route([start]+first[0]+middle[0]+last[0],
                                            first[1]+middle[1]+last[1],'corridor_hint')
                            checked=self.validate_seed(candidate,start,goal,dynamic,
                                                       extra_margin,goal_articulation)
                            if checked.states:
                                checked.reason='validated_corridor'
                                return checked
        return Route([],[],'no_corridor_route')

    def _search(self,start,goal,dynamic=(),extra_margin=0.02,goal_articulation=None):
        goal_articulation=self.world.spec.max_articulation if goal_articulation is None else goal_articulation
        if len(goal)!=3 or not all(math.isfinite(v) for v in goal):
            raise ValueError('goal must be finite [x,y,chassis_yaw]')
        if not self.world.valid(start,dynamic,extra_margin):
            return Route([],[],'invalid_start')
        for seed in self.seeds:
            checked=self.validate_seed(seed,start,goal,dynamic,extra_margin,goal_articulation)
            if checked.states:
                return checked
        # Goal cart yaw is a state to solve for, not assumed attached collinearly.
        def heuristic(s):
            return max(math.hypot(s.x-goal[0],s.y-goal[1]),
                       1.5*abs(wrap(s.yaw-goal[2])))
        nodes=[(start,None,[],0.0)]
        queue=[(heuristic(start),0)]
        best={self.key(start):0.0}
        deadline=time.monotonic()+self.timeout
        expanded=0
        while queue and expanded<self.max_expansions:
            if time.monotonic()>deadline:
                return Route([],[],'planning_timeout',expanded)
            _,idx=heapq.heappop(queue)
            state,parent,edge,cost=nodes[idx]
            if cost>best.get(self.key(state),math.inf)+1e-8:
                continue
            expanded+=1
            if expanded==1 or expanded%5==0:
                shots=[]
                # A final straight pull aligns the cart for parking/return docking.
                for lead in (0.0,1.6,2.5,4.0):
                    approach=(goal[0]-lead*math.cos(goal[2]),
                              goal[1]-lead*math.sin(goal[2]),goal[2])
                    shots.extend(s+([(lead,0.0)] if lead else [])
                                 for s in dubins_connections(state,approach,self.min_turn_radius))
                for shot in sorted(shots,key=lambda s:sum(p[0] for p in s)):
                    shot_state=state
                    shot_edge=[]
                    for length,k in shot:
                        if length<1e-6:
                            continue
                        samples=self.world.rollout(shot_state,length,k,dynamic,extra_margin)
                        if not samples:
                            break
                        shot_edge.extend((p,k) for p in samples[1:])
                        shot_state=samples[-1]
                    else:
                        if (math.hypot(shot_state.x-goal[0],shot_state.y-goal[1])<0.07
                                and abs(wrap(shot_state.yaw-goal[2]))<0.07
                                and abs(shot_state.articulation)<goal_articulation):
                            nodes.append((shot_state,idx,shot_edge,cost+sum(p[0] for p in shot)))
                            idx=len(nodes)-1
                            state=shot_state
                            break
            if (math.hypot(state.x-goal[0],state.y-goal[1])<0.07
                    and abs(wrap(state.yaw-goal[2]))<0.07
                    and abs(state.articulation)<goal_articulation):
                chunks=[]
                while nodes[idx][1] is not None:
                    chunks.append(nodes[idx][2]); idx=nodes[idx][1]
                states=[start]; curves=[]
                for chunk in reversed(chunks):
                    for sample,k in chunk:
                        states.append(sample); curves.append(k)
                return Route(states,curves,'planned',expanded)
            # Direct final straight primitive preserves precise portal/aisle alignment.
            distance=math.hypot(goal[0]-state.x,goal[1]-state.y)
            heading=math.atan2(goal[1]-state.y,goal[0]-state.x)
            _R=self.min_turn_radius
            options=[(self.length,k) for k in (0.0,-1/_R,1/_R,-1/(2*_R),1/(2*_R))]
            if distance<4 and abs(wrap(heading-state.yaw))<0.03:
                options.insert(0,(distance,0.0))
            for length,k in options:
                samples=self.world.rollout(state,length,k,dynamic,extra_margin)
                if not samples:
                    continue
                nxt=samples[-1]; key=self.key(nxt)
                new_cost=cost+length*(1+0.10*abs(k))
                if new_cost>=best.get(key,math.inf):
                    continue
                best[key]=new_cost
                nodes.append((nxt,idx,[(p,k) for p in samples[1:]],new_cost))
                # Weighted search keeps CPU bring-up bounded; not optimal-path evidence.
                heapq.heappush(queue,(new_cost+1.5*heuristic(nxt),len(nodes)-1))
        return Route([],[],'no_route' if not queue else 'expansion_limit',expanded)


class ArticulatedController(Controller):
    """Measured-state route following and braking sweep for the physical tow rig."""
    def __init__(self, grid, spec=None, *, require_temporal_map=False, planner_timeout=None,
                 lane_graph=None, lane_static_tol=0.0, lane_trust_static=False,
                 lane_reacquire_along_route=False, lane_cart_yaw_tol=0.20,
                 lane_curve_speed_floor=0.25, lane_max_curvature=None,
                 lane_coast_refusals=False, lane_trust_unknown=False,
                 measured_turn_fault_s=None, lane_track_tol=0.25,
                 lane_track_envelope=False, lane_cart_yaw_rebase=False,
                 coupled_creep_escape=False, coupled_creep_floor=0.03,
                 coupled_progress_escape=False, coupled_progress_min_m=0.05,
                 coupled_progress_escape_s=3.0, coupled_progress_fault_s=8.0,
                 coupled_progress_recovery=False, coupled_breakaway_speed_mps=0.18,
                 coupled_breakaway_window_s=2.0, coupled_breakaway_pulse_s=0.5,
                 coupled_breakaway_attempts=3,
                 coupled_breakaway_rock=False, coupled_breakaway_launch_mps=0.30,
                 lane_acquire_timeout_s=0.0, tow_lane_goal_stop=False,
                 coupled_heading_hold=False, heading_hold_bound_deg=1.0,
                 heading_hold_creep_mps=0.14, heading_hold_straight_kappa=0.08,
                 heading_hold_prethroat_m=0.0, heading_hold_throat=None,
                 heading_hold_align_timeout_s=20.0, heading_hold_axis_deg=None, **kwargs):
        self.require_temporal_map=require_temporal_map
        # fm481/fm482 (navigation.coupled_creep_escape, default off => byte-identical): the coupled
        # measured-curvature brake divides odom.angular by odom.linear, and at a near-stationary crawl that
        # ratio is velocity NOISE, not a real sub-R_min arc. run-352/355 outbound: the tow stopped abruptly
        # ~0.4-0.5 m in (0.23 -> 0.03 m/s in one tick), then |rv|~|rw|~0.01-0.06 jitter tripped
        # measured_turn_radius on most ticks; each refusal commands a stop, so the base never re-left the
        # noise band (121 s livelock -> nav-stuck fail). With the escape ON, inside the band
        # |odom.linear| < max(0.06, 2*floor) a FORWARD command keeps a creep (v >= coupled_creep_floor) with
        # the arc clamped to the min turn radius, and the commanded creep is still footprint- and
        # braking-rollout-checked, so the base accelerates OUT of the band. Above the band the brake is
        # unchanged (a genuine sub-R_min arc at tow speed still refuses).
        self.coupled_creep_escape=bool(coupled_creep_escape)
        if not (isinstance(coupled_creep_floor,(int,float)) and math.isfinite(coupled_creep_floor)
                and 0.0<coupled_creep_floor<=0.1):
            raise ValueError('coupled_creep_floor must be in (0, 0.1] m/s')
        self.coupled_creep_floor=float(coupled_creep_floor)
        self.creep_escapes=0                                # ticks the escape replaced a noise refusal
        # fm-sim-robust-outbound (navigation.coupled_progress_escape, default off => byte-identical): a
        # SIM-CLOCK commanded-vs-measured NET-progress monitor for the coupled tow. warm9/warm10 outbound
        # pinned at x~=-3.9 (y=15.25) with the follower still COMMANDING forward (reason='tracking' on most
        # ticks, only a few 'measured_turn_radius'), the base jittering in place (measured |rv| noise to
        # ~0.23 m/s but ~0 NET displacement), under ~20x render pacing -- so the base made no net progress
        # for ~14 s SIM but ~285 s WALL, and the 120 s WALL nav-stuck budget (restock.runtime) failed the
        # stage (3x) only after burning GPU wall. coupled_creep_escape (fm482) missed it: its trigger needs
        # |odom.linear| INSIDE a narrow noise band, which the jitter spikes leave, and measured_turn_fault_s
        # (fm446) only accumulates on measured_turn_radius COASTS, not on the dominant 'tracking' ticks.
        # This monitor keys on NET base displacement over SIM time instead of instantaneous velocity: while
        # the coupled base commands forward but has not moved >= coupled_progress_min_m from its anchor for
        # coupled_progress_escape_s SIM seconds, FORCE the committed creep (the fm482 forced-floor path --
        # keep a clean forward command, never thrash to a measured-curvature STOP); if still no net progress
        # after coupled_progress_fault_s SIM seconds, FAULT 'coupled_no_progress' terminal so the stage
        # StageRetry-replans in a few SIM seconds -- before (and on the correct clock for) the WALL budget.
        self.coupled_progress_escape=bool(coupled_progress_escape)
        for _nm,_v,_hi in (('coupled_progress_min_m',coupled_progress_min_m,1.0),
                           ('coupled_progress_escape_s',coupled_progress_escape_s,120.0),
                           ('coupled_progress_fault_s',coupled_progress_fault_s,120.0)):
            if not (isinstance(_v,(int,float)) and not isinstance(_v,bool)
                    and math.isfinite(_v) and 0.0<float(_v)<=_hi):
                raise ValueError(f'{_nm} must be a finite number in (0, {_hi:g}]')
        if float(coupled_progress_escape_s)>float(coupled_progress_fault_s):
            raise ValueError('coupled_progress_escape_s must be <= coupled_progress_fault_s')
        self.coupled_progress_min_m=float(coupled_progress_min_m)
        self.coupled_progress_escape_s=float(coupled_progress_escape_s)
        self.coupled_progress_fault_s=float(coupled_progress_fault_s)
        self._prog_anchor=None        # (sim_now, x, y) of the last confirmed base progress (window start)
        self._prog_last_now=None      # sim time of the last monitor tick (contiguity/gap check)
        self._prog_stuck_s=0.0        # SIM seconds the base has sat below coupled_progress_min_m
        self._force_progress_creep=False   # set per-tick: force the committed creep this gate call
        self.progress_escapes=0       # ticks the no-progress monitor forced a creep
        # fm-sim-robust-outbound-stall-recovery (navigation.coupled_progress_recovery, default off =>
        # byte-identical to the fm-sim-robust-outbound escape->fault ladder above): a BOUNDED break-away
        # RECOVERY maneuver that runs between the no-progress ESCALATION and the terminal FAULT, so a
        # coupled tow that fails to break away from rest at the portal throat (coldlivefull outbound:
        # NAV_STEP cycled measured_turn_radius<->tracking with parked=True, the base jittering in place
        # with ~0 NET displacement) gets REAL re-tow attempts instead of the single weak 0.03 m/s creep +
        # immediate fail. Root cause it addresses (our own code): a measured_turn_radius COAST tick returns
        # _wait(...) == velocity (0,0), so the stuck tow thrashes stop<->forward every few ticks and never
        # sustains drive effort to overcome the coupled static resistance from rest; the forced creep
        # (fm-sim-robust-outbound) only engages after escape_s and is a static-crawl 0.03 m/s floor, too
        # weak for a re-break-away under the live_pose load. The recovery holds a CONTINUOUS, STRONGER,
        # PULSED committed-forward command (coupled_breakaway_speed_mps, square-wave surged every
        # coupled_breakaway_pulse_s to "rock" the contact) for up to coupled_breakaway_attempts windows of
        # coupled_breakaway_window_s SIM seconds each; the instant net displacement from a window's anchor
        # reaches coupled_progress_min_m the break-away SUCCEEDED (resume normal tracking); if every attempt
        # elapses with no net progress, FAULT 'coupled_no_progress' terminal (same name/StageRetry contract
        # as the escape-only path). NOTE on "back off a few cm": reverse towing is deliberately clamped off
        # for a coupled train (mobile.wheel_targets: coupled v>=0), so the recovery uses a forward
        # surge/relax PULSE (the implementable, safety-preserving break-away rock) rather than a reverse
        # back-off. All brakes are already force-released every drive tick in MobileController.command, so no
        # stale park/cart brake can hold the base -- the stall is the physics break-away itself.
        self.coupled_progress_recovery=bool(coupled_progress_recovery)
        if not (isinstance(coupled_breakaway_speed_mps,(int,float)) and not isinstance(coupled_breakaway_speed_mps,bool)
                and math.isfinite(coupled_breakaway_speed_mps) and 0.0<float(coupled_breakaway_speed_mps)<=1.0):
            raise ValueError('coupled_breakaway_speed_mps must be a finite number in (0, 1.0] m/s')
        for _nm,_v,_hi in (('coupled_breakaway_window_s',coupled_breakaway_window_s,60.0),
                           ('coupled_breakaway_pulse_s',coupled_breakaway_pulse_s,10.0)):
            if not (isinstance(_v,(int,float)) and not isinstance(_v,bool)
                    and math.isfinite(_v) and 0.0<float(_v)<=_hi):
                raise ValueError(f'{_nm} must be a finite number in (0, {_hi:g}]')
        if not (isinstance(coupled_breakaway_attempts,int) and not isinstance(coupled_breakaway_attempts,bool)
                and 1<=coupled_breakaway_attempts<=20):
            raise ValueError('coupled_breakaway_attempts must be an int in [1, 20]')
        self.coupled_breakaway_speed=float(coupled_breakaway_speed_mps)
        self.coupled_breakaway_window_s=float(coupled_breakaway_window_s)
        self.coupled_breakaway_pulse_s=float(coupled_breakaway_pulse_s)
        self.coupled_breakaway_attempts=int(coupled_breakaway_attempts)
        self._recovery_active=False       # inside a break-away recovery window sequence
        self._recovery_attempt=0          # 0-based index of the current break-away attempt window
        self._recovery_window_start=None  # sim time the current attempt window began
        self._recovery_anchor=None        # (x, y) base pose at the current window's start
        self._breakaway_cmd=None          # per-tick committed forward speed override for gate_velocity
        self._breakaway_turn=False        # per-tick: saturate the corrective turn (rock hi phase)
        self.breakaway_recoveries=0       # recovery sequences entered
        self.breakaway_successes=0        # recovery sequences that broke the base free
        # fm-sim-robust-coupled-breakaway-from-rest (navigation.coupled_breakaway_rock, default off =>
        # byte-identical): coldwin9 outbound pinned at the sales_backroom_portal throat (x=-3.93) with the
        # base yawed ~5 deg off the lane axis -- a base+cart envelope catches the throat lip and the base
        # stops dead while still COMMANDING forward (a physical wedge, not a gate refusal). PR 236's
        # forward-only break-away pulse (0.18 m/s) cannot free a forward wedge: nine surges over the three
        # BT re-begins netted zero advance, so each re-begin from rest repeats the same push-into-the-lip.
        # A coupled train cannot reverse (mobile.wheel_targets clamps v>=0 by a deliberate bring-up
        # interlock; gate_velocity rejects linear<0), so the rock uses the only authority it has -- a
        # STRONGER forward LAUNCH surge (coupled_breakaway_launch_mps, clamped to max_speed at the gate: more
        # momentum + drive force, and a higher coupled steering cap wmax=v/1.5) with the corrective turn
        # SATURATED to the full +-wmax toward the route heading on the hi pulse phase, to rotate the
        # yaw-wedged base back toward alignment and swivel the casters free; the lo phase relaxes to "rock"
        # the contact. It replaces ONLY what the break-away pulse commands -- the PR 201 net-progress monitor
        # and PR 236's bounded coupled_breakaway_attempts windows (then fault 'coupled_no_progress') are the
        # untouched backstop. Requires coupled_progress_recovery (fail loud, never a silent no-op).
        self.coupled_breakaway_rock=bool(coupled_breakaway_rock)
        if not (isinstance(coupled_breakaway_launch_mps,(int,float)) and not isinstance(coupled_breakaway_launch_mps,bool)
                and math.isfinite(coupled_breakaway_launch_mps) and 0.0<float(coupled_breakaway_launch_mps)<=1.0):
            raise ValueError('coupled_breakaway_launch_mps must be a finite number in (0, 1.0] m/s')
        self.coupled_breakaway_launch=float(coupled_breakaway_launch_mps)
        if self.coupled_breakaway_rock:
            if not self.coupled_progress_recovery:
                raise ValueError('coupled_breakaway_rock requires coupled_progress_recovery '
                                 '(the rock shapes the break-away recovery; without it it is a no-op)')
            if self.coupled_breakaway_launch<self.coupled_breakaway_speed:
                raise ValueError('coupled_breakaway_launch_mps must be >= coupled_breakaway_speed_mps '
                                 '(the launch surge is an ELEVATED break-away speed)')
        # fm-sim-robust-outbound-throat-yaw-align (navigation.coupled_heading_hold, default off =>
        # byte-identical): keep the coupled base ALIGNED to the lane axis during the slow from-rest creep so
        # the base+cart envelope never enters the sales_backroom_portal throat (x~=-3.875) yawed and wedges on
        # the lip. coldwin9/coldwin10 outbound drifted CCW off the east travel axis during the crawl right
        # after the couple->outbound plan-wait dip (coldwin9: exits the dip at -1.7 deg, creeps at ~0.11 m/s,
        # drifts monotonically to +4.9 deg at the throat; coldwin10: carries +1.2 deg into the throat at
        # +2.2 deg) and wedged (reached_goal=False); coldwin7/coldwin8/rock1 held within +-1.2 deg of east
        # (cross the throat at -0.2..+0.3 deg) and threaded. The follower's pure-pursuit law WANTS to correct
        # the drift, but the coupled steering cap wmax = v/1.5 collapses at the crawl (failing runs: mean cap
        # ~0.03 rad/s vs ~0.12 for the passing runs), so the correction is throttled to near zero. The throat
        # is INVISIBLE to the nav static map (footprint valid at every base yaw -8..+6 deg; slack 0.8 m), so
        # the wedge is a PhysX contact on door-frame geometry finer than the AABB map (the gate returns
        # 'tracking', never 'obstacle_stop') and a slack-triggered pre-throat guard is impossible -- the
        # trigger is the CCW yaw drift on the lane itself. When ON, the coupled follower, on a STRAIGHT
        # authored segment, once the base has drifted more than `heading_hold_bound` to the CCW/throat-lip
        # (wedge) side of the lane goal (travel) heading: (1) holds a fixed modest forward creep
        # `heading_hold_creep_mps` -- restoring a steering authority wmax = creep/1.5 that does NOT depend on
        # the (collapsed) tracking speed schedule AND capping the forward speed so a misaligned base slows and
        # corrects rather than accelerating into the throat; and (2) SATURATES the corrective turn toward the
        # axis (curvature = -1/min_turn_radius, the coupled cap, swept-checked by the unchanged gate). It is
        # ONE-SIDED (a base on the safe CW/south side is never force-turned -- that would push it toward the
        # lip) and FORWARD-only (v>0 always; the coupled-reverse interlock is honoured). Because the
        # correction rate (~5 deg/s at the creep) dominates the measured drift (~0.4 deg/s), the base reaches
        # the throat aligned and never pins, so PR 201's net-progress monitor / PR 236's bound / PR 250's rock
        # stay dormant as the untouched backstop. Off => the legacy follower, byte-identical.
        self.coupled_heading_hold=bool(coupled_heading_hold)
        if not (isinstance(heading_hold_bound_deg,(int,float)) and not isinstance(heading_hold_bound_deg,bool)
                and math.isfinite(heading_hold_bound_deg) and 0.0<float(heading_hold_bound_deg)<=45.0):
            raise ValueError('heading_hold_bound_deg must be a finite number of degrees in (0, 45]')
        if not (isinstance(heading_hold_creep_mps,(int,float)) and not isinstance(heading_hold_creep_mps,bool)
                and math.isfinite(heading_hold_creep_mps) and 0.0<float(heading_hold_creep_mps)<=1.0):
            raise ValueError('heading_hold_creep_mps must be a finite number in (0, 1.0] m/s')
        if not (isinstance(heading_hold_straight_kappa,(int,float)) and not isinstance(heading_hold_straight_kappa,bool)
                and math.isfinite(heading_hold_straight_kappa) and 0.0<float(heading_hold_straight_kappa)<=1.0):
            raise ValueError('heading_hold_straight_kappa must be a finite curvature in (0, 1] 1/m')
        self.heading_hold_bound=math.radians(float(heading_hold_bound_deg))
        self.heading_hold_creep=float(heading_hold_creep_mps)
        self.heading_hold_straight_kappa=float(heading_hold_straight_kappa)
        self.heading_holds=0                               # ticks the heading hold overrode the follower
        # fm-sim-robust-heading-hold-bound (navigation.heading_hold_prethroat_m, default 0.0 => the new
        # robust path is OFF and the legacy one-sided hold above is byte-identical): the PR 253 hold had
        # THREE gaps the coldwin17 wedge exposed (CPU-reproduced in data/sim-robust-heading-hold-bound-v1):
        #   (a) the drift is introduced on the CURVED authored merge lead-in (the lane bridges a ~0.25 m
        #       lateral seat offset), where the straight-segment guard (|curv|<=straight_kappa) DISABLES the
        #       hold; by the time the base reaches the straight segment it is already ~+5 deg yawed and ~0.2 m
        #       from the lip (coldwin17: the real hold first fired at x=-4.079, base_yaw +5.18 deg -- far too
        #       late). The drift is DRIVEN by the follower's lateral term 2*lateral (~+0.52 for the seat
        #       offset), which commands a CCW turn that INCREASES the yaw vs the travel axis (reproduced: the
        #       follower law commands CCW on ~90% of approach ticks, matching the recorded cw>0 on 100%).
        #   (b) the hold was ONE-SIDED (only the CCW/lip side), so a CW seat drift is never corrected.
        #   (c) it bounds vs the lane goal (travel) axis but never GATES the throat entry, so a base that
        #       fails to align simply wedges on the lip with no named failure.
        # When heading_hold_prethroat_m > 0 (and a throat point is configured) the hold becomes a two-sided
        # PRE-THROAT ALIGNMENT GATE scoped to the window within that distance of the throat along the travel
        # axis: it (1) CAPS the follower's lateral/heading correction so it can never command a turn that
        # drives the yaw FURTHER off the travel axis (the cap works on the curved lead-in too -- it keeps the
        # authored turn, only removing the off-axis overshoot), (2) slows to the creep (restoring the
        # speed-schedule-independent authority wmax=creep/1.5), (3) on a straight authored segment SATURATES
        # the corrective turn toward the axis on EITHER side, and (4) if the base cannot reach within the
        # bound inside heading_hold_align_timeout_s SIM seconds in the window, FAILS by the named reason
        # 'throat_misaligned' (bounded, never a silent wedge). Reference axis = heading_hold_axis_deg if set
        # else the lane goal heading (the portal-crossing/travel direction the passing runs held; the
        # physically min-clearance heading through the x-gap -- NOT the descending centreline tangent, which
        # would widen the envelope through the opening). Forward-only (v>=0); PR 201/236/250 stay the
        # untouched backstop. See docs/restock_navigation.md "Pre-throat heading alignment".
        for _nm,_v in (('heading_hold_prethroat_m',heading_hold_prethroat_m),
                       ('heading_hold_align_timeout_s',heading_hold_align_timeout_s)):
            if not (isinstance(_v,(int,float)) and not isinstance(_v,bool)
                    and math.isfinite(_v) and 0.0<=float(_v)<=120.0):
                raise ValueError('%s must be a finite number in [0, 120]'%_nm)
        self.heading_hold_prethroat_m=float(heading_hold_prethroat_m)
        self.heading_hold_align_timeout_s=float(heading_hold_align_timeout_s)
        if heading_hold_throat is None:
            self.heading_hold_throat=None
        else:
            try:
                tx,ty=heading_hold_throat
                tx,ty=float(tx),float(ty)
            except (TypeError,ValueError):
                raise ValueError('heading_hold_throat must be None or an (x, y) pair')
            if not (math.isfinite(tx) and math.isfinite(ty)):
                raise ValueError('heading_hold_throat (x, y) must be finite')
            self.heading_hold_throat=(tx,ty)
        if heading_hold_axis_deg is None:
            self.heading_hold_axis=None
        else:
            if not (isinstance(heading_hold_axis_deg,(int,float)) and not isinstance(heading_hold_axis_deg,bool)
                    and math.isfinite(heading_hold_axis_deg) and -180.0<=float(heading_hold_axis_deg)<=180.0):
                raise ValueError('heading_hold_axis_deg must be None or a finite number of degrees in [-180, 180]')
            self.heading_hold_axis=math.radians(float(heading_hold_axis_deg))
        if self.heading_hold_prethroat_m>0.0 and self.heading_hold_throat is None:
            raise ValueError('heading_hold_prethroat_m>0 requires heading_hold_throat (the throat (x, y))')
        if self.heading_hold_prethroat_m>0.0 and not self.coupled_heading_hold:
            raise ValueError('heading_hold_prethroat_m>0 requires coupled_heading_hold (it shapes the hold)')
        self._align_anchor=None            # SIM time the base entered the window still misaligned (for the timeout)
        # fm437: on a static-validated authored lane, trust UNOBSERVED (unknown) temporal-map cells in the
        # swept train envelope as free (the static map proved that corridor clear), so the base is not
        # frozen at an unmapped turn (run-327b: outbound drove 4.87 m then 'unknown_navigation_cell' at the
        # first southward turn south of the dock corridor). A positive (occupied) return still stops the tow.
        self.lane_trust_unknown=bool(lane_trust_unknown)
        self.spec=spec or MobileSpec()
        self.world=ArticulatedMap(grid,self.spec)
        # Opt-in lidar-proud tolerance for the AUTHORED tow lanes only (navigation.tow_lane_static_tol_m).
        # 0.0 (default) => OFF: a lane is re-validated against the full live obstacle cloud exactly as
        # before (byte-identical). When >0, a live return within this distance of KNOWN static-map
        # geometry is discounted as a proud return of that geometry when re-validating a lane's swept
        # envelope -- so the authored centreline that clears the STATIC map (verified at build) is not
        # falsely blocked by returns sitting a few cm proud of a shelf/support it hugs (the run-321b
        # outbound lane_blocked). This NEVER touches the live braking gate (gate_velocity stays fully
        # live against obs.obstacles), and a genuinely NEW obstacle -- not within the tolerance of any
        # static geometry -- is kept, so it still blocks the lane at plan time AND the forward
        # directional braking zone still stops the tow for it. Mirrors couple_static_drive's
        # config-driven, default-off trusted-static style but scoped to lane plan-validation alone.
        if not (isinstance(lane_static_tol,(int,float)) and math.isfinite(lane_static_tol)
                and 0.0<=lane_static_tol<1.0):
            raise ValueError('lane_static_tol must be a finite distance in [0, 1) metres')
        self.lane_static_tol=float(lane_static_tol)
        # Opt-in trusted-static lane plan-validation (navigation.tow_lane_trust_static), default off =>
        # byte-identical. The authored tow lane is expanded AND swept-validated clear of the static map at
        # build; when ON, `_plan_lane` re-validates it against the static map ALONE (dyn=()), so proud OR
        # unmapped-in-the-planning-grid static returns in the LIVE cloud cannot false-block a route that is
        # already proven clear of the static map (run-322b: with no dynamic-obstacle source every live
        # return is static, yet 128 returns sat >0.12 m from any mapped AABB -- proud-beyond-tol or off
        # finer sim geometry than the AABB map -- so the narrower `lane_static_tol` could not reach them,
        # and the 797-AABB map makes raising that tol unsafe). This is the couple_static_drive trusted-
        # static pattern scoped to lane plan-validation ONLY: it NEVER sets self.static_drive, so the live
        # braking gate (gate_velocity) keeps the full live obstacle cloud and a genuinely new obstacle on
        # the lane STILL stops the tow -- the plan trusts the fixed, pre-validated geometry while the live
        # direction-aware gate owns new-obstacle avoidance. `lane_static_tol` still applies underneath for
        # the narrower proud-of-mapped case; trust_static is the superset.
        self.lane_trust_static=bool(lane_trust_static)
        # Opt-in re-acquire-along-the-route (navigation.tow_lane_reacquire_along_route), default off =>
        # byte-identical (start-only acquire). run-323b: trust_static let the lane validate, the follower
        # drove ~0.36 m along it, dropped the route on a tracking transient, then could NOT re-acquire
        # because `acquirable` only checked the fixed START (0.36 m away > start_tol_xy) though the train
        # was 0.06 m from the route -> lane_not_acquired forever, train frozen. When set, `_plan_lane`
        # checks acquirability against the NEAREST route point (same tolerances), so a follower that has
        # drifted a little along its own lane re-acquires at its progress instead of being trapped.
        self.lane_reacquire_along_route=bool(lane_reacquire_along_route)
        # fm429 (run-324b/c): curvature-aware coupled-tow tracking knobs for the authored lane follower,
        # all defaulting to the prior hardcoded values => byte-identical. run-324b/c measured: the cart
        # yaw-RATE tracks the base well (|rw-cw| max 0.097, mean 0.015 rad/s), but at the curve-slowed
        # crawl (rv mean 0.033 m/s) the MEASURED arc runs far tighter than the 1/1.5 build cap and the
        # tow makes ~0 net progress, tripping stop_replan_tracking_error on the cart_yaw offset. These
        # let the owner (a) relax the cart_yaw replan tolerance to the measured lag WITHOUT masking real
        # divergence (the swept envelope is re-proven on the lagged cart pose offline), (b) tune the
        # curve speed floor (slower on curves so the cart stays inside the planned articulation), and (c)
        # cap the follower's COMMANDED curvature below the physical min-turn-radius so corrections are
        # gentler (the authored route itself stays build-validated at the physical cap). lane_max_curvature
        # None => the physical cap 1/min_turn_radius (unchanged).
        phys_kappa = 1.0 / 1.5   # 1.5 m = the coupled min turn radius (the Footprint value below)
        if not (isinstance(lane_cart_yaw_tol,(int,float)) and math.isfinite(lane_cart_yaw_tol)
                and 0.0 < lane_cart_yaw_tol <= self.spec.max_articulation):
            raise ValueError('lane_cart_yaw_tol must be in (0, max_articulation] radians')
        if not (isinstance(lane_curve_speed_floor,(int,float)) and math.isfinite(lane_curve_speed_floor)
                and 0.0 < lane_curve_speed_floor <= 1.0):
            raise ValueError('lane_curve_speed_floor must be a fraction in (0, 1]')
        if lane_max_curvature is not None and not (isinstance(lane_max_curvature,(int,float))
                and math.isfinite(lane_max_curvature) and 0.0 < lane_max_curvature <= phys_kappa+1e-9):
            raise ValueError('lane_max_curvature must be a positive curvature <= 1/min_turn_radius')
        self.lane_cart_yaw_tol=float(lane_cart_yaw_tol)
        self.lane_curve_speed_floor=float(lane_curve_speed_floor)
        self.lane_max_curvature=phys_kappa if lane_max_curvature is None else float(lane_max_curvature)
        # fm433 (A'-2) / run-326b: following an AUTHORED lane, a jittering live cloud flickers transient
        # gate refusals (insufficient_sensor_coverage, proud-return obstacle_stop) that DROP the validated
        # route; the next tick re-plans on the changed cloud (replan_changed_observations), and the
        # drop<->replan churn stalls the tow (run-326b: 58 replan + 23 measured_turn_radius while the base
        # only crept 1.09 m). measured_turn_radius already COASTS (keeps the route); when this is set, EVERY
        # non-tracking refusal on an active lane coasts too -- the route is KEPT and the base simply STOPS
        # this tick (velocity 0 via _wait), so a genuinely new obstacle still halts the tow and a PERSISTENT
        # refusal still fails via the blocked_timeout, but a transient cloud flicker no longer discards the
        # lane. Scoped to an active authored lane (its geometry is fixed + build-validated). Default False
        # => the legacy drop-and-replan (byte-identical).
        self.lane_coast_refusals=bool(lane_coast_refusals)
        # fm446 (C): a PERSISTENT measured_turn_radius is a drive/route FAULT, not an endless coast.
        # measured_turn_radius coasts (keeps the route) for transient over-rotation, but a sub-min-radius
        # authored turn the coupled base physically cannot make over-rotates EVERY tick (run-330b: 186
        # coasts, ~0 net progress) -- and the sparse tracking tick between coasts keeps resetting the 8 s
        # blocked_timeout, so the controller livelocks until the 120 s mission nav-stuck. This net-
        # accumulating window (increment dt on a measured_turn_radius coast, DECAY on a clean tracking tick)
        # faults 'measured_turn_radius_persistent' once the net stuck time exceeds measured_turn_fault_s, so
        # it fails fast instead of coasting. None/<=0 => off (byte-identical: the legacy coast + 8 s timeout).
        self.measured_turn_fault_s=(float(measured_turn_fault_s)
                                    if measured_turn_fault_s not in (None, 0, 0.0) else None)
        self._mtr_window=0.0
        # fm449: the stop_replan_tracking_error xy tolerance -- how far the base may be from the nearest
        # authored route point before the route is DROPPED and re-planned. 0.25 m (default) = legacy
        # (byte-identical). run-332: reacquire + W2 solved the over-rotation + lane_not_acquired, but the
        # coupled turn's tracking error exceeds 0.25 m so the base re-acquires then re-drops (583x churn).
        # A larger value lets the base RIDE the turn's tracking error instead of churning -- bounded by the
        # swept-envelope proof (check_tow_lanes check_track_tol_envelope): the relaxed tol must stay inside
        # the lateral offset whose robot+cart envelope is still clear of the static map (never mask a real
        # departure). Matches the cart_yaw_tol prove-then-relax pattern (fm429).
        self.lane_track_tol=float(lane_track_tol)
        # fm449: a SINGLE scalar track tol is wrong for this lane -- the swept-envelope lateral slack is
        # position-dependent (check_tow_lanes: ~0.80 m at the dock turn where run-332 churned, but only
        # 0.14 m at the shelf-end pinch far downstream). With this on, the effective stop_replan xy tol at
        # each route point is min(lane_track_tol, local_lateral_slack - 0.05), floored at 0.10 -- so the
        # base may ride the turn's tracking error (loosened toward lane_track_tol where the envelope is
        # wide) while staying TIGHT at a narrow pinch (never beyond the envelope). Off (default) => the flat
        # lane_track_tol everywhere (byte-identical). The per-state slack is memoised (computed once).
        self.lane_track_envelope=bool(lane_track_envelope)
        self._slack_cache={}
        # fm451 (run-333): the authored route's cart_yaw is the off-axle trailer integration from the
        # authored lane START (aligned there) -- so by the turn it carries a +0.11 rad off-axle swing. But
        # the train ENTERS the follow partway and ALIGNED (from the couple), having done no such swing, so
        # at the nearest route point the authored cart_yaw diverges ~0.30 rad from the live cart and
        # stop_replan drops the route BEFORE the base can drive the turn (chicken-and-egg). This re-bases the
        # expected cart_yaw: from the base's nearest route point, re-integrate the SAME off-axle kinematics
        # forward with cart_yaw there = the live cart_yaw -- so the expectation is what a correctly-tracking
        # ALIGNED-entry train actually does. The (x,y,base_yaw) centreline is unchanged; the re-based profile
        # is re-checked against the swept envelope and only used if clear (a genuinely departed cart still
        # trips). Off (default) => the authored profile (byte-identical). Proven: check_cart_yaw_rebase_envelope.
        self.lane_cart_yaw_rebase=bool(lane_cart_yaw_rebase)
        # fm-sim-robust-outbound: fail a NON-ACQUIRED authored lane by name FAST instead of churning
        # until the 120 s mission nav-stuck WALL budget. When the active lane yields an empty route
        # (lane_not_acquired / lane_blocked / lane_leadin_unreachable) the base COMMANDS nothing, so no
        # progress is made; but the empty route cycles reason=lane_not_acquired <-> replan_changed_
        # observations under the jittery live cloud, which never reaches the base `_wait` 8 s
        # blocked_timeout (coldlivewin: ~120 s WALL frozen before the external nav-stuck fired). This
        # SIM-clock accumulator keys on contiguous no-base-progress ticks with the active lane
        # unacquired and, past `lane_acquire_timeout_s` SIM seconds, FAILS the leg terminal with the
        # specific reason + '_timeout' -- a named, pace-immune fail in a few sim seconds. Reset on any
        # base progress or a non-empty route. 0 (default) => off => the legacy churn (byte-identical).
        if not (isinstance(lane_acquire_timeout_s,(int,float)) and math.isfinite(lane_acquire_timeout_s)
                and lane_acquire_timeout_s>=0.0):
            raise ValueError('lane_acquire_timeout_s must be a finite number of seconds >= 0')
        self.lane_acquire_timeout_s=float(lane_acquire_timeout_s)
        self._acquire_fail_s=0.0        # contiguous SIM seconds the active lane has been unacquired
        self._acquire_fail_pose=None    # base pose at the start of the current no-progress window
        # fm-sim-robust-outbound-leadin-overshoot (navigation.tow_lane_goal_stop, default off =>
        # byte-identical): make the COUPLED authored-lane goal arrival SEAT-INDEPENDENT, so the tow
        # stops AT its goal instead of overshooting a ball it can never hit. A coupled tow tracks its
        # lane with a steady-state LATERAL offset (coldwin3: the lead-in's descending final segment +
        # the base's steering lag left it riding ~0.09 m NORTH of the centreline), but the arrival
        # capture was a HARDCODED 0.08 m / 0.08 rad ball -- SMALLER than that physical offset. So the
        # train reached the goal x at 0.09 m lateral (just outside 0.08), never latched, and a
        # forward-only (no-reverse) train sailed 0.3 m past the goal into lane_leadin_unreachable
        # (the lead-in rebuilt from past-the-goal rolls further away forever). The authored lane ALREADY
        # declares its ACHIEVABLE arrival band -- goal_tol_xy (0.30) / goal_tol_yaw -- so when this is on
        # AND an authored lane is active: (1) the coupled arrival capture uses the lane's goal_tol_xy /
        # goal_tol_yaw (not 0.08), so the observed ~0.09 m offset latches; (2) the follower decelerates
        # by REMAINING ROUTE distance (not Euclidean distance-to-goal, which a laterally-offset base
        # never drives to zero) so it comes to rest AT the route end and the 0.03 m/s floor is dropped
        # there -- the train stops before/at the goal and NEVER passes it (a coupled train cannot
        # reverse); and (3) a seat left genuinely PAST the goal (beyond the route end, forward-only
        # unreachable) fails FAST by name (lane_leadin_past_goal) instead of churning to the acquire
        # timeout. The authored goal_tol IS the arrival contract; a seat whose residual offset exceeds it
        # still refuses (the author's achievable band is the authority). Off => the legacy hardcoded 0.08
        # ball + Euclidean speed profile, byte-identical.
        self.tow_lane_goal_stop=bool(tow_lane_goal_stop)
        # Authored pre-planned tow lanes (TowLaneGraph) for COUPLED drives. None => every coupled
        # drive free-plans exactly as before (byte-identical). When set, a coupled drive whose goal
        # matches a lane follows that lane's validated trajectory and NEVER free-searches (so a
        # loaded host cannot starve it into a planning_timeout, and the reorientation loop that made
        # the east-facing return time out is replaced by the authored west-facing path). Lanes are
        # expanded against the static map by the factory before the controller is used.
        self.lane_graph=lane_graph
        self._active_lane=None
        # Coupled-train A* wall-clock budget. The coupled outbound/return plan normally resolves in
        # ~30 ms via the cheap corridor route (map passage hints), never touching the full search; but
        # if the corridor is perturbed and the plan falls to the full search, a HEAVILY LOADED host
        # (the mission shares the box with Isaac render + commissioning) can starve the search inside
        # the default 30 s wall deadline and return 'planning_timeout' -> blocked_no_route_timeout
        # (the failure mode the fullscope nav05 proof hit under load). This is a COMPUTE budget, not a
        # collision tolerance: raising it never accepts any unverified geometry. None => default 30 s.
        if planner_timeout is None:
            self.planner=ArticulatedPlanner(self.world)
        else:
            if not (isinstance(planner_timeout,(int,float)) and math.isfinite(planner_timeout)
                    and planner_timeout>0):
                raise ValueError('planner_timeout must be a positive, finite number of seconds')
            self.planner=ArticulatedPlanner(self.world,timeout=float(planner_timeout))
        super().__init__(grid,Footprint(self.spec.transport_radius,abs(self.spec.hitch_x),
                         math.hypot(self.spec.cart_hitch_x+self.spec.cart_length/2,self.spec.cart_width/2),
                         margin=0.05,min_turn_radius=1.5,
                         max_articulation=self.spec.max_articulation),**kwargs)
        self.route=None
        self.index=0
        self.planning_stop=False

    def start(self,goal,coupled,*,goal_articulation=None,arrival_xy_tol=None,arrival_yaw_tol=None,static_drive=False,carry_drive=False,strict_xy_arrival=False):
        super().start(goal,coupled,arrival_xy_tol=arrival_xy_tol,arrival_yaw_tol=arrival_yaw_tol,static_drive=static_drive,carry_drive=carry_drive,strict_xy_arrival=strict_xy_arrival)
        self.route=None
        self.index=0
        self.planning_stop=False
        self._mtr_window=0.0                               # fm446 (C): fresh goal -> fresh over-rotation window
        self._prog_anchor=None                             # fresh goal -> fresh net-progress anchor/window
        self._prog_last_now=None
        self._prog_stuck_s=0.0
        self._force_progress_creep=False
        self._recovery_active=False                        # fresh goal -> no break-away recovery in flight
        self._recovery_attempt=0
        self._recovery_window_start=None
        self._recovery_anchor=None
        self._breakaway_cmd=None
        self._breakaway_turn=False
        self._acquire_fail_s=0.0                           # fresh goal -> fresh lane-acquire window
        self._acquire_fail_pose=None
        self.goal_articulation=self.spec.max_articulation if goal_articulation is None else goal_articulation
        # Resolve the authored tow lane for this COUPLED goal (if any). Uncoupled drives never use a
        # lane. Pose acquirability + live-obstacle swept validation are checked when the route is built.
        self._active_lane=(self.lane_graph.resolve_goal(self.goal)
                           if (coupled and self.lane_graph) else None)
        if self._active_lane is not None:
            # The lane is the authority on its OWN kinematically-achievable terminal state, so arrival
            # articulation follows the lane (not a generic per-action value the coupled train may be
            # unable to reach into this goal). Documented in docs/restock_navigation.md.
            self.goal_articulation=self._active_lane.goal_articulation
        if _NAV_PLAN_DIAG:
            print('NAV_PLAN_DIAG start goal=%s coupled=%s active_lane=%s lanes=%s'
                  % (tuple(round(c,3) for c in self.goal), coupled,
                     (self._active_lane.name if self._active_lane is not None else None),
                     ([ln.name for ln in self.lane_graph.lanes] if self.lane_graph else None)),
                  flush=True)

    def plan_observed_route(self,state,obs,uncertainty):
        """Plan against current nvblox unknown/occupied cells when configured."""
        if self._active_lane is not None:
            return self._plan_lane(state,obs,uncertainty)
        planner=self.planner
        if obs.temporal_map is not None:
            planner=ArticulatedPlanner(ArticulatedMap(obs.temporal_map.planning_grid(self.grid),self.spec))
            uncertainty=max(.10,uncertainty)
        return planner.plan(state,self.goal,obs.obstacles,uncertainty,self.goal_articulation)

    def _plan_lane(self,state,obs,uncertainty):
        """Follow the active authored tow lane instead of free planning.

        The lane's trajectory was expanded + validated against the static map at build time; here
        it is re-validated against the LIVE obstacle observation (the same swept-envelope replay
        `validate_seed` performs) from the lane's OWN authored start -- so acceptance means the
        authored centreline is still clear, not that an open-loop replay from the jittered measured
        pose happens to land on the goal (the controller closes that loop by tracking the route).
        The measured base must be close enough to acquire the lane; a live obstacle on the lane
        blocks the tow. Crucially this NEVER runs the expensive hybrid-A* search, so the coupled
        drive is bounded in CPU and cannot be starved into a planning_timeout under host load.
        """
        lane=self._active_lane
        if obs.temporal_map is not None:
            world=ArticulatedMap(obs.temporal_map.planning_grid(self.grid),self.spec)
            uncertainty=max(.10,uncertainty)
            planner=ArticulatedPlanner(world,timeout=self.planner.timeout)
        else:
            world=self.world
            planner=self.planner
        # Re-validate the authored lane against the LIVE obstacle cloud. When the opt-in proud
        # tolerance is set, discount live returns within that distance of KNOWN static geometry
        # (proud returns of the shelf/support the lane hugs) so they cannot false-block the lane;
        # genuinely new returns in open space are kept and still block. OFF (0.0) => the full live
        # cloud, byte-identical. Computed before the acquire so the lead-in fallback shares the same
        # (discounted) cloud it validates the authored route against.
        dyn=obs.obstacles
        discounted=0
        if self.lane_trust_static:
            # Trust the static map the lane was build-validated against: validate the authored route
            # against static geometry ALONE. The live braking gate (gate_velocity) is untouched and
            # still stops the tow for a genuinely new obstacle ahead.
            dyn=()
            discounted=len(obs.obstacles)
        elif self.lane_static_tol>0 and dyn:
            kept=tuple((x,y) for (x,y) in dyn if not world.near_static(x,y,self.lane_static_tol))
            discounted=len(dyn)-len(kept)
            dyn=kept
        if self.lane_cart_yaw_rebase:
            # fm452(A): the FOLLOW tracks the re-based cart_yaw (zeroed at the base's aligned entry, then
            # growing as the cart lags the sharpening turn -- see `_rebase_cart_yaw`), so the ACQUIRE must
            # use the SAME expectation, not the authored (swung-from-lane-start) cart_yaw. Checked against
            # the authored cart_yaw, an aligned-but-lagging cart after a drop diverges past start_tol_cart_yaw
            # and re-acquire refuses forever -> lane_not_acquired churn (run-334: 859 trips while the base was
            # otherwise driving the turn). Re-use the re-base code as the gate: acquire on xy-proximity to the
            # route (cart_yaw handled by the re-base, same as the stop_replan drop) AND the live cart re-basing
            # cleanly -- a cart genuinely off the lane re-bases to a swept-envelope breach, so `_rebase_cart_yaw`
            # returns the authored route UNCHANGED (its `is` identity).
            # fm455 b-fix: the acquire cart gate must use the SAME predicate+threshold as the FOLLOW's
            # stop_replan (cart_err <= lane_cart_yaw_tol against the re-based nearest), NOT the stricter
            # rebase-envelope-clear test. The envelope test refused at ~0.29 while the follow tolerates
            # 0.30 -> a dead zone that converted a tolerable lag into lane_not_acquired churn (run-335:
            # cart lag ~0.287 stalled the tow at 852 lane_not_acquired though the follow would have
            # continued). Re-base the route; measure cart_err at the nearest route point exactly as the
            # follow does; acquire iff within lane_cart_yaw_tol. A cart genuinely past the follow tol is
            # still refused (the follow would stop_replan it anyway) -- no dead zone, no false refusal.
            xy_ok=lane.acquirable(state.pose,None,along_route=self.lane_reacquire_along_route)
            rebased=self._rebase_cart_yaw(lane.route,state,world)
            near=min(rebased.states,key=lambda s:math.hypot(s.x-state.x,s.y-state.y)) if rebased.states else None
            cart_ok=near is not None and abs(wrap(near.cart_yaw-state.cart_yaw))<=self.lane_cart_yaw_tol
            acquired=bool(xy_ok and cart_ok)
            if _NAV_PLAN_DIAG:
                print('NAV_PLAN_DIAG acquire(rebase) lane=%s xy_ok=%s cart_ok=%s pose=%s lane_start=%s '
                      'cart_yaw=%.3f near_cart_yaw=%s cart_tol=%.3f reacquire=%s'
                      % (lane.name,xy_ok,cart_ok,tuple(round(c,3) for c in state.pose),
                         tuple(round(c,3) for c in lane.start),state.cart_yaw,
                         (round(near.cart_yaw,3) if near is not None else None),
                         self.lane_cart_yaw_tol,self.lane_reacquire_along_route),flush=True)
        else:
            acquired=lane.acquirable(state.pose,state.cart_yaw,along_route=self.lane_reacquire_along_route)
            if _NAV_PLAN_DIAG:
                print('NAV_PLAN_DIAG acquire lane=%s acquirable=%s pose=%s lane_start=%s cart_yaw=%.3f reacquire=%s'
                      % (lane.name,acquired,tuple(round(c,3) for c in state.pose),
                         tuple(round(c,3) for c in lane.start),state.cart_yaw,self.lane_reacquire_along_route),
                      flush=True)
        if not acquired:
            # fm-sim-robust-outbound (option c): the base seated OFF the fine centreline. If a lead-in is
            # authorised, synthesise a bounded swept-validated coupled merge from the seat ONTO the lane and
            # follow a path that STARTS where the base actually is; the follower then acquires trivially.
            # A seat beyond the band the lane can absorb returns None -> a distinct NAMED reason so the
            # acquire watchdog fails the leg fast instead of leaving it to the 120 s mission nav-stuck.
            li=(lane.build_lead_in(state,world,self.spec,dyn,max(.02,uncertainty))
                if lane.lead_in_max_m>0 else None)
            if li is not None and li.states:
                if _NAV_PLAN_DIAG:
                    print('NAV_PLAN_DIAG lane_leadin lane=%s states=%d from_pose=%s converges_to_goal=%s'
                          % (lane.name,len(li.states),tuple(round(c,3) for c in state.pose),
                             tuple(round(c,3) for c in li.states[-1].pose)),flush=True)
                li.reason='validated_lane_leadin'
                if self.lane_cart_yaw_rebase:
                    li=self._rebase_cart_yaw(li,state,world)
                    li.reason='validated_lane_leadin'
                return li
            # fm-sim-robust-outbound-leadin-overshoot: a seat left genuinely PAST the goal (beyond the
            # route end, the base pointing away so a forward-only train can never return) is provably
            # unrecoverable -- fail FAST by name NOW instead of churning the empty route to the acquire
            # timeout. Detect it from the route end: the base is past the last route point AND displaced
            # from the goal in the route's own travel direction. Gated on tow_lane_goal_stop => off is
            # byte-identical (the legacy lane_leadin_unreachable + acquire-timeout path).
            if (self.tow_lane_goal_stop and lane.lead_in_max_m>0 and lane.route is not None
                    and lane.route.states and self._past_route_end(state,lane.route)):
                self.terminal=Result(status='failed',reason='lane_leadin_past_goal')
                if _NAV_PLAN_DIAG:
                    print('NAV_PLAN_DIAG lane_leadin_past_goal lane=%s pose=%s goal=%s'
                          % (lane.name,tuple(round(c,3) for c in state.pose),
                             tuple(round(c,3) for c in self.goal)),flush=True)
                return Route([],[],'lane_leadin_past_goal')
            return Route([],[],'lane_leadin_unreachable' if lane.lead_in_max_m>0 else 'lane_not_acquired')
        checked=planner.validate_seed(lane.route,lane.start_state(),self.goal,dyn,
                                      max(.02,uncertainty),self.goal_articulation)
        if not checked.states:
            # Keep the contract reason 'lane_blocked', but never leave WHY opaque again (run-321b
            # burned an hour on a bare 'lane_blocked'): name the specific validate_seed rejection and
            # the live/discounted cloud sizes so the next run's cause is readable from the console.
            # Also DUMP a capped sample of the KEPT returns with each one's distance to the nearest
            # static-map geometry (fm425: the raw cloud was never persisted, so the run-322b 128-kept
            # points could not be classified post-hoc -- now every lane_blocked records, for free, how
            # far the surviving returns sit from mapped static, which tells proud-of-mapped (small
            # distance) from off-finer-geometry/unmapped (large distance) at a glance).
            kept_diag=tuple((round(x,2),round(y,2),round(world.nearest_static(x,y),3))
                            for (x,y) in dyn[:20]) if dyn else ()
            print('LANE_BLOCKED_DIAG lane=%r vseed_reason=%s live_pts=%d discounted_near_static=%d '
                  'kept=%d proud_tol_m=%.3f trust_static=%s start=%s goal=%s kept_near_static=%s'
                  % (lane.name,checked.reason,len(obs.obstacles),discounted,len(dyn),
                     self.lane_static_tol,self.lane_trust_static,tuple(round(c,2) for c in lane.start),
                     tuple(round(c,2) for c in self.goal),kept_diag),flush=True)
            return Route([],[],'lane_blocked')
        checked.reason='validated_lane'
        if self.lane_cart_yaw_rebase:
            checked=self._rebase_cart_yaw(checked,state,world)
        return checked

    def _past_route_end(self,state,route):
        """True iff the base has overshot the route's final point in the route's own travel direction.

        A coupled tow cannot reverse, so a base beyond the end heading can never drive back onto the
        route -- this is the provably-unrecoverable past-goal case the goal-stop fast-fail names."""
        st=route.states
        end=st[-1]
        hx,hy=(math.cos(end.yaw),math.sin(end.yaw))
        if len(st)>=2:
            dx,dy=end.x-st[-2].x,end.y-st[-2].y
            if math.hypot(dx,dy)>1e-9:
                hx,hy=dx,dy
        hn=math.hypot(hx,hy) or 1.0
        proj=((state.x-end.x)*hx+(state.y-end.y)*hy)/hn
        return proj>0.05

    def _acquire_watchdog(self,dt,reason,odom):
        """fm-sim-robust-outbound: fail a persistently NON-ACQUIRED authored lane by name, FAST.

        The empty-route churn (lane_not_acquired/lane_blocked/lane_leadin_unreachable cycling with
        replan_changed_observations as the live cloud jitters) commands the base nothing and never
        reaches the base `_wait` 8 s blocked_timeout, so a genuine acquisition deadlock otherwise sits
        until the external 120 s mission nav-stuck WALL budget (coldlivewin). This accrues the
        CONTIGUOUS SIM seconds the active lane has stayed unacquired with NO base progress and, past
        `lane_acquire_timeout_s`, sets a terminal failed Result naming the specific reason + '_timeout'
        -- a pace-immune fail in a few sim seconds. Any base progress (the lead-in acquired and the
        train is moving) or a non-empty route re-anchors the window. Off (timeout <= 0) or no active
        lane => never fails (byte-identical)."""
        if (self.lane_acquire_timeout_s<=0 or self._active_lane is None
                or reason not in ('lane_not_acquired','lane_blocked','lane_leadin_unreachable')):
            self._acquire_fail_s=0.0
            self._acquire_fail_pose=None
            return None
        pose=getattr(odom,'pose',None)
        progressed=False
        if pose is not None and self._acquire_fail_pose is not None:
            dyaw=wrap(pose[2]-self._acquire_fail_pose[2])
            progressed=(math.hypot(pose[0]-self._acquire_fail_pose[0],
                                   pose[1]-self._acquire_fail_pose[1])>0.02 or abs(dyaw)>math.radians(1.0))
        if self._acquire_fail_pose is None or progressed:
            self._acquire_fail_pose=pose
            self._acquire_fail_s=0.0
            return None
        self._acquire_fail_s+=dt
        if self._acquire_fail_s>=self.lane_acquire_timeout_s:
            self.terminal=Result(status='failed',reason=reason+'_timeout')
            if _NAV_PLAN_DIAG:
                print('NAV_PLAN_DIAG lane_acquire_timeout lane=%s reason=%r sim_s=%.2f pose=%s'
                      % (self._active_lane.name,reason,self._acquire_fail_s,
                         tuple(round(c,3) for c in pose) if pose else None),flush=True)
            return self.terminal
        return None

    def _rebase_cart_yaw(self,route,state,world):
        """fm451: re-base the expected cart_yaw to the base's ACTUAL (aligned) entry. From the nearest route
        point to `state`, set cart_yaw = the live cart_yaw and re-integrate the off-axle trailer kinematics
        (advance()'s cart_rate, same hitch_x/cart_hitch_x) forward to the goal. The (x,y,base_yaw) centreline
        is untouched. Only used if every re-based state stays inside the swept envelope -- else the authored
        route is returned unchanged (so a cart that has genuinely departed the lane still trips stop_replan)."""
        st=route.states
        if not st:
            return route
        if route.locked and any(route.locked):
            return route    # fm455: a rigid-hitch-locked route carries cart_yaw=base_yaw by the pin; the
                            # off-axle re-base does not apply (and must not corrupt the locked steps)
        k=min(range(len(st)),key=lambda i:math.hypot(st[i].x-state.x,st[i].y-state.y))
        spec=self.spec; cy=state.cart_yaw
        rebased=[st[i] for i in range(k)]+[TrainState(st[k].x,st[k].y,st[k].yaw,cy)]
        for i in range(k,len(st)-1):
            ds=math.hypot(st[i+1].x-st[i].x,st[i+1].y-st[i].y)
            curv=wrap(st[i+1].yaw-st[i].yaw)/ds if ds>1e-9 else 0.0
            beta=st[i].yaw-cy
            cy=wrap(cy+(math.sin(beta)+spec.hitch_x*curv*math.cos(beta))/spec.cart_hitch_x*ds)
            rebased.append(TrainState(st[i+1].x,st[i+1].y,st[i+1].yaw,cy))
        if any(not world.valid(s,(),0.0) for s in rebased[k:]):     # a departed cart would breach -> keep authored
            return route
        return Route(rebased,route.curvatures,route.reason)

    def _coverage_ok(self,odom,obs,radius,stop_distance,v,w):
        """Uncoupled braking-sweep coverage (the reference base path via super()).

        Applies the same occupied-envelope subtraction as the coupled gate, with the
        base-only footprint: self-occluded side and rear beams that only enter the
        swept capsule as the heading changes are exempt (footprint-covered), while a
        genuinely unobserved forward or lateral-bulge zone still refuses. Fixtures with
        no coverage field fall back to the legacy scalar reach.
        """
        if getattr(obs,'coverage',None) is None:
            return obs.coverage_radius>=radius+stop_distance
        k=w/v if v>1e-6 else 0.0
        return self._envelope_coverage_ok(odom,obs,stop_distance,k,
                                          max(.02,odom.position_uncertainty),coupled=False)

    def _envelope_coverage_ok(self,odom,obs,stop,k,uncertainty,*,coupled):
        """Certify the braking sweep by subtracting the current occupied envelope.

        Builds the convex occupied envelope and its braking sweep at the commanded
        speed/curvature (base-only when uncoupled, base + cart when coupled) and requires
        non-self observation only for the newly swept forward crescent in the direction of
        travel (`CoverageField.certifies_sweep`). The lateral extent of the capsule is
        covered by the footprint and the rear is never swept; the envelopes carry the
        0.05 m geometry margin plus the localization inflation, and the swept hull carries
        the full (cart) sweep at the commanded curvature, so a turn's outward lateral bulge
        still demands observation.
        """
        field=obs.coverage
        margin=self.world.margin+uncertainty
        yaw=odom.pose[2]
        if coupled:
            state=TrainState(*odom.pose,wrap(yaw-obs.articulation))
            end=advance(state,stop,k,self.spec)
            occupied=occupied_envelope(*state.bodies(self.spec),spec=self.spec,margin=margin)
            swept=swept_envelope([state.bodies(self.spec),end.bodies(self.spec)],
                                 self.spec,margin=margin)
        else:
            robot=Pose2(odom.pose[0],odom.pose[1],yaw)
            if abs(k)<1e-9:
                end=Pose2(odom.pose[0]+stop*math.cos(yaw),odom.pose[1]+stop*math.sin(yaw),yaw)
            else:
                eyaw=yaw+k*stop
                end=Pose2(odom.pose[0]+(math.sin(eyaw)-math.sin(yaw))/k,
                          odom.pose[1]-(math.cos(eyaw)-math.cos(yaw))/k,eyaw)
            occupied=occupied_envelope(robot,None,spec=self.spec,margin=margin)
            swept=swept_envelope([(robot,None),(end,None)],self.spec,margin=margin)
        travel=(math.cos(yaw),math.sin(yaw))
        return field.certifies_sweep(occupied,swept,travel,geometry_margin=self.world.margin)

    def _lateral_slack(self,s):
        """fm449: the max clear LATERAL (perpendicular-to-heading) offset of the base+cart swept envelope
        at route state `s` against the STATIC map -- the headroom that bounds how far the base may safely
        ride off the authored centreline here. Memoised per rounded state (computed once per route point),
        so the position-aware stop_replan tol costs ~one binary search per distinct point, not per tick."""
        key=(round(s.x,2),round(s.y,2),round(wrap(s.yaw),2),round(wrap(s.cart_yaw),2))
        cached=self._slack_cache.get(key)
        if cached is not None:
            return cached
        nx,ny=-math.sin(s.yaw),math.cos(s.yaw)
        lo,hi=0.0,0.8
        for _ in range(14):
            mid=(lo+hi)/2
            ok=all(self.world.valid(TrainState(s.x+sg*mid*nx,s.y+sg*mid*ny,s.yaw,s.cart_yaw),(),0.0)
                   for sg in (+1,-1))
            lo,hi=(mid,hi) if ok else (lo,mid)
        self._slack_cache[key]=lo
        return lo

    def gate_velocity(self,now,odom,obs,linear,angular):
        if self.require_temporal_map and (obs is None or obs.temporal_map is None):
            return Result(reason='missing_navigation_map')
        if self.require_temporal_map and not self.coupled:
            return Result(reason='temporal_route_requires_coupled')
        if not self.coupled:
            return super().gate_velocity(now,odom,obs,linear,angular)
        reason=self._inputs(now,odom,obs)
        if reason:
            return Result(reason=reason)
        if obs.observed_space is not None:
            return Result(reason='instantaneous_coverage_retired')
        temporal=obs.temporal_map
        if temporal is not None:
            from .temporal import TemporalMap
            if not isinstance(temporal,TemporalMap):return Result(reason='invalid_temporal_map')
            reason=temporal.readiness(now)
            if reason:return Result(reason=reason)
        if not all(math.isfinite(v) for v in (linear,angular)) or linear<0:
            return Result(reason='reverse_or_invalid_command')
        # fm437/fm439: on a static-validated authored lane, trust UNOBSERVED space in the swept envelope.
        # Applies to BOTH gate paths: the CoverageField certification below (reference nav_mode -- the
        # live run-328 path: the ~90 deg southward turn can't be pre-observed because the base faces east,
        # a sensing-geometry deadlock) AND the temporal check_train. A positive/observed obstacle still
        # stops the tow (world.valid/rollout on obs.obstacles below), so only the unobserved-space refusal
        # is waived -- the static map (validated at lane build + trust_static live) is the authority there.
        trust_unknown=self.lane_trust_unknown and self._active_lane is not None
        v=min(linear,self.max_speed)
        w=max(-self.max_angular,min(angular,self.max_angular))
        if abs(w)*1.5>v+1e-9:
            return Result(reason='turn_radius')
        # fm481/fm482 creep escape (see __init__): decided BEFORE the braking computations so the
        # stop distance, coverage and rollouts below all check the command actually sent.
        escape=(self.coupled_creep_escape and v>1e-6
                and 0.02<abs(odom.linear)<max(0.06,2*self.coupled_creep_floor)
                and abs(odom.angular/odom.linear)>1/1.5+0.03)
        # fm-sim-robust-outbound: the net-no-progress monitor (step) forces the SAME committed creep,
        # keyed on NET displacement rather than the instantaneous noise band -- so a forward-commanding
        # tow that is pinned with velocity SPIKES outside the band still keeps a clean forward command
        # instead of thrashing to a measured-curvature STOP. Only forces where there IS a forward command.
        escape=escape or (self._force_progress_creep and v>1e-6)
        # fm-sim-robust-outbound-stall-recovery: a committed break-away surge REPLACES the command with a
        # clean continuous forward (the recovery pulse), bypassing the measured_turn_radius coast that
        # returns v=0 and re-rests the stuck tow. Obstacle/coverage/map safety below is UNCHANGED -- a real
        # positive obstacle, a coverage gap or a stale map still refuses the surge.
        breakaway=self._breakaway_cmd is not None
        if breakaway:
            v=min(self.max_speed,max(1e-3,self._breakaway_cmd))
            wmax=v/self.footprint.min_turn_radius
            if self._breakaway_turn and w!=0.0:
                # fm-breakaway-rock hi phase: saturate the corrective turn to the full +-wmax in the
                # follower's tracking direction (toward the route heading), to rotate a yaw-wedged base
                # back into alignment with the lane. Bounded by the coupled cap wmax=v/R; still swept-
                # checked by the braking rollout below exactly as any other commanded (v, w).
                w=wmax if w>0 else -wmax
            else:
                w=max(-wmax,min(w,wmax))
        elif escape:
            v=max(v,min(self.coupled_creep_floor,self.max_speed))
            wmax=v/self.footprint.min_turn_radius
            w=max(-wmax,min(w,wmax))
        state=TrainState(*odom.pose,wrap(odom.pose[2]-obs.articulation))
        speed=max(v,abs(odom.linear))
        delay=self.stale_after+(temporal.reaction_delay(now) if temporal else 0.)
        stop=speed*delay+speed*speed/(2*self.brake_acceleration)
        uncertainty=max(.10 if temporal else .02,odom.position_uncertainty)
        k=w/v if v>1e-6 else 0.0
        if temporal is None:
            if getattr(obs,'coverage',None) is None:
                ok=obs.coverage_radius>=self.footprint.radius(True)+uncertainty+stop
            else:
                ok=self._envelope_coverage_ok(odom,obs,stop,k,uncertainty,coupled=True)
            if not ok and not trust_unknown:
                return Result(reason='insufficient_sensor_coverage')
        if not self.world.valid(state,obs.obstacles,uncertainty):
            return Result(reason='obstacle_stop')
        predicted=self.world.rollout(state,stop,k,obs.obstacles,uncertainty)
        if not predicted:
            return Result(reason='obstacle_stop')
        # During a command change braking follows measured curvature, too.
        if abs(odom.linear)>0.02:
            if escape or breakaway:
                measured_k=k        # committed command: brake along the clamped command, not the noise arc
                if escape:
                    self.creep_escapes+=1
                    if self._force_progress_creep:
                        self.progress_escapes+=1
            else:
                measured_k=odom.angular/odom.linear
                if abs(measured_k)>1/1.5+0.03:
                    return Result(reason='measured_turn_radius')
            braking=self.world.rollout(state,stop,max(-1/1.5,min(1/1.5,measured_k)),
                                      obs.obstacles,uncertainty)
            if not braking:
                return Result(reason='obstacle_stop')
            predicted+=braking
        if temporal is not None:
            reason=temporal.check_train(predicted,self.spec,uncertainty,now,trust_unknown=trust_unknown)
            if reason:return Result(reason=reason)
        return Result(v,w,reason='tracking')

    def step(self,dt,odometry,observations,*,now=None):
        if self.require_temporal_map and (observations is None or observations.temporal_map is None):
            return Result(reason='missing_navigation_map')
        if self.require_temporal_map and not self.coupled:
            return Result(reason='temporal_route_requires_coupled')
        if not self.coupled:
            return super().step(dt,odometry,observations,now=now)
        if self.terminal:
            return self.terminal
        if not math.isfinite(dt) or not 0<dt<=self.stale_after:
            return Result(reason='invalid_control_period')
        self.elapsed+=dt
        now=self.elapsed if now is None else now
        self._force_progress_creep=False   # fm-sim-robust-outbound: set only by the net-progress monitor
        self._breakaway_cmd=None           # fm-outbound-recovery: set only by the break-away recovery below
        self._breakaway_turn=False         # fm-breakaway-rock: saturate the corrective turn (hi phase only)
        reason=self._inputs(now,odometry,observations)
        if reason:
            return self._wait(dt,reason)
        if observations.temporal_map is not None:
            from .temporal import TemporalMap
            if not isinstance(observations.temporal_map,TemporalMap):
                return self._wait(dt,'invalid_temporal_map')
            reason=observations.temporal_map.readiness(now)
            if reason:return self._wait(dt,reason)
        if self.goal is None or self.elapsed>self.goal_timeout:
            self.terminal=Result(status='failed',reason='no_goal_or_goal_timeout')
            return self.terminal
        odom,obs=odometry,observations
        state=TrainState(*odom.pose,wrap(odom.pose[2]-obs.articulation))
        distance=math.dist(odom.pose[:2],self.goal[:2])
        # fm-sim-robust-outbound-leadin-overshoot: on an active authored lane with goal-stop enabled,
        # the arrival capture is the lane's ACHIEVABLE goal_tol (goal_tol_xy/goal_tol_yaw), not the
        # hardcoded 0.08 -- a coupled tow rides a lateral offset larger than 0.08 m and could never
        # latch the tight ball. Off / no active lane => the legacy 0.08 ball (byte-identical).
        if self.tow_lane_goal_stop and self._active_lane is not None:
            arr_xy,arr_yaw=self._active_lane.goal_tol_xy,self._active_lane.goal_tol_yaw
        else:
            arr_xy=arr_yaw=0.08
        in_ball=(distance<arr_xy and abs(wrap(odom.pose[2]-self.goal[2]))<arr_yaw
                 and abs(obs.articulation)<self.goal_articulation)
        moving=abs(odom.linear)>0.02 or abs(odom.angular)>0.03
        # fm-sim-robust-outbound-leadin-overshoot: with goal-stop active and the base STILL MOVING
        # inside the (wide) goal ball but with route left to run, do NOT slam a (0,0) stop at the ball
        # EDGE (that halts the tow ~goal_tol_xy short of the goal); fall through to the route-follow,
        # which decelerates by remaining-route distance and coasts the train to rest AT the route end,
        # where the next tick latches here. Legacy (goal-stop off) always enters the block => byte-identical.
        decel_follow=(self.tow_lane_goal_stop and self._active_lane is not None
                      and self.route is not None and self.route.states)
        if in_ball and not (decel_follow and moving):
            stopped=self.gate_velocity(now,odom,obs,0.0,0.0)
            if stopped.reason!='tracking':
                return self._wait(dt,stopped.reason)
            if moving:
                return Result(reason='settling_at_goal')
            self.terminal=Result(status='succeeded',reason='arrived_stopped')
            return self.terminal
        if self.route is None:
            if not self.planning_stop:
                self.planning_stop=True
                if _NAV_PLAN_DIAG:
                    print('NAV_PLAN_DIAG route_none->planning_stop active_lane=%s elapsed=%.2f pose=%s'
                          % ((self._active_lane.name if self._active_lane is not None else None),
                             self.elapsed, tuple(round(c,3) for c in odom.pose)), flush=True)
                return Result(reason='planning_stop')
            if abs(odom.linear)>0.02 or abs(odom.angular)>0.03:
                if _NAV_PLAN_DIAG:
                    print('NAV_PLAN_DIAG waiting_stopped_for_plan rv=%.4f rw=%.4f'
                          % (odom.linear, odom.angular), flush=True)
                return Result(reason='waiting_stopped_for_plan')
            if obs.temporal_map is not None:
                reason=obs.temporal_map.readiness(now)
                if reason:
                    if _NAV_PLAN_DIAG:
                        print('NAV_PLAN_DIAG temporal_map_not_ready reason=%r' % reason, flush=True)
                    return self._wait(dt,reason)
            self.route=self.plan_observed_route(state,obs,max(0.02,odom.position_uncertainty))
            self.planned_obstacles=obs.obstacles
            self.planned_map=obs.temporal_map.slice if obs.temporal_map is not None else None
            self.index=0
            if _NAV_PLAN_DIAG:
                print('NAV_PLAN_DIAG plan_observed_route active_lane=%s states=%d reason=%r goal=%s state=%s'
                      % ((self._active_lane.name if self._active_lane is not None else None),
                         len(self.route.states), self.route.reason,
                         tuple(round(c,3) for c in self.goal),
                         (round(state.x,3),round(state.y,3),round(state.yaw,3),round(state.cart_yaw,3))),
                      flush=True)
            if self.route.states:
                self._acquire_fail_s=0.0; self._acquire_fail_pose=None   # lane acquired -> fresh window
                return Result(reason='route_ready')
            if self.terminal is not None:        # _plan_lane named a terminal fast-fail (past-goal)
                return self.terminal
            term=self._acquire_watchdog(dt,self.route.reason,odom)
            if term is not None:
                return term
            return Result(reason=self.route.reason)
        if not self.route.states:
            if self.terminal is not None:        # _plan_lane named a terminal fast-fail (past-goal)
                return self.terminal
            term=self._acquire_watchdog(dt,self.route.reason,odom)
            if term is not None:
                return term
            if (obs.obstacles!=self.planned_obstacles or
                    (obs.temporal_map is not None and obs.temporal_map.slice is not self.planned_map)):
                self.route=None
                return Result(reason='replan_changed_observations')
            return self._wait(dt,'blocked_no_route')
        candidates=range(self.index,min(len(self.route.states),self.index+20))
        self.index=min(candidates,key=lambda i:math.hypot(self.route.states[i].x-state.x,
                                                        self.route.states[i].y-state.y))
        nearest=self.route.states[self.index]
        track_tol=self.lane_track_tol
        slack=None
        track_err=math.hypot(nearest.x-state.x,nearest.y-state.y)
        if self.lane_track_envelope:                     # fm449: cap by the local swept-envelope slack
            slack=self._lateral_slack(nearest)
            track_tol=min(self.lane_track_tol, max(0.10, slack-0.05))
        # fm450: per-point tracking margin, so run-333 shows how close the base runs to its eff tol at the
        # narrow shelf-end pinch (eff tol -> 0.10 m there) vs the wide turn. Throttled.
        cart_err=abs(wrap(nearest.cart_yaw-state.cart_yaw))
        self._trackn=getattr(self,'_trackn',0)+1
        if self._trackn % 30 == 1:
            # fm449/450 + run-333: log BOTH stop_replan conditions -- xy (err vs tol+slack) AND the cart_yaw
            # divergence (cart_err vs lane_cart_yaw_tol). run-333 showed xy err ~0.066 << 0.35 so the trip is
            # the CART_YAW, not xy; this exposes which condition actually fires and by how much.
            print('TRACK_DIAG err=%.3f tol=%.3f slack=%s cart_err=%.3f cart_tol=%.3f at=(%.2f,%.2f)' %
                  (track_err, track_tol, ('%.3f'%slack) if slack is not None else 'n/a',
                   cart_err, self.lane_cart_yaw_tol, nearest.x, nearest.y), flush=True)
        if track_err>track_tol or cart_err>self.lane_cart_yaw_tol:
            self.route=None
            return Result(reason='stop_replan_tracking_error')
        # Follow the checked curvature, rather than cutting tight portal corners
        # with a long pure-pursuit lookahead. Project onto the closest local edge
        # to avoid waypoint-quantization heading error.
        choices=[]
        for j in range(max(0,self.index-2),min(len(self.route.states)-1,self.index+3)):
            a,b=self.route.states[j:j+2]
            dx,dy=b.x-a.x,b.y-a.y
            t=max(0.0,min(1.0,((state.x-a.x)*dx+(state.y-a.y)*dy)/max(1e-9,dx*dx+dy*dy)))
            x,y=a.x+t*dx,a.y+t*dy
            choices.append((math.hypot(state.x-x,state.y-y),j,t,x,y))
        _,j,t,x,y=min(choices)
        a,b=self.route.states[j:j+2]
        ref_yaw=wrap(a.yaw+t*wrap(b.yaw-a.yaw))
        lateral=-math.sin(ref_yaw)*(x-state.x)+math.cos(ref_yaw)*(y-state.y)
        curvature=self.route.curvatures[j]+1.5*wrap(ref_yaw-state.yaw)+2*lateral
        curvature=max(-self.lane_max_curvature,min(self.lane_max_curvature,curvature))
        v=min(self.max_speed,max(0.03,distance*0.5))
        # Slow for tight turns so the under-powered coupled base does not over-rotate the
        # MEASURED arc past the min radius: at the curvature cap command lane_curve_speed_floor*max_speed,
        # scaling up to full speed when straight. Keeps the commanded speed near what the base
        # actually achieves while towing, so the braking-curvature gate refuses far less. (fm429: the
        # floor is config-tunable; the normaliser uses the physical cap so a lower lane_max_curvature
        # does not itself change the straight-vs-curve speed schedule.)
        floor=self.lane_curve_speed_floor
        v=min(v,self.max_speed*max(floor,1.0-(1.0-floor)*min(1.0,abs(curvature)*self.footprint.min_turn_radius)))
        # fm450: slack-proportional speed cap -- SLOW THROUGH a narrow pinch (not loosen the tol there), so
        # the tracking error stays within the tight eff tol. Tracking error scales ~linearly with speed
        # (run-332: ~0.3 m error at 0.25 m/s on the turn), so cap v <= (slack-0.05)/1.5 to keep the error
        # under the envelope headroom; a wide-slack turn is unaffected (the cap exceeds max_speed there),
        # only the shelf-end pinch (slack 0.14 -> ~0.06 m/s) slows. Kept moving by a 0.05 m/s floor. Gated on
        # lane_track_envelope (the same opt-in as the position-aware tol); off => unchanged.
        if self.lane_track_envelope and slack is not None:
            v=min(v,max(0.05,(slack-0.05)/1.5))
        # fm-sim-robust-outbound-leadin-overshoot: decelerate to a true stop AT the route end so a
        # forward-only coupled tow halts at its goal instead of coasting past it (it cannot reverse).
        # The remaining distance is measured ALONG THE ROUTE from the nearest route point to the end --
        # NOT the Euclidean distance-to-goal, which a laterally-offset base (coldwin3: ~0.09 m north)
        # never drives to zero, so a Euclidean brake would keep commanding speed as the base slid past
        # the goal x. v <= sqrt(2 * brake_acceleration * remaining) brakes within the reachable
        # distance, and near the end the 0.03 m/s floor above is OVERRIDDEN (min, not max) so the train
        # can actually come to rest. The arrival capture (lane goal_tol, above) latches once stopped
        # within the achievable ball. Gated on tow_lane_goal_stop + an active lane => off is byte-identical.
        if self.tow_lane_goal_stop and self._active_lane is not None and self.route.states:
            st=self.route.states
            remaining=sum(math.dist(st[i].pose[:2],st[i+1].pose[:2])
                          for i in range(self.index,len(st)-1))
            v=min(v,math.sqrt(2.0*self.brake_acceleration*max(0.0,remaining)))
        # fm-sim-robust-outbound: SIM-clock net-progress monitor (default off => byte-identical). We are
        # about to command a forward v (>=0.03) along the route; measure whether the base has actually
        # MOVED >= coupled_progress_min_m from its anchor. Keyed on NET displacement + the SIM clock, so
        # render pacing cannot distort it and the 'tracking'-dominated warm10 pin (which measured_turn_
        # fault_s / the noise-band creep both miss) is caught: ESCALATE (force the committed creep) while
        # trying to drive through, then FAULT fast -> StageRetry replan well before the WALL nav-stuck.
        if self.coupled_progress_escape:
            # Only CONTIGUOUS tracking-with-no-progress accrues. A GAP since the last monitor tick (the
            # base legitimately sat in a preceding planning_stop/blocked-aisle wait, which return before
            # this monitor) re-anchors a fresh window, so a long legitimate wait is never charged against
            # the resumed drive; a route drop->replan mid-stall is a 1-2 tick gap and stays contiguous.
            gap=(self._prog_anchor is None or self._prog_last_now is None
                 or (now-self._prog_last_now)>max(5*dt,0.5))
            if gap or math.hypot(state.x-self._prog_anchor[1],
                                 state.y-self._prog_anchor[2])>=self.coupled_progress_min_m:
                self._prog_anchor=(now,state.x,state.y)           # fresh window / real net progress
                self._prog_stuck_s=0.0
            else:
                self._prog_stuck_s=max(0.0,now-self._prog_anchor[0])   # SIM seconds pinned below min_m
            self._prog_last_now=now
            if self.coupled_progress_recovery:
                # fm-sim-robust-outbound-stall-recovery: a BOUNDED break-away maneuver replaces the
                # escape->fault timing. Once the base has been pinned below min_m for escape_s SIM seconds,
                # run up to coupled_breakaway_attempts windows of coupled_breakaway_window_s SIM seconds,
                # each commanding a CONTINUOUS pulsed committed-forward surge (set _breakaway_cmd; consumed
                # by gate_velocity) so the tow sustains drive effort instead of the stop<->forward thrash.
                # A window that nets >= min_m displacement resets _prog_stuck_s above (<escape_s) and ends
                # the recovery (broke free); exhausting every attempt faults 'coupled_no_progress' terminal.
                if self._prog_stuck_s>=self.coupled_progress_escape_s:
                    if not self._recovery_active:
                        self._recovery_active=True
                        self._recovery_attempt=0
                        self._recovery_window_start=now
                        self._recovery_anchor=(state.x,state.y)
                        self.breakaway_recoveries+=1
                        print('NAV_BREAKAWAY enter attempt=0 at=(%.3f,%.3f) stuck=%.2fs rock=%d'
                              % (state.x,state.y,self._prog_stuck_s,int(self.coupled_breakaway_rock)),
                              flush=True)
                    elif now-self._recovery_window_start>=self.coupled_breakaway_window_s:
                        # window elapsed with < min_m net progress (progress would have reset _prog_stuck_s)
                        self._recovery_attempt+=1
                        if self._recovery_attempt>=self.coupled_breakaway_attempts:
                            self.terminal=Result(status='failed',reason='coupled_no_progress')
                            print('NAV_BREAKAWAY exhausted attempts=%d -> coupled_no_progress'
                                  % self._recovery_attempt, flush=True)
                            return self.terminal
                        self._recovery_window_start=now
                        self._recovery_anchor=(state.x,state.y)
                        print('NAV_BREAKAWAY next attempt=%d at=(%.3f,%.3f)'
                              % (self._recovery_attempt,state.x,state.y), flush=True)
                    # pulse the committed forward speed (square wave) to rock the stuck contact; reverse is
                    # clamped off for a coupled train (wheel_targets), so the low phase is a relaxed forward.
                    hi=(int((now-self._recovery_window_start)/self.coupled_breakaway_pulse_s)%2)==0
                    if self.coupled_breakaway_rock:
                        # fm-breakaway-rock: a STRONGER forward launch surge (more momentum + drive force,
                        # and a higher steering cap wmax=v/R) with the corrective turn SATURATED on the hi
                        # phase (via _breakaway_turn) to rotate a yaw-wedged base back toward the lane
                        # heading; the lo phase relaxes (0.4x, straight clamp) to rock the contact.
                        self._breakaway_cmd=(self.coupled_breakaway_launch if hi
                                             else self.coupled_breakaway_launch*0.4)
                        self._breakaway_turn=hi
                    else:
                        self._breakaway_cmd=(self.coupled_breakaway_speed if hi
                                             else self.coupled_breakaway_speed*0.4)
                elif self._recovery_active:
                    # Stuck cleared while a recovery was in flight: broke free iff real net displacement
                    # (not a legitimate-wait gap) accrued from the active window's anchor.
                    moved=(self._recovery_anchor is not None
                           and math.hypot(state.x-self._recovery_anchor[0],
                                          state.y-self._recovery_anchor[1])>=self.coupled_progress_min_m)
                    self._recovery_active=False
                    if moved:
                        self.breakaway_successes+=1
                        print('NAV_BREAKAWAY success: base broke free', flush=True)
            else:
                if self._prog_stuck_s>=self.coupled_progress_fault_s:
                    self.terminal=Result(status='failed',reason='coupled_no_progress')
                    return self.terminal
                self._force_progress_creep=self._prog_stuck_s>=self.coupled_progress_escape_s
        # fm-sim-robust-outbound-throat-yaw-align: the pre-throat heading hold. On a STRAIGHT authored
        # segment, once the base has drifted more than `heading_hold_bound` to the CCW (throat-lip / wedge)
        # side of the lane goal (travel) heading, OVERRIDE the follower command: hold a fixed modest forward
        # creep (restoring a speed-schedule-independent steering authority wmax = creep/1.5 AND capping the
        # forward speed so a misaligned base slows and aligns before entering the throat) and SATURATE the
        # corrective turn toward the axis. ONE-SIDED (the safe CW/south side is never force-turned) and
        # FORWARD-only. Skipped while PR 250's break-away owns the command (_breakaway_cmd set, the pinned
        # backstop) so the two never fight. Off / off-lane / within bound / on a curved segment => untouched.
        _hh=(self.coupled_heading_hold and self._active_lane is not None and self._breakaway_cmd is None)
        if _hh and self.heading_hold_prethroat_m>0.0 and self.heading_hold_throat is not None:
            # fm-sim-robust-heading-hold-bound: the two-sided PRE-THROAT ALIGNMENT GATE (see __init__). The
            # reference is the TRAVEL axis (heading_hold_axis else the lane goal heading) -- the direction
            # that threads the x-gap, which the passing runs held -- NOT the local descending segment tangent.
            axis=self.heading_hold_axis if self.heading_hold_axis is not None else self.goal[2]
            ae=wrap(state.yaw-axis)                            # +: yawed CCW (throat-lip/wedge side)
            tx,ty=self.heading_hold_throat
            # distance to the throat ALONG the travel axis; >0 => throat is still ahead of the base
            ahead=(tx-state.x)*math.cos(axis)+(ty-state.y)*math.sin(axis)
            in_window=0.0<=ahead<=self.heading_hold_prethroat_m
            if in_window and abs(ae)>self.heading_hold_bound:
                # The pre-throat window's single job is to point the base at the TRAVEL axis (the heading
                # that threads the x-gap) before it reaches the lip, so: (1) slow to the creep -- restoring a
                # steering authority wmax=creep/1.5 that does NOT depend on the collapsed tracking speed
                # schedule and capping the forward speed so a misaligned base aligns rather than accelerating
                # into the throat; and (2) SATURATE the corrective turn toward the axis, TWO-SIDED
                # (curvature = -sign(ae)/min_turn_radius). Overriding the follower curvature here fully
                # subsumes the lateral-offset correction, so the lateral term can never command a turn that
                # drives the yaw further off-axis (brief clause B, strongest form). The window must be sized
                # (heading_hold_prethroat_m) to the STRAIGHT run-up to the throat -- it is not meant to span a
                # genuine authored turn; the descending centreline tangent is deliberately NOT the target
                # (aligning to it would widen the base+cart envelope through the opening).
                v=min(self.max_speed,self.heading_hold_creep)
                curvature=-math.copysign(1.0/self.footprint.min_turn_radius,ae)
                self.heading_holds+=1
                # (4) bounded in time: FAIL by name if the base cannot align inside the window
                if self._align_anchor is None:
                    self._align_anchor=now
                elif (self.heading_hold_align_timeout_s>0.0
                      and now is not None and (now-self._align_anchor)>=self.heading_hold_align_timeout_s):
                    self.terminal=Result(status='failed',reason='throat_misaligned')
                    print('NAV_HEADING_HOLD throat_misaligned at=(%.3f,%.3f) base_yaw=%.2f ae=%.2f deg '
                          'bound=%.2f after=%.1fs' % (state.x,state.y,math.degrees(state.yaw),
                          math.degrees(ae),math.degrees(self.heading_hold_bound),now-self._align_anchor),
                          flush=True)
                    return self.terminal
                if self.heading_holds==1 or self.heading_holds%30==0:
                    print('NAV_HEADING_HOLD n=%d at=(%.3f,%.3f) base_yaw=%.2f ae=%+.2f deg bound=%.2f '
                          'creep=%.3f ahead=%.2f two_sided=1'
                          % (self.heading_holds,state.x,state.y,math.degrees(state.yaw),math.degrees(ae),
                             math.degrees(self.heading_hold_bound),v,ahead),flush=True)
            else:
                self._align_anchor=None                        # aligned, or out of the pre-throat window
        elif (_hh and abs(self.route.curvatures[j])<=self.heading_hold_straight_kappa
                and wrap(state.yaw-self.goal[2])>self.heading_hold_bound):
            ae=wrap(state.yaw-self.goal[2])
            v=min(self.max_speed,self.heading_hold_creep)
            curvature=-1.0/self.footprint.min_turn_radius     # saturate CW toward the lane axis (reduce ae)
            self.heading_holds+=1
            if self.heading_holds==1 or self.heading_holds%30==0:
                print('NAV_HEADING_HOLD n=%d at=(%.3f,%.3f) base_yaw=%.2f ae=%.2f deg bound=%.2f creep=%.3f'
                      % (self.heading_holds,state.x,state.y,math.degrees(state.yaw),math.degrees(ae),
                         math.degrees(self.heading_hold_bound),v),flush=True)
        result=self.gate_velocity(now,odom,obs,v,v*curvature)
        if result.reason!='tracking':
            # A transient measured-curvature trip (the base is momentarily under commanded
            # speed on a tight turn, so its MEASURED arc is tighter than the min radius) must
            # NOT discard the whole planned route: keep it and coast so the measured curvature
            # recovers, then resume the same route - instead of a full stop + replan each trip,
            # which cycles and stalls the tow. Other refusals (obstacle, coverage, map, tracking
            # error) still drop the route and replan. A persistent trip still fails via _wait's
            # blocked timeout; self.blocked resets on the next clean tracking tick.
            # fm433 (A'-2): measured_turn_radius always coasts (keeps the route). When lane_coast_refusals
            # is set and an authored lane is active, EVERY non-tracking refusal coasts too -- the route is
            # kept and the base stops this tick (velocity 0), so a new obstacle still halts the tow and a
            # persistent refusal still fails via _wait's blocked_timeout, but a transient live-cloud flicker
            # no longer drops+replans the validated lane. Otherwise the legacy drop (byte-identical).
            coast = (result.reason=='measured_turn_radius'
                     or (self.lane_coast_refusals and self._active_lane is not None))
            if not coast:
                self.route=None
            elif result.reason=='measured_turn_radius' and self.measured_turn_fault_s is not None:
                # fm446 (C): net-accumulate the over-rotation coast; a sub-min-radius turn the coupled base
                # cannot make trips every tick -> fail fast instead of livelocking to the mission nav-stuck.
                self._mtr_window += dt
                if self._mtr_window >= self.measured_turn_fault_s:
                    self.terminal=Result(status='failed',reason='measured_turn_radius_persistent')
                    return self.terminal
            return self._wait(dt,result.reason)
        self.blocked=0.0
        self._mtr_window=max(0.0, self._mtr_window-dt)   # decay on a clean tracking tick (fm446 C)
        return result
