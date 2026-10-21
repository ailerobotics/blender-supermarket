"""Forward hybrid A* for the actual base, rear hitch and independently yawing cart.

The trailer model is a quasi-static no-side-slip centre-axle approximation, not
a PhysX substitute. Runtime starts every collision rollout from measured hitch
articulation. Slip/model error must stay within the configured tracking margin.
"""
from dataclasses import dataclass
import heapq
import math
import time

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


def advance(state, distance, curvature, spec):
    """Midpoint integration of an off-axle rear hitch trailer, in path length."""
    def cart_rate(yaw, phi):
        beta = yaw-phi
        return (math.sin(beta)+spec.hitch_x*curvature*math.cos(beta))/spec.cart_hitch_x
    half_yaw = state.yaw+curvature*distance/2
    half_phi = state.cart_yaw+cart_rate(state.yaw,state.cart_yaw)*distance/2
    return TrainState(state.x+distance*math.cos(half_yaw),
                      state.y+distance*math.sin(half_yaw),
                      wrap(state.yaw+curvature*distance),
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

    def rollout(self,state,distance,curvature,dynamic=(),extra_margin=0.0,step=0.08):
        """Check each local swept hull, never one convex hull of the whole turn."""
        if distance<0 or abs(curvature)>1/1.5+1e-9:
            return []
        count=max(1,math.ceil(distance/step))
        ds=distance/count
        samples=[state]
        for _ in range(count):
            nxt=advance(samples[-1],ds,curvature,self.spec)
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


class ArticulatedPlanner:
    """Bounded hybrid A*: (x,y,base yaw,cart articulation), forward arcs only."""
    def __init__(self, world, *, xy_resolution=0.20, yaw_bins=72,
                 primitive_length=0.4, max_expansions=60000, timeout=30.0):
        self.world=world
        self.xy_resolution=xy_resolution
        self.yaw_step=2*math.pi/yaw_bins
        self.length=primitive_length
        self.max_expansions,self.timeout=max_expansions,timeout
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
        states=[start]; curvatures=[]
        for a,b,k in zip(seed.states,seed.states[1:],seed.curvatures):
            ds=math.dist(a.pose[:2],b.pose[:2])
            if not math.isfinite(k) or not 0<ds<=0.10:
                return Route([],[],'invalid_seed_primitive')
            expected=advance(a,ds,k,self.world.spec)
            if (math.dist(expected.pose[:2],b.pose[:2])>1e-5
                    or abs(wrap(expected.yaw-b.yaw))>1e-5
                    or abs(wrap(expected.cart_yaw-b.cart_yaw))>1e-5):
                return Route([],[],'seed_kinematics_mismatch')
            samples=self.world.rollout(states[-1],ds,k,dynamic,extra_margin,step=0.10)
            if not samples:
                return Route([],[],'seed_collision')
            states.extend(samples[1:]); curvatures.extend([k]*(len(samples)-1))
        end=states[-1]
        if (math.dist(end.pose[:2],goal[:2])>=0.07 or abs(wrap(end.yaw-goal[2]))>=0.07
                or abs(end.articulation)>=limit):
            return Route([],[],'seed_goal_mismatch')
        return Route(states,curvatures,'validated_seed')

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
                for shot in dubins_connections(start,entry):
                    if time.monotonic()>deadline:
                        return Route([],[],'corridor_budget')
                    first=extend(start,shot)
                    if first is None or not first[0]:
                        continue
                    for departure in (.5,.625,.75,1.0,1.5,2.0,2.5,3.0):
                        middle=extend(first[0][-1],[(approach+departure,0.0)])
                        if middle is None:
                            continue
                        for final in dubins_connections(middle[0][-1],goal):
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
                                 for s in dubins_connections(state,approach))
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
            options=[(self.length,k) for k in (0.0,-1/1.5,1/1.5,-1/3,1/3)]
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
                 lane_curve_speed_floor=0.25, lane_max_curvature=None, **kwargs):
        self.require_temporal_map=require_temporal_map
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

    def start(self,goal,coupled,*,goal_articulation=None,arrival_xy_tol=None,arrival_yaw_tol=None,static_drive=False):
        super().start(goal,coupled,arrival_xy_tol=arrival_xy_tol,arrival_yaw_tol=arrival_yaw_tol,static_drive=static_drive)
        self.route=None
        self.index=0
        self.planning_stop=False
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
        if not lane.acquirable(state.pose,state.cart_yaw,along_route=self.lane_reacquire_along_route):
            return Route([],[],'lane_not_acquired')
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
        # cloud, byte-identical.
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
        return checked

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
        v=min(linear,self.max_speed)
        w=max(-self.max_angular,min(angular,self.max_angular))
        if abs(w)*1.5>v+1e-9:
            return Result(reason='turn_radius')
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
            if not ok:
                return Result(reason='insufficient_sensor_coverage')
        if not self.world.valid(state,obs.obstacles,uncertainty):
            return Result(reason='obstacle_stop')
        predicted=self.world.rollout(state,stop,k,obs.obstacles,uncertainty)
        if not predicted:
            return Result(reason='obstacle_stop')
        # During a command change braking follows measured curvature, too.
        if abs(odom.linear)>0.02:
            measured_k=odom.angular/odom.linear
            if abs(measured_k)>1/1.5+0.03:
                return Result(reason='measured_turn_radius')
            braking=self.world.rollout(state,stop,max(-1/1.5,min(1/1.5,measured_k)),
                                      obs.obstacles,uncertainty)
            if not braking:
                return Result(reason='obstacle_stop')
            predicted+=braking
        if temporal is not None:
            reason=temporal.check_train(predicted,self.spec,uncertainty,now)
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
        if (distance<0.08 and abs(wrap(odom.pose[2]-self.goal[2]))<0.08
                and abs(obs.articulation)<self.goal_articulation):
            stopped=self.gate_velocity(now,odom,obs,0.0,0.0)
            if stopped.reason!='tracking':
                return self._wait(dt,stopped.reason)
            if abs(odom.linear)>0.02 or abs(odom.angular)>0.03:
                return Result(reason='settling_at_goal')
            self.terminal=Result(status='succeeded',reason='arrived_stopped')
            return self.terminal
        if self.route is None:
            if not self.planning_stop:
                self.planning_stop=True
                return Result(reason='planning_stop')
            if abs(odom.linear)>0.02 or abs(odom.angular)>0.03:
                return Result(reason='waiting_stopped_for_plan')
            if obs.temporal_map is not None:
                reason=obs.temporal_map.readiness(now)
                if reason:return self._wait(dt,reason)
            self.route=self.plan_observed_route(state,obs,max(0.02,odom.position_uncertainty))
            self.planned_obstacles=obs.obstacles
            self.planned_map=obs.temporal_map.slice if obs.temporal_map is not None else None
            self.index=0
            return Result(reason='route_ready' if self.route.states else self.route.reason)
        if not self.route.states:
            if (obs.obstacles!=self.planned_obstacles or
                    (obs.temporal_map is not None and obs.temporal_map.slice is not self.planned_map)):
                self.route=None
                return Result(reason='replan_changed_observations')
            return self._wait(dt,'blocked_no_route')
        candidates=range(self.index,min(len(self.route.states),self.index+20))
        self.index=min(candidates,key=lambda i:math.hypot(self.route.states[i].x-state.x,
                                                        self.route.states[i].y-state.y))
        nearest=self.route.states[self.index]
        if math.hypot(nearest.x-state.x,nearest.y-state.y)>0.25 or abs(wrap(nearest.cart_yaw-state.cart_yaw))>self.lane_cart_yaw_tol:
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
        result=self.gate_velocity(now,odom,obs,v,v*curvature)
        if result.reason!='tracking':
            # A transient measured-curvature trip (the base is momentarily under commanded
            # speed on a tight turn, so its MEASURED arc is tighter than the min radius) must
            # NOT discard the whole planned route: keep it and coast so the measured curvature
            # recovers, then resume the same route - instead of a full stop + replan each trip,
            # which cycles and stalls the tow. Other refusals (obstacle, coverage, map, tracking
            # error) still drop the route and replan. A persistent trip still fails via _wait's
            # blocked timeout; self.blocked resets on the next clean tracking tick.
            if result.reason!='measured_turn_radius':
                self.route=None
            return self._wait(dt,result.reason)
        self.blocked=0.0
        return result
