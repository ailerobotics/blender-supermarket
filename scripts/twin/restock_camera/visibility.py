"""Offline ray/triangle design checks. These are never sensor observations."""
from dataclasses import dataclass
import numpy as np
from .geometry import project, transform_points


@dataclass
class Surface:
    name: str
    triangles: np.ndarray


def box_surface(name, center, dimensions, transform=None):
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    corners = corners*np.asarray(dimensions)/2+center
    if transform is not None:
        corners = transform_points(transform,corners)
    faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    return Surface(name,np.array([corners[[f[0],f[i],f[i+1]]]
                                 for f in faces for i in (1,2)]))


def ray_hits(eye, point, triangles):
    """Two-sided segment intersection; excludes contact at the target endpoint."""
    t = np.asarray(triangles,float)
    if not len(t):
        return False
    direction = np.asarray(point)-eye
    e1,e2 = t[:,1]-t[:,0],t[:,2]-t[:,0]
    h = np.cross(np.broadcast_to(direction,e2.shape),e2)
    det = np.einsum('ij,ij->i',e1,h)
    usable = np.abs(det)>1e-10
    inverse = np.divide(1.,det,out=np.zeros_like(det),where=usable)
    s = np.asarray(eye)-t[:,0]
    u = inverse*np.einsum('ij,ij->i',s,h)
    q = np.cross(s,e1)
    v = inverse*(q @ direction)
    along = inverse*np.einsum('ij,ij->i',e2,q)
    return bool(np.any(usable & (u>=-1e-8) & (v>=-1e-8) & (u+v<=1+1e-8)
                       & (along>1e-6) & (along<1-1e-5)))


def inspect_view(world_from_camera, points, intrinsics, surfaces):
    camera_points = transform_points(np.linalg.inv(world_from_camera),points)
    uv,valid = project(camera_points,intrinsics)
    hits = {}
    visible = []
    for i,point in enumerate(points):
        names = [s.name for s in surfaces if ray_hits(world_from_camera[:3,3],point,s.triangles)]
        if names:
            hits[str(i)] = names
        visible.append(bool(valid[i] and not names))
    return dict(samples=len(points), visible_samples=sum(visible),
                all_samples_visible=all(visible), in_fov_and_range=int(valid.sum()),
                optical_depth_range_m=[float(camera_points[:,2].min()),float(camera_points[:,2].max())],
                pixel_bounds=[*uv.min(axis=0).tolist(),*uv.max(axis=0).tolist()],
                occluders=sorted({name for names in hits.values() for name in names}),
                blocked_sample_indices=list(hits))


def sample_plane(center, size, yaw, count=9):
    c,s = np.cos(yaw),np.sin(yaw)
    return np.array([[center[0]+c*x-s*y,center[1]+s*x+c*y,center[2]]
                     for x in np.linspace(-size[0]/2,size[0]/2,count)
                     for y in np.linspace(-size[1]/2,size[1]/2,count)])


def usd_surfaces(path, bounds, *, exclude=()):
    """Read visible Mesh/Cube prims with instance proxies; no Isaac imports.

    Geometry outside the supplied ray volume is culled. Unsupported boundable
    types in that volume raise, rather than silently deleting an occluder.
    Mesh polygons are fan triangulated (authored store faces are convex).
    """
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError('USD stage could not be opened')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
    xf = UsdGeom.XformCache()
    surfaces = []
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        name = str(prim.GetPath())
        if any(name == prefix or name.startswith(prefix+'/') for prefix in exclude):
            continue
        if not prim.IsA(UsdGeom.Boundable):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty() or np.any(np.asarray(r.GetMax())<bounds[0]) or np.any(np.asarray(r.GetMin())>bounds[1]):
            continue
        matrix = np.array(xf.GetLocalToWorldTransform(prim)).T
        if prim.IsA(UsdGeom.Cube):
            # USD matrices include scale; transform explicitly, not rigid().
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            surface = box_surface(name,[0,0,0],[size]*3)
            surface.triangles = surface.triangles @ matrix[:3,:3].T+matrix[:3,3]
        elif prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(),float)
            vertices = vertices @ matrix[:3,:3].T+matrix[:3,3]
            indices = list(mesh.GetFaceVertexIndicesAttr().Get())
            triangles,offset = [],0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset:offset+count]
                triangles.extend(vertices[[face[0],face[i],face[i+1]]] for i in range(1,count-1))
                offset += count
            surface = Surface(name,np.array(triangles))
        else:
            raise ValueError(f'unsupported occluder: {name} ({prim.GetTypeName()})')
        surfaces.append(surface)
    layers = [layer.realPath for layer in stage.GetUsedLayers() if layer.realPath]
    return surfaces,layers
