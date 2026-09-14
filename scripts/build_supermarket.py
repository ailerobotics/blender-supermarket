"""Procedural supermarket builder for Blender 4.5 (runs inside Blender).

Usage (inside the running Blender via MCP execute_blender_code):
    exec(open('/home/manishsaini/workspaces/blender-supermarket/scripts/build_supermarket.py').read())
    build(stages=["all"])            # or a subset, see STAGES

Usage (headless):
    blender -b scenes/supermarket_base.blend --python scripts/build_supermarket.py -- --all --save scenes/supermarket.blend

Scale: metres, Z up, origin at the centre of the shop floor. Products share one
mesh per SKU (linked duplicates) so the file stays light and USD export can
instance them. Every product carries custom properties (sku, category, brand,
price, slot) for downstream use in Isaac Sim.
"""
import bpy, bmesh, math, json, os, sys, random, time
from mathutils import Vector

# Project root: derive from this script's own location so the generator runs in any
# checkout/worktree (env override wins). Falls back to the canonical path only when
# executed without __file__ (e.g. the live GUI's `exec(open(...).read())` over MCP).
try:
    ROOT = os.environ.get("SUPERMARKET_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    ROOT = os.environ.get("SUPERMARKET_ROOT", "/home/manishsaini/workspaces/blender-supermarket")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import importlib, catalog
importlib.reload(catalog)
from catalog import SKUS, BY_ID, CATEGORIES

PLAN = json.load(open(os.path.join(ROOT, "scripts", "planogram.json")))
FIX = PLAN["fixture"]; STORE = PLAN["store"]; BAY_W = PLAN["bay_w"]
LABELS = os.path.join(ROOT, "assets", "labels"); SIGNS = os.path.join(ROOT, "assets", "signs")
W2, D2, H_CEIL = STORE["width"] / 2, STORE["depth"] / 2, STORE["height"]
RNG = random.Random(11)
MAX_ROWS = 3       # depth rows per facing (front row + 2 behind); deeper rows are never visible
LIB = {}          # sku id -> mesh
TAGMESH = {}      # sku id -> price tag mesh
PROTO = {}        # mesh name -> prototype collection (for collection instancing)
STATS = {}
INSTANCE_MODE = "collection"   # "collection": products are collection instances (-> real USD instances); "mesh": linked duplicates

# ----------------------------------------------------------------------------- utils
def hexrgb(h, a=1.0):
    h = h.lstrip("#"); return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)

def srgb_to_lin(c):
    return tuple((v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) if i < 3 else v for i, v in enumerate(c))

def col(h, a=1.0):
    return srgb_to_lin(hexrgb(h, a))

def coll(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c

def add_obj(name, mesh, collection, loc=(0, 0, 0), rot=(0, 0, 0), scale=None, props=None):
    ob = bpy.data.objects.new(name, mesh)
    ob.location = loc; ob.rotation_euler = rot
    if scale: ob.scale = scale
    collection.objects.link(ob)
    if props:
        for k, v in props.items(): ob[k] = v
    return ob

def proto_collection(mesh):
    """One hidden prototype collection per mesh; instances reference it (exports as USD instanceable refs)."""
    c = PROTO.get(mesh.name)
    if c is None:
        root = coll("Prototypes")
        c = bpy.data.collections.new(f"P_{mesh.name}")
        root.children.link(c)
        ob = bpy.data.objects.new(f"proto_{mesh.name}", mesh)
        c.objects.link(ob)
        PROTO[mesh.name] = c
    return c

def add_instance(name, mesh, collection, loc=(0, 0, 0), rot=(0, 0, 0), props=None):
    """Place a product: collection instance (default) or a linked-duplicate mesh object."""
    if INSTANCE_MODE != "collection":
        return add_obj(name, mesh, collection, loc, rot, props=props)
    ob = bpy.data.objects.new(name, None)
    ob.instance_type = 'COLLECTION'; ob.instance_collection = proto_collection(mesh)
    ob.empty_display_type = 'PLAIN_AXES'; ob.empty_display_size = 0.02
    ob.location = loc; ob.rotation_euler = rot
    collection.objects.link(ob)
    if props:
        for k, v in props.items(): ob[k] = v
    return ob

def bm_to_mesh(name, bm, materials):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    for m in materials: me.materials.append(m)
    me.update()
    return me

# ----------------------------------------------------------------------------- materials
_MATS = {}
def principled(name, color=(0.8, 0.8, 0.8, 1), rough=0.5, metal=0.0, coat=0.0, alpha=1.0, emit=None, emit_str=0.0,
               transmission=0.0, ior=1.45, spec=0.5):
    if name in _MATS: return _MATS[name]
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Specular IOR Level"].default_value = spec
    if emit:
        bsdf.inputs["Emission Color"].default_value = emit
        bsdf.inputs["Emission Strength"].default_value = emit_str
    if transmission > 0 or alpha < 1:
        m.surface_render_method = 'DITHERED'
        try: m.use_raytrace_refraction = transmission > 0
        except Exception: pass
    _MATS[name] = m
    return m

def image_mat(name, path, rough=0.5, metal=0.0, coat=0.0, emit_str=0.0, tile=(1, 1)):
    """Principled material driven by an image (UV)."""
    if name in _MATS: return _MATS[name]
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path, check_existing=True)
    tex.location = (-400, 300)
    if tile != (1, 1):
        mp = nt.nodes.new("ShaderNodeMapping"); tc = nt.nodes.new("ShaderNodeTexCoord")
        mp.inputs["Scale"].default_value = (tile[0], tile[1], 1)
        nt.links.new(tc.outputs["UV"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Coat Weight"].default_value = coat
    if emit_str:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emit_str
    _MATS[name] = m
    return m

def polyhaven_mat(name, uv_scale):
    """Re-use a material imported by the MCP Poly Haven tool, scaling its Mapping node."""
    m = bpy.data.materials.get(name)
    if m is None:
        return principled(name + "_fallback", (0.7, 0.7, 0.68, 1), 0.6)
    nt = m.node_tree
    mp = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeMapping"), None)
    if mp: mp.inputs["Scale"].default_value = (uv_scale, uv_scale, 1)
    bsdf = nt.nodes.get("Principled BSDF")
    # the MCP importer leaves the normal map unconnected: wire nor_gl -> Normal Map -> BSDF
    if bsdf and not bsdf.inputs["Normal"].is_linked:
        nor = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage" and n.image and "nor_gl" in n.image.name), None)
        if nor:
            nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.6
            nt.links.new(nor.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if "plaster" in name and bsdf and not any(l.to_node.bl_idname == "ShaderNodeBrightContrast" for l in nt.links):
        src = bsdf.inputs["Base Color"].links[0].from_socket if bsdf.inputs["Base Color"].is_linked else None
        if src:
            bc = nt.nodes.new("ShaderNodeBrightContrast"); bc.inputs["Bright"].default_value = 0.12; bc.inputs["Contrast"].default_value = -0.05
            nt.links.new(src, bc.inputs["Color"]); nt.links.new(bc.outputs["Color"], bsdf.inputs["Base Color"])
    m.displacement_method = 'BUMP'
    m.use_fake_user = True
    return m

def fixture_mats():
    return dict(
        steel=principled("FX_steel_painted", col("#d8d8d6"), 0.42, 0.25),
        steel_dark=principled("FX_steel_dark", col("#5c5f63"), 0.45, 0.5),
        pegboard=principled("FX_pegboard", col("#e4e4e2"), 0.55, 0.1),
        kick=principled("FX_kickplate", col("#3a3c40"), 0.5, 0.3),
        rail=principled("FX_ticket_rail", col("#f4f4f4"), 0.25, 0.0, alpha=0.92),
        chrome=principled("FX_chrome", (0.9, 0.9, 0.9, 1), 0.18, 1.0),
        alu=principled("FX_aluminium", col("#3b3d42"), 0.35, 0.8),
        glass=principled("FX_glass", (1, 1, 1, 1), 0.02, 0.0, transmission=1.0, ior=1.5),
        rubber=principled("FX_rubber", col("#1e1e1e"), 0.7, 0.0),
        counter=principled("FX_counter", col("#d9d6cf"), 0.35, 0.0, coat=0.3),
        wood=principled("FX_wood", col("#8a5a2b"), 0.55, 0.0),
        wood_light=principled("FX_wood_light", col("#c69c6d"), 0.5, 0.0),
        plastic_red=principled("FX_plastic_red", col("#c81e1e"), 0.35, 0.0, coat=0.2),
        plastic_black=principled("FX_plastic_black", col("#202020"), 0.4, 0.0),
        screen=principled("FX_screen", col("#0b2540"), 0.2, 0.0, emit=col("#2a6fd6"), emit_str=2.0),
        led=principled("FX_led", (1, 1, 1, 1), 0.5, 0.0, emit=(1, 0.98, 0.95, 1), emit_str=5.0),
        led_soft=principled("FX_led_soft", (1, 1, 1, 1), 0.5, 0.0, emit=(0.95, 0.97, 1, 1), emit_str=6.0),
        ceiling=principled("FX_ceiling", col("#f2f2f0"), 0.8, 0.0),
        concrete=principled("FX_concrete", col("#8d8b86"), 0.85, 0.0),
        cardboard=principled("FX_cardboard", col("#b58a55"), 0.7, 0.0),
        chiller_body=principled("FX_chiller_body", col("#eeeeec"), 0.35, 0.2, coat=0.2),
        chiller_inner=principled("FX_chiller_inner", col("#d4d8dc"), 0.4, 0.3),
        sign_blue=principled("FX_sign_blue", col("#14285a"), 0.5, 0.0),
        cable=principled("FX_cable", col("#222222"), 0.5, 0.3),
        fruit_apple_red=principled("PR_apple_red", col("#b3191e"), 0.35, 0.0, coat=0.5),
        fruit_apple_green=principled("PR_apple_green", col("#7fb534"), 0.35, 0.0, coat=0.5),
        fruit_orange=principled("PR_orange", col("#f0801a"), 0.55, 0.0),
        fruit_lemon=principled("PR_lemon", col("#f2d21f"), 0.5, 0.0),
        fruit_lime=principled("PR_lime", col("#59a52c"), 0.5, 0.0),
        fruit_tomato=principled("PR_tomato", col("#d8281c"), 0.25, 0.0, coat=0.6),
        bread=principled("PR_bread_crust", col("#b8742f"), 0.7, 0.0),
    )

# ----------------------------------------------------------------------------- mesh primitives
def bm_box(w, d, h, z0=0.0, cx=0.0, cy=0.0, bm=None):
    bm = bm or bmesh.new()
    x0, x1, y0, y1, z1 = cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2, z0 + h
    v = [bm.verts.new(p) for p in [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    out = [bm.faces.new([v[i] for i in f]) for f in faces]
    return bm, out

def box_uv(bm, faces, w, d, h, z0=0.0, cx=0.0, cy=0.0, scale=(1.0, 1.0)):
    uv = bm.loops.layers.uv.verify()
    bm.normal_update()
    su, sv = scale
    for f in faces:
        n = f.normal
        for l in f.loops:
            co = l.vert.co; x, y, z = co.x - cx, co.y - cy, co.z - z0
            if n.y < -0.5:   u, v = (x + w / 2) / w, z / h
            elif n.y > 0.5:  u, v = (w / 2 - x) / w, z / h
            elif n.x > 0.5:  u, v = (y + d / 2) / d, z / h
            elif n.x < -0.5: u, v = (d / 2 - y) / d, z / h
            elif n.z > 0.5:  u, v = (x + w / 2) / w, (y + d / 2) / d
            else:            u, v = (x + w / 2) / w, (d / 2 - y) / d
            l[uv].uv = (u * su, v * sv)

def box_mesh(name, w, d, h, mat, z0=0.0, uv=True):
    bm, faces = bm_box(w, d, h, z0)
    if uv: box_uv(bm, faces, w, d, h, z0)
    return bm_to_mesh(name, bm, [mat])

def bevel(bm, offset, segments=2, edges=None):
    bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
    bmesh.ops.bevel(bm, geom=(edges or bm.edges[:]), offset=offset, offset_type='OFFSET', segments=segments,
                    profile=0.5, affect='EDGES', clamp_overlap=True, loop_slide=True)

def bm_revolve(profile, mats, label_segs=(), n=32, bm=None, uv_h=None):
    """Revolve a (r, z) profile. mats[i] = material index of band i. Closed if r==0 at both ends.
    Label bands get v normalised 0..1 across their own z range; u=0.5 faces -Y (the shopper)."""
    bm = bm or bmesh.new()
    uv = bm.loops.layers.uv.verify()
    rings = []
    for (r, z) in profile:
        if r <= 1e-6: rings.append([bm.verts.new((0.0, 0.0, z))])
        else: rings.append([bm.verts.new((-r * math.sin(2 * math.pi * k / n), r * math.cos(2 * math.pi * k / n), z)) for k in range(n)])
    zmin, zmax = min(p[1] for p in profile), max(p[1] for p in profile)
    faces = []
    for i in range(len(profile) - 1):
        a, b = rings[i], rings[i + 1]
        za, zb = profile[i][1], profile[i + 1][1]
        lab = i in label_segs
        for k in range(n):
            k2 = (k + 1) % n
            u0, u1 = k / n, (k + 1) / n
            if len(a) == 1 and len(b) == 1: continue
            if len(a) == 1: verts, us = (a[0], b[k], b[k2]), ((u0 + u1) / 2, u0, u1)
            elif len(b) == 1: verts, us = (a[k], a[k2], b[0]), (u0, u1, (u0 + u1) / 2)
            else: verts, us = (a[k], a[k2], b[k2], b[k]), (u0, u1, u1, u0)
            try: f = bm.faces.new(verts)
            except ValueError: continue
            f.material_index = mats[i]
            for l, u in zip(f.loops, us):
                vz = l.vert.co.z
                v = (vz - za) / (zb - za) if (lab and abs(zb - za) > 1e-6) else (vz - zmin) / max(1e-6, (zmax - zmin))
                l[uv].uv = (u, v)
            faces.append(f)
    bmesh.ops.recalc_face_normals(bm, faces=faces)
    return bm, faces

# ----------------------------------------------------------------------------- product library
def label_path(s): return os.path.join(LABELS, f"{s['id']}.png")

def product_mesh(s):
    form, w, d, h = s["form"], s["w"], s["d"], s["h"]
    sid = s["id"]
    bg = col(s["bg"]); fg = col(s["fg"])
    foil = s["cat"] in ("snacks", "confectionery") or sid.startswith("cereal_") and form == "bag"
    if form == "box":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.5, coat=0.15)
        side = principled(f"MAT_{sid}_side", tuple(c * 0.85 for c in bg[:3]) + (1,), 0.55)
        bm, faces = bm_box(w, d, h); box_uv(bm, faces, w, d, h)
        for f in faces: f.material_index = 1 if abs(f.normal.y) > 0.5 else 0
        bevel(bm, min(0.003, 0.08 * d), 2)
        return bm_to_mesh(f"M_{sid}", bm, [side, label])
    if form == "flat":   # chocolate bar / butter: label on top, foil sides
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.35, coat=0.3)
        side = principled(f"MAT_{sid}_side", tuple(c * 0.8 for c in fg[:3]) + (1,), 0.35, 0.4)
        bm, faces = bm_box(w, d, h); box_uv(bm, faces, w, d, h)
        for f in faces: f.material_index = 1 if f.normal.z > 0.5 else 0
        bevel(bm, min(0.002, 0.2 * h), 2)
        return bm_to_mesh(f"M_{sid}", bm, [side, label])
    if form == "carton":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.45, coat=0.25)
        top = principled(f"MAT_{sid}_top", (0.95, 0.95, 0.95, 1), 0.45)
        bm, faces = bm_box(w, d, h); box_uv(bm, faces, w, d, h)
        for f in faces: f.material_index = 0 if abs(f.normal.z) > 0.5 else 1
        bevel(bm, 0.002, 2)
        # little screw cap on top, towards the front
        cap = principled(f"MAT_{sid}_cap", bg, 0.35)
        bm_revolve([(0, h), (0.012, h), (0.012, h + 0.012), (0, h + 0.012)], [2, 2, 2], n=16, bm=bm)
        for v in bm.verts:
            if v.co.z >= h - 1e-6 and (v.co.x ** 2 + v.co.y ** 2) < 0.0002: v.co.y -= d * 0.25
        return bm_to_mesh(f"M_{sid}", bm, [top, label, cap])
    if form == "bag":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=(0.3 if foil else 0.6), metal=(0.35 if foil else 0.0), coat=(0.2 if foil else 0.0))
        side = principled(f"MAT_{sid}_side", bg, (0.3 if foil else 0.6), (0.35 if foil else 0.0))
        bm, faces = bm_box(w, d, h); box_uv(bm, faces, w, d, h)
        for f in faces: f.material_index = 1 if abs(f.normal.y) > 0.5 else 0
        bevel(bm, min(0.45 * d, 0.12 * w, 0.03), 4)
        # pillow puff: push front/back centre outwards a little
        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            if abs(v.co.y) > d * 0.3:
                fall = (1 - (2 * v.co.x / w) ** 2) * (1 - (2 * (v.co.z - h / 2) / h) ** 2)
                v.co.y += math.copysign(0.25 * d * max(0, fall), v.co.y)
        return bm_to_mesh(f"M_{sid}", bm, [side, label])
    if form == "bottle_hh":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.35, coat=0.3)
        body = principled(f"MAT_{sid}_body", bg, 0.3, coat=0.3)
        cap = principled(f"MAT_{sid}_cap", col(s.get("cap", "#ffffff")), 0.3, coat=0.2)
        hb = h * 0.9
        bm, faces = bm_box(w, d, hb); box_uv(bm, faces, w, d, hb)
        for f in faces: f.material_index = 1 if abs(f.normal.y) > 0.5 else 0
        bevel(bm, min(0.35 * d, 0.012), 4)
        r = min(w, d) * 0.28
        bm_revolve([(0, hb - 0.002), (r, hb - 0.002), (r, h), (0, h)], [2, 2, 2], n=20, bm=bm)
        return bm_to_mesh(f"M_{sid}", bm, [body, label, cap])
    r = w / 2
    if form == "can":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.28, coat=0.35)
        metal = principled("MAT_can_metal", (0.85, 0.85, 0.87, 1), 0.28, 1.0)
        lip = 0.004
        prof = [(0, 0), (r - 0.004, 0), (r, lip), (r, h - lip - 0.003), (r - 0.001, h - 0.003), (r - 0.004, h), (0, h)]
        mats = [0, 0, 1, 0, 0, 0]
        bm, _ = bm_revolve(prof, mats, label_segs={2})
        return bm_to_mesh(f"M_{sid}", bm, [metal, label])
    if form == "jar":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.45)
        glass = principled("MAT_glass_clear", (1, 1, 1, 1), 0.05, transmission=1.0, ior=1.5)
        lid = principled(f"MAT_{sid}_lid", col(s.get("lid", "#c9a227")), 0.3, 0.8)
        content = principled(f"MAT_{sid}_content", col(s.get("content", "#b71c1c")), 0.6)
        P = [(0, 0), (0.85, 0), (1, 0.03), (1, 0.16), (1, 0.70), (1, 0.79), (0.9, 0.83), (0.93, 0.85), (0.93, 1.0), (0, 1.0)]
        prof = [(pr * r, pz * h) for pr, pz in P]
        mats = [1, 1, 1, 0, 1, 1, 2, 2, 2]
        bm, _ = bm_revolve(prof, mats, label_segs={3})
        inner = [(0, 0.012 * h), (0.9 * r, 0.012 * h), (0.9 * r, 0.8 * h), (0, 0.8 * h)]
        bm_revolve(inner, [3, 3, 3], n=24, bm=bm)
        return bm_to_mesh(f"M_{sid}", bm, [label, glass, lid, content])
    if form == "tub":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.4, coat=0.2)
        base = principled(f"MAT_{sid}_base", (0.95, 0.95, 0.95, 1), 0.4)
        lid = principled(f"MAT_{sid}_lid", col(s["accent"]) if s["cat"] != "confectionery" else col(s["bg"]), 0.35, 0.3 if s["cat"] == "confectionery" else 0.0)
        P = [(0, 0), (0.88, 0), (0.9, 0.05), (1.0, 0.88), (1.0, 0.9), (1.04, 0.9), (1.04, 1.0), (0, 1.0)]
        prof = [(pr * r, pz * h) for pr, pz in P]
        mats = [1, 1, 0, 1, 2, 2, 2]
        bm, _ = bm_revolve(prof, mats, label_segs={2})
        return bm_to_mesh(f"M_{sid}", bm, [label, base, lid])
    if form == "bottle":
        label = image_mat(f"MAT_{sid}_label", label_path(s), rough=0.4, coat=0.2)
        cap = principled(f"MAT_{sid}_cap", col(s.get("cap", "#ffffff")), 0.3, coat=0.2)
        if s["cat"] == "beer_wine":
            glass = principled(f"MAT_{sid}_glass", col(s.get("glass", "#1f2a1a")), 0.08, transmission=0.6, ior=1.5)
            P = [(0, 0), (0.7, 0), (1, 0.03), (1, 0.22), (1, 0.56), (1, 0.66), (0.78, 0.74), (0.5, 0.80), (0.36, 0.86), (0.36, 0.92), (0.38, 0.92), (0.38, 1.0), (0, 1.0)]
            mats = [1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2]
            label_segs = {3}
        else:
            liquid = s.get("liquid", "#e9f4ff")
            glass = principled(f"MAT_{sid}_pet", col(liquid), 0.06, transmission=1.0, ior=1.36)
            P = [(0, 0), (0.62, 0), (0.94, 0.02), (1.0, 0.08), (1.0, 0.30), (1.0, 0.62), (0.9, 0.74), (0.62, 0.84), (0.36, 0.89), (0.33, 0.905), (0.38, 0.915), (0.38, 1.0), (0, 1.0)]
            mats = [1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 2]
            label_segs = {4}
        prof = [(pr * r, pz * h) for pr, pz in P]
        bm, _ = bm_revolve(prof, mats, label_segs=label_segs)
        return bm_to_mesh(f"M_{sid}", bm, [label, glass, cap])
    raise ValueError(form)

def tag_mesh(s):
    m = image_mat(f"MAT_tag_{s['id']}", os.path.join(LABELS, f"tag_{s['id']}.png"), rough=0.4, coat=0.2)
    bm = bmesh.new(); uv = bm.loops.layers.uv.verify()
    w, h = 0.06, 0.03
    v = [bm.verts.new(p) for p in [(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, 0, h), (-w / 2, 0, h)]]
    f = bm.faces.new(v)
    for l, u in zip(f.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]): l[uv].uv = u
    return bm_to_mesh(f"M_tag_{s['id']}", bm, [m])

def build_library():
    t = time.time()
    for s in SKUS:
        LIB[s["id"]] = product_mesh(s)
        TAGMESH[s["id"]] = tag_mesh(s)
    STATS["library_s"] = round(time.time() - t, 1)
    print(f"library: {len(LIB)} product meshes in {STATS['library_s']} s")

# ----------------------------------------------------------------------------- placement
FACING = {"-X": (-1, 0), "+X": (1, 0), "-Y": (0, -1), "+Y": (0, 1)}
ROT_Z = {"-X": -math.pi / 2, "+X": math.pi / 2, "-Y": 0.0, "+Y": math.pi}

def shelf_frame(facing):
    """Returns (right, front) unit vectors: the shopper's right, and back-edge -> shopper direction."""
    fx, fy = FACING[facing]
    return Vector((-fy, fx, 0)), Vector((fx, fy, 0))

def place_products(collection, origin_xy, z, facing, width, depth, items, slot, tags_at=None, tag_coll=None, gap=0.012, front_gap=0.02):
    """Fill a shelf. origin_xy = centre of the shelf's back edge; products face `facing`."""
    right, front = shelf_frame(facing)
    rot = ROT_Z[facing]
    ox, oy = origin_xy
    base = Vector((ox, oy, z))
    total = sum(f * (BY_ID[sid]["w"] + gap) for sid, f in items)
    x = -min(total, width) / 2
    count = 0
    for sid, facings in items:
        s = BY_ID[sid]; w, d, h = s["w"], s["d"], s["h"]
        block_x0 = x
        rows = max(1, min(MAX_ROWS, int((depth - front_gap) // (d + 0.006))))
        for fi in range(facings):
            if x + w > width / 2 + 1e-6: break
            r = RNG.random()
            empty = r < 0.025
            missing_front = 0 if r > 0.20 else (2 if r < 0.06 else 1)
            stack = max(1, min(5, int(0.16 // h))) if s["form"] == "flat" else 1
            for ri in range(rows):
                if empty or ri < missing_front: continue
                for si in range(stack):
                    jx = RNG.uniform(-0.004, 0.004); jr = RNG.uniform(-0.03, 0.03) if s["form"] in ("can", "bottle", "jar", "tub") else RNG.uniform(-0.02, 0.02)
                    if RNG.random() < 0.02: jr += RNG.uniform(-0.25, 0.25)   # the odd knocked product
                    pos = base + right * (x + w / 2 + jx) + front * (depth - front_gap - (ri + 0.5) * (d + 0.006)) + Vector((0, 0, si * h))
                    name = f"{sid}.{slot}_f{fi}_r{ri}" + (f"_s{si}" if stack > 1 else "")
                    add_instance(name, LIB[sid], collection, loc=pos, rot=(0, 0, rot + jr),
                                 props=dict(sku=sid, category=s["cat"], brand=s["brand"], price=s["price"], slot=slot, facing=fi, row=ri))
                    count += 1
            x += w + gap
        # price tag centred under the block, on the ticket rail front
        if tags_at is not None and tag_coll is not None:
            cx = (block_x0 + x - gap) / 2
            tpos = base + right * cx + front * tags_at[0] + Vector((0, 0, tags_at[1]))
            add_instance(f"tag_{sid}.{slot}", TAGMESH[sid], tag_coll, loc=tpos, rot=(0, 0, rot), props=dict(sku=sid, slot=slot))
    return count

# ----------------------------------------------------------------------------- fixtures
def shelf_object(name, coll, back_xy, z, facing, width, depth, M, with_rail=True, thickness=None, props=None):
    """One shelf board with its ticket rail, front edge at back + depth. Origin = back-edge centre at top surface."""
    t = thickness or FIX["shelf_t"]
    bm, faces = bm_box(width, depth, t, z0=-t, cy=depth / 2)
    mats = [M["steel"]]
    if with_rail:
        _, rf = bm_box(width, 0.006, 0.042, z0=-0.036, cy=depth + 0.003, bm=bm)
        for f in rf: f.material_index = 1
        mats.append(M["rail"])
    me = bm_to_mesh(name, bm, mats)
    return add_obj(name, me, coll, loc=(back_xy[0], back_xy[1], z), rot=(0, 0, ROT_Z[facing] + math.pi), props=props)

def frame_object(name, coll, back_xy, facing, run_len, height, depths, M, double=False, sides=None):
    """Uprights, back panel, base decks and kick plates for a run. back_xy = centre of the back panel line."""
    bm = bmesh.new()
    bt = FIX["back_t"]; uw = FIX["upright_w"]
    # back panel (double-sided gondola: pegboard both faces)
    _, pf = bm_box(run_len, bt, height, z0=0.0, cy=0.0, bm=bm)
    for f in pf: f.material_index = 1
    # uprights at every bay boundary
    nb = int(round(run_len / BAY_W))
    for i in range(nb + 1):
        x = -run_len / 2 + i * BAY_W
        _, uf = bm_box(uw, bt + 0.02, height + 0.02, z0=0.0, cx=x, cy=0.0, bm=bm)
        for f in uf: f.material_index = 0
        # base feet supporting the deck
        for sgn in ((1, -1) if double else (1,)):
            _, ff = bm_box(uw, depths[0], FIX["base_z"] - 0.03, z0=0.0, cx=x, cy=sgn * (bt / 2 + depths[0] / 2), bm=bm)
            for f in ff: f.material_index = 2
    # base deck + kick plate per side
    for sgn in ((1, -1) if double else (1,)):
        _, df = bm_box(run_len, depths[0], 0.03, z0=FIX["base_z"] - 0.03, cy=sgn * (bt / 2 + depths[0] / 2), bm=bm)
        for f in df: f.material_index = 0
        _, kf = bm_box(run_len, 0.015, FIX["base_z"] - 0.03, z0=0.0, cy=sgn * (bt / 2 + depths[0] - 0.06), bm=bm)
        for f in kf: f.material_index = 2
        # ticket rail on the base deck
        _, rf = bm_box(run_len, 0.006, 0.042, z0=FIX["base_z"] - 0.036, cy=sgn * (bt / 2 + depths[0] + 0.003), bm=bm)
        for f in rf: f.material_index = 3
    # top cap
    _, tf = bm_box(run_len + uw, bt + 0.04, 0.03, z0=height, bm=bm)
    for f in tf: f.material_index = 0
    me = bm_to_mesh(name, bm, [M["steel"], M["pegboard"], M["kick"], M["rail"]])
    # frame's local +Y is the "front" side for a single-sided unit -> rotate so +Y points to `facing`
    rot = {"+Y": 0.0, "-X": math.pi / 2, "-Y": math.pi, "+X": -math.pi / 2}[facing]
    return add_obj(name, me, coll, loc=(back_xy[0], back_xy[1], 0), rot=(0, 0, rot), props=dict(fixture="gondola" if double else "wall_unit"))

def build_gondolas(M):
    t = time.time(); n = 0
    fx = coll("Fixtures"); pr = coll("Products"); tg = coll("PriceTags")
    run_len = PLAN["gondolas"][0]["n_bays"] * BAY_W
    for g in PLAN["gondolas"]:
        gx, y0 = g["x"], g["y0"]
        yc = y0 + run_len / 2
        gcoll = coll(f"Fixtures_{g['id']}", fx); pcoll = coll(f"Products_{g['id']}", pr); tcoll = coll(f"Tags_{g['id']}", tg)
        frame_object(f"{g['id']}_frame", gcoll, (gx, yc), "+X", run_len, FIX["height"], [FIX["base_depth"]], M, double=True)
        for side, facing in (("L", "-X"), ("R", "+X")):
            sx = gx + FACING[facing][0] * FIX["back_t"] / 2
            for bay in g["sides"][side]["bays"]:
                by = y0 + (bay["index"] + 0.5) * BAY_W
                slot_base = f"{g['id']}{side}_b{bay['index']}"
                for li, sh in enumerate(bay["shelves"]):
                    width = BAY_W - FIX["upright_w"] - 0.01
                    if li > 0:
                        shelf_object(f"{slot_base}_shelf{li}", gcoll, (sx, by), sh["z"], facing, width, sh["depth"], M,
                                     props=dict(fixture="shelf", gondola=g["id"], side=side, bay=bay["index"], level=li))
                    n += place_products(pcoll, (sx, by), sh["z"], facing, width, sh["depth"], sh["items"], f"{slot_base}_s{li}",
                                        tags_at=(sh["depth"] + 0.007, -0.033), tag_coll=tcoll)
        # end caps
    for ec in PLAN["endcaps"]:
        g = next(gg for gg in PLAN["gondolas"] if ec["id"].startswith(gg["id"] + "_"))
        gcoll = coll(f"Fixtures_{g['id']}", fx); pcoll = coll(f"Products_{g['id']}", pr); tcoll = coll(f"Tags_{g['id']}", tg)
        y_back = (g["y0"] - 0.03 - FIX["back_t"] / 2) if ec["facing"] == "-Y" else (g["y0"] + run_len + 0.03 + FIX["back_t"] / 2)
        frame_object(f"{ec['id']}_frame", gcoll, (ec["x"], y_back), ec["facing"], ec["w"], FIX["height"], [FIX["base_depth"]], M, double=False)
        sy = y_back + FACING[ec["facing"]][1] * FIX["back_t"] / 2
        for li, sh in enumerate(ec["shelves"]):
            width = ec["w"] - FIX["upright_w"] - 0.01
            if li > 0:
                shelf_object(f"{ec['id']}_shelf{li}", gcoll, (ec["x"], sy), sh["z"], ec["facing"], width, sh["depth"], M,
                             props=dict(fixture="shelf", gondola=g["id"], side="endcap", level=li))
            n += place_products(pcoll, (ec["x"], sy), sh["z"], ec["facing"], width, sh["depth"], sh["items"], f"{ec['id']}_s{li}",
                                tags_at=(sh["depth"] + 0.007, -0.033), tag_coll=tcoll)
        # header sign with the category on top of the end cap
        s = BY_ID[ec["sku"]]
        sign_plane(f"{ec['id']}_header", gcoll, os.path.join(SIGNS, f"cat_{s['cat']}.png"), 0.95, 0.26,
                   (ec["x"] + FACING[ec["facing"]][0] * 0.1, sy + FACING[ec["facing"]][1] * 0.1, FIX["height"] + 0.16), ec["facing"])
    STATS["gondola_products"] = n; STATS["gondolas_s"] = round(time.time() - t, 1)
    print(f"gondolas: {n} products in {STATS['gondolas_s']} s")

def build_wall_units(M):
    t = time.time(); n = 0
    fx = coll("Fixtures"); pr = coll("Products"); tg = coll("PriceTags")
    for wu in PLAN["wall_units"]:
        gcoll = coll(f"Fixtures_{wu['id']}", fx); pcoll = coll(f"Products_{wu['id']}", pr); tcoll = coll(f"Tags_{wu['id']}", tg)
        facing = wu["facing"]; fxv, fyv = FACING[facing]
        run_len = wu["n_bays"] * BAY_W
        along = Vector((-fyv, fxv, 0))  # direction bays advance (shopper's right)
        start = Vector((wu["x"], wu["y0"], 0))
        centre = start + along * (run_len / 2)
        frame_object(f"{wu['id']}_frame", gcoll, (centre.x, centre.y), facing, run_len, 2.2, [FIX["base_depth"]], M, double=False)
        for bay in wu["bays"]:
            bc = start + along * ((bay["index"] + 0.5) * BAY_W) + Vector((fxv, fyv, 0)) * (FIX["back_t"] / 2)
            slot_base = f"{wu['id']}_b{bay['index']}"
            for li, sh in enumerate(bay["shelves"]):
                width = BAY_W - FIX["upright_w"] - 0.01
                if li > 0:
                    shelf_object(f"{slot_base}_shelf{li}", gcoll, (bc.x, bc.y), sh["z"], facing, width, sh["depth"], M,
                                 props=dict(fixture="shelf", unit=wu["id"], bay=bay["index"], level=li))
                n += place_products(pcoll, (bc.x, bc.y), sh["z"], facing, width, sh["depth"], sh["items"], f"{slot_base}_s{li}",
                                    tags_at=(sh["depth"] + 0.007, -0.033), tag_coll=tcoll)
        cat = wu["cats"][0]
        sign_plane(f"{wu['id']}_header", gcoll, os.path.join(SIGNS, f"cat_{cat}.png" if cat != "bakery" else "bakery.png"), 2.0, 0.55,
                   (centre.x + fxv * 0.35, centre.y + fyv * 0.35, 2.75), facing)
    STATS["wall_products"] = n; print(f"wall units: {n} products in {round(time.time() - t, 1)} s")

def cabinet_object(name, coll, back_xy, facing, width, height, depth, M, doors=True):
    """Reach-in chiller (glass doors) or open multideck (doors=False). Local +Y = front."""
    bm = bmesh.new()
    wall = 0.05
    # body: back, two sides, top, bottom, plus a header band
    parts = [(width, wall, height, 0.0, wall / 2), (wall, depth, height, -(width - wall) / 2, depth / 2), (wall, depth, height, (width - wall) / 2, depth / 2),
             (width, depth, 0.10, 0.0, depth / 2), (width, depth, wall, 0.0, depth / 2)]
    for i, (w, d, h, cx, cy) in enumerate(parts):
        z0 = {3: 0.0, 4: height - wall}.get(i, 0.0)
        _, f = bm_box(w, d, h, z0=z0, cx=cx, cy=cy, bm=bm)
        for ff in f: ff.material_index = 0
    # inner back wall (lighter) + LED strips
    _, f = bm_box(width - 2 * wall, 0.01, height - 0.15, z0=0.10, cy=wall + 0.005, bm=bm)
    for ff in f: ff.material_index = 1
    _, f = bm_box(width - 2 * wall, 0.02, 0.02, z0=height - wall - 0.03, cy=depth - 0.06, bm=bm)
    for ff in f: ff.material_index = 3
    mats = [M["chiller_body"], M["chiller_inner"], M["glass"], M["led_soft"], M["alu"]]
    if doors:
        nd = max(1, int(round(width / 0.7)))
        dw = (width - 2 * wall) / nd
        for i in range(nd):
            cx = -(width - 2 * wall) / 2 + (i + 0.5) * dw
            _, f = bm_box(dw - 0.02, 0.008, height - 0.30, z0=0.16, cx=cx, cy=depth - 0.02, bm=bm)
            for ff in f: ff.material_index = 2
            for sx in (-1, 1):   # door frame verticals
                _, f = bm_box(0.02, 0.03, height - 0.28, z0=0.15, cx=cx + sx * (dw / 2 - 0.01), cy=depth - 0.02, bm=bm)
                for ff in f: ff.material_index = 4
            _, f = bm_box(0.02, 0.03, 0.5, z0=0.85, cx=cx + dw / 2 - 0.05, cy=depth + 0.01, bm=bm)   # handle
            for ff in f: ff.material_index = 4
        _, f = bm_box(width, 0.02, 0.28, z0=height - 0.30, cy=depth - 0.02, bm=bm)   # header band
        for ff in f: ff.material_index = 0
    else:
        _, f = bm_box(width, 0.35, 0.22, z0=height - 0.22, cy=depth - 0.175, bm=bm)   # canopy
        for ff in f: ff.material_index = 0
        _, f = bm_box(width - 2 * wall, 0.02, 0.03, z0=height - 0.25, cy=depth - 0.30, bm=bm)   # canopy LED
        for ff in f: ff.material_index = 3
    me = bm_to_mesh(name, bm, mats)
    rot = {"+Y": 0.0, "-X": math.pi / 2, "-Y": math.pi, "+X": -math.pi / 2}[facing]
    return add_obj(name, me, coll, loc=(back_xy[0], back_xy[1], 0), rot=(0, 0, rot), props=dict(fixture="chiller"))

def build_chillers(M):
    t = time.time(); n = 0
    fx = coll("Fixtures"); pr = coll("Products"); tg = coll("PriceTags")
    for ch in PLAN["chillers"]:
        gcoll = coll(f"Fixtures_{ch['id']}", fx); pcoll = coll(f"Products_{ch['id']}", pr); tcoll = coll(f"Tags_{ch['id']}", tg)
        facing = ch["facing"]; fxv, fyv = FACING[facing]
        doors = ch["id"] == "C_right"
        height, depth = (2.1, 0.8) if doors else (2.0, 0.9)
        along = Vector((-fyv, fxv, 0)); start = Vector((ch["x"], ch["y0"], 0))
        run_len = ch["n_bays"] * BAY_W
        # one cabinet per bay
        for bay in ch["bays"]:
            bc = start + along * ((bay["index"] + 0.5) * BAY_W)
            cabinet_object(f"{ch['id']}_cab{bay['index']}", gcoll, (bc.x, bc.y), facing, BAY_W, height, depth, M, doors=doors)
            inner_back = bc + Vector((fxv, fyv, 0)) * 0.06
            slot_base = f"{ch['id']}_b{bay['index']}"
            for li, sh in enumerate(bay["shelves"]):
                width = BAY_W - 0.14
                shelf_object(f"{slot_base}_shelf{li}", gcoll, (inner_back.x, inner_back.y), sh["z"], facing, width, sh["depth"], M, with_rail=True, thickness=0.02,
                             props=dict(fixture="shelf", unit=ch["id"], bay=bay["index"], level=li))
                n += place_products(pcoll, (inner_back.x, inner_back.y), sh["z"], facing, width, sh["depth"], sh["items"], f"{slot_base}_s{li}",
                                    tags_at=(sh["depth"] + 0.007, -0.033), tag_coll=tcoll)
        centre = start + along * (run_len / 2)
        sign_plane(f"{ch['id']}_header", gcoll, os.path.join(SIGNS, "dairy.png" if doors else "drinks.png"), 2.2, 0.6,
                   (centre.x + fxv * 0.5, centre.y + fyv * 0.5, height + 0.5), facing)
    STATS["chiller_products"] = n; print(f"chillers: {n} products in {round(time.time() - t, 1)} s")

# ----------------------------------------------------------------------------- signage
def sign_plane(name, coll_, path, w, h, loc, facing, double=True, emit=0.0, hang=0.0):
    """Vertical image panel facing `facing` (or a pair of cables when hang>0)."""
    m = image_mat(f"MAT_sign_{os.path.basename(path)}", path, rough=0.45, emit_str=emit)
    bm = bmesh.new(); uv = bm.loops.layers.uv.verify()
    for side in ((1,) if not double else (1, -1)):
        y = -0.005 * side
        v = [bm.verts.new(p) for p in [(-w / 2, y, -h / 2), (w / 2, y, -h / 2), (w / 2, y, h / 2), (-w / 2, y, h / 2)]]
        f = bm.faces.new(v if side == 1 else list(reversed(v)))
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)] if side == 1 else [(1, 1), (0, 1), (0, 0), (1, 0)]
        for l, u in zip(f.loops, uvs): l[uv].uv = u
    mats = [m]
    if hang > 0:
        for sx in (-1, 1):
            _, cf = bm_box(0.006, 0.006, hang, z0=h / 2, cx=sx * w * 0.4, bm=bm)
            for ff in cf: ff.material_index = 1
        mats.append(principled("FX_cable", col("#222222"), 0.5, 0.3))
    me = bm_to_mesh(name, bm, mats)
    return add_obj(name, me, coll_, loc=loc, rot=(0, 0, ROT_Z[facing]), props=dict(fixture="sign"))

def build_signage(M):
    sg = coll("Signage")
    for a in PLAN["aisles"]:
        sign_plane(f"aisle_sign_{a['no']}", sg, os.path.join(SIGNS, f"aisle_{a['no']}.png"), 1.2, 0.48, (a["x"], a["y"], 3.0), "-Y", hang=H_CEIL - 3.0 - 0.24)
    sign_plane("sign_welcome", sg, os.path.join(SIGNS, "welcome.png"), 4.8, 1.0, (9.5, -D2 + 1.9, 3.2), "-Y", hang=0.3)
    sign_plane("sign_store", sg, os.path.join(SIGNS, "store_sign.png"), 6.0, 1.3, (5.0, D2 - 0.12, 3.25), "-Y", double=False)
    sign_plane("sign_exit", sg, os.path.join(SIGNS, "exit.png"), 1.5, 0.5, (-9.5, -D2 + 1.0, 3.1), "+Y", hang=0.4)
    sign_plane("sign_entrance", sg, os.path.join(SIGNS, "entrance.png"), 1.5, 0.5, (9.5, -D2 + 0.6, 3.1), "-Y", hang=0.4)
    sign_plane("sign_produce", sg, os.path.join(SIGNS, "produce.png"), 2.4, 0.65, (9.0, -D2 + 3.7, 3.0), "-Y", hang=0.8)
    sign_plane("sign_bakery", sg, os.path.join(SIGNS, "bakery.png"), 2.4, 0.65, (-8.0, 7.2, 3.0), "-Y", hang=0.8)
    for i in range(4):
        sign_plane(f"lane_sign_{i + 1}", sg, os.path.join(SIGNS, f"lane_{i + 1}.png"), 0.4, 0.5, (-5.6 + i * 2.3 - 0.4, -D2 + 3.5 - 2.6, 2.62), "+Y", hang=0.0)

# ----------------------------------------------------------------------------- structure
def build_structure(M):
    st = coll("Structure")
    floor = polyhaven_mat("floor_tiles_08", 1.0)
    wallm = polyhaven_mat("white_plaster_02", 1.0)
    # floor with UVs in metres/2 (one texture repeat = 2 m)
    bm = bmesh.new(); uv = bm.loops.layers.uv.verify()
    v = [bm.verts.new(p) for p in [(-W2, -D2, 0), (W2, -D2, 0), (W2, D2, 0), (-W2, D2, 0)]]
    f = bm.faces.new(v)
    for l in f.loops: l[uv].uv = ((l.vert.co.x + W2) / 2.0, (l.vert.co.y + D2) / 2.0)
    add_obj("Floor", bm_to_mesh("Floor", bm, [floor]), st, props=dict(structure="floor"))
    # exterior ground
    bm = bmesh.new(); _, gf = bm_box(120, 120, 0.02, z0=-0.03, bm=bm)
    add_obj("Ground_exterior", bm_to_mesh("Ground_exterior", bm, [M["concrete"]]), st)
    # ceiling with tile grid material
    cm = M["ceiling"]
    nt = cm.node_tree; bsdf = nt.nodes["Principled BSDF"]
    brick = nt.nodes.new("ShaderNodeTexBrick"); tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
    brick.inputs["Scale"].default_value = 1.0; brick.inputs["Mortar Size"].default_value = 0.01; brick.offset = 0.0
    brick.inputs["Color1"].default_value = (0.92, 0.92, 0.9, 1); brick.inputs["Color2"].default_value = (0.92, 0.92, 0.9, 1); brick.inputs["Mortar"].default_value = (0.5, 0.5, 0.5, 1)
    brick.inputs["Brick Width"].default_value = 0.6; brick.inputs["Row Height"].default_value = 0.6
    mp.inputs["Scale"].default_value = (1, 1, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], brick.inputs["Vector"]); nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    bm = bmesh.new(); _, cf = bm_box(2 * W2, 2 * D2, 0.05, z0=H_CEIL, bm=bm)
    add_obj("Ceiling", bm_to_mesh("Ceiling", bm, [cm]), st, props=dict(structure="ceiling"))
    # walls (0.25 thick) with UVs; front wall = glazed shopfront
    wt = 0.25
    def wall(name, w, h, loc, rot, mat=wallm):
        bm, faces = bm_box(w, wt, h, z0=0.0); box_uv(bm, faces, w, wt, h, scale=(w / 2.0, h / 2.0))   # one repeat per 2 m
        return add_obj(name, bm_to_mesh(name, bm, [mat]), st, loc=loc, rot=(0, 0, rot), props=dict(structure="wall"))
    wall("Wall_back", 2 * W2 + 2 * wt, H_CEIL + 0.3, (0, D2 + wt / 2, 0), 0)
    wall("Wall_left", 2 * D2, H_CEIL + 0.3, (-W2 - wt / 2, 0, 0), math.pi / 2)
    wall("Wall_right", 2 * D2, H_CEIL + 0.3, (W2 + wt / 2, 0, 0), math.pi / 2)
    # front: sill wall, header wall, glazing + mullions, two door openings (entrance right, exit left)
    wall("Wall_front_sill", 2 * W2 + 2 * wt, 0.45, (0, -D2 - wt / 2, 0), 0)
    wall("Wall_front_head", 2 * W2 + 2 * wt, 0.8 + 0.3, (0, -D2 - wt / 2, 3.2), 0)
    bm = bmesh.new(); mats = [M["glass"], M["alu"]]
    openings = [(-11.0, -8.0), (8.0, 11.0)]
    x = -W2
    while x < W2 - 1e-6:
        x1 = min(x + 1.0, W2)
        if not any(o0 - 1e-6 <= x and x1 <= o1 + 1e-6 for o0, o1 in openings):
            _, gf = bm_box(x1 - x - 0.06, 0.01, 2.75, z0=0.45, cx=(x + x1) / 2, cy=-D2 - 0.05, bm=bm)
            for f in gf: f.material_index = 0
        _, mf = bm_box(0.06, 0.12, 2.75, z0=0.45, cx=x, cy=-D2 - 0.05, bm=bm)
        for f in mf: f.material_index = 1
        x = x1
    _, mf = bm_box(2 * W2, 0.12, 0.08, z0=3.12, cy=-D2 - 0.05, bm=bm); [setattr(f, "material_index", 1) for f in mf]
    _, mf = bm_box(2 * W2, 0.12, 0.08, z0=0.45, cy=-D2 - 0.05, bm=bm); [setattr(f, "material_index", 1) for f in mf]
    for o0, o1 in openings:   # door frames + open sliding leaves parked at the sides
        for cx in (o0, o1):
            _, mf = bm_box(0.12, 0.2, 3.2 - 0.45, z0=0.45, cx=cx, cy=-D2 - 0.05, bm=bm); [setattr(f, "material_index", 1) for f in mf]
        _, mf = bm_box(o1 - o0, 0.2, 0.25, z0=2.95, cx=(o0 + o1) / 2, cy=-D2 - 0.05, bm=bm); [setattr(f, "material_index", 1) for f in mf]
        for cx in (o0 + 0.42, o1 - 0.42):
            _, gf = bm_box(0.76, 0.02, 2.4, z0=0.05, cx=cx, cy=-D2 + 0.12, bm=bm); [setattr(f, "material_index", 0) for f in gf]
            _, mf = bm_box(0.8, 0.04, 0.05, z0=2.45, cx=cx, cy=-D2 + 0.12, bm=bm); [setattr(f, "material_index", 1) for f in mf]
        # floor mat + entry gates
    add_obj("Shopfront_glazing", bm_to_mesh("Shopfront_glazing", bm, mats), st, props=dict(structure="glazing"))
    # ground-level door sills are open: leave the floor as is. Baseboard strip along walls:
    bm = bmesh.new()
    for (w, d, cx, cy) in [(2 * W2, 0.02, 0, D2 - 0.01), (0.02, 2 * D2, -W2 + 0.01, 0), (0.02, 2 * D2, W2 - 0.01, 0)]:
        bm_box(w, d, 0.1, z0=0, cx=cx, cy=cy, bm=bm)
    add_obj("Baseboard", bm_to_mesh("Baseboard", bm, [M["kick"]]), st)

# ----------------------------------------------------------------------------- front of store, produce, bakery, props
def build_front(M):
    fr = coll("Fixtures"); fc = coll("Fixtures_Front", fr); pc = coll("Products_Front", coll("Products"))
    # checkouts: 4 lanes along Y, customers walk -Y (toward the front wall)
    for i in range(4):
        x = -5.6 + i * 2.3; y = -D2 + 3.5
        bm = bmesh.new()
        parts = [  # (w, d, h, z0, cx, cy, mat)
            (0.6, 2.2, 0.85, 0, 0, 0.4, 0),          # belt counter body
            (0.56, 2.0, 0.02, 0.85, 0, 0.4, 1),      # belt
            (0.9, 0.9, 0.85, 0, 0.0, -1.2, 0),       # cashier / scanner station
            (0.6, 1.2, 0.85, 0, 0.0, -2.3, 0),       # bagging area
            (0.02, 3.6, 0.35, 0.85, 0.31, -0.3, 2),  # divider rail
            (0.3, 0.3, 0.05, 0.85, 0.15, -1.2, 3),   # scanner glass
            (0.06, 0.06, 1.2, 0.85, -0.35, -1.5, 2), # pos pole
            (0.35, 0.02, 0.26, 1.85, -0.35, -1.5, 4),# pos screen
            (0.12, 0.08, 0.06, 0.85, 0.3, -1.6, 5),  # card terminal
            (0.05, 0.05, 1.6, 0.85, -0.4, -2.6, 2),  # lane light pole
        ]
        mats = [M["counter"], M["rubber"], M["alu"], M["glass"], M["screen"], M["plastic_black"]]
        for (w, d, h, z0, cx, cy, mi) in parts:
            _, f = bm_box(w, d, h, z0=z0, cx=cx, cy=cy, bm=bm)
            for ff in f: ff.material_index = mi
        add_obj(f"Checkout_{i + 1}", bm_to_mesh(f"Checkout_{i + 1}", bm, mats), fc, loc=(x, y, 0), props=dict(fixture="checkout", lane=i + 1))
        # impulse shelf beside the lane with sweets
        bx = x + 1.15
        frame_object(f"Impulse_{i + 1}_frame", fc, (bx, y - 0.3), "+X", 1.0, 1.4, [0.3], M, double=False)
        for li, z in enumerate((0.17, 0.5, 0.8, 1.1)):
            if li > 0: shelf_object(f"Impulse_{i + 1}_shelf{li}", fc, (bx + FIX["back_t"] / 2, y - 0.3), z, "+X", 0.94, 0.3, M)
            items = [["candy_gummy", 3], ["choc_milk", 2], ["candy_mints", 4]] if li % 2 == 0 else [["choc_dark", 2], ["tortilla", 1], ["cola_can", 3]]
            place_products(pc, (bx + FIX["back_t"] / 2, y - 0.3), z, "+X", 0.94, 0.3, items, f"impulse{i + 1}_s{li}")
    # shopping baskets stack + carts near the entrance
    basket_me = None
    for k in range(8):
        if basket_me is None:
            bm, f = bm_box(0.45, 0.32, 0.22, z0=0); bevel(bm, 0.02, 3)
            basket_me = bm_to_mesh("M_basket", bm, [M["plastic_red"]])
        add_obj(f"Basket_{k}", basket_me, fc, loc=(11.6, -D2 + 1.2, 0.05 + k * 0.045), rot=(0, 0, math.pi / 2), props=dict(prop="basket"))
    for k in range(4):
        cart_object(f"Cart_{k}", fc, (12.3, -D2 + 2.6 + k * 0.45), 0.0, M)
    # promo pallets with cases
    for k, (px, py, sid) in enumerate([(-9.0, -D2 + 3.0, "cola_15l"), (0.0, 7.2, "water_15l"), (4.0, 7.2, "beer_can"), (-4.0, 7.2, "chips_salt")]):
        pallet_object(f"Pallet_{k}", fc, (px, py), sid, M, pc)
    # dump bin of chips in the front cross aisle
    bm = bmesh.new()
    for cx, cy, w, d in [(0, -0.5, 1.2, 0.04), (0, 0.5, 1.2, 0.04), (-0.6, 0, 0.04, 1.0), (0.6, 0, 0.04, 1.0)]:
        _, f = bm_box(w, d, 0.7, z0=0, cx=cx, cy=cy, bm=bm)
    _, f = bm_box(1.2, 1.0, 0.03, z0=0.4, bm=bm)
    add_obj("DumpBin", bm_to_mesh("DumpBin", bm, [M["steel_dark"]]), fc, loc=(4.5, -5.0, 0), props=dict(fixture="dump_bin"))
    for k in range(18):
        sid = RNG.choice(["chips_salt", "chips_paprika", "chips_cheese", "tortilla"])
        add_instance(f"{sid}.dump_{k}", LIB[sid], pc, loc=(4.5 + RNG.uniform(-0.45, 0.45), -5.0 + RNG.uniform(-0.35, 0.35), 0.43 + (k // 9) * 0.08),
                rot=(RNG.uniform(-0.3, 0.3), RNG.uniform(-1.4, -1.0), RNG.uniform(0, 6.28)), props=dict(sku=sid, category="snacks", slot="dump_bin"))
    # entry gates
    for gx in (8.3, 10.7):
        bm = bmesh.new(); bm_box(0.08, 0.3, 1.0, z0=0, bm=bm); bm_box(0.9, 0.03, 0.03, z0=0.9, cx=0.45 if gx < 9 else -0.45, bm=bm)
        add_obj(f"Gate_{gx}", bm_to_mesh(f"Gate_{gx}", bm, [M["alu"]]), fc, loc=(gx, -D2 + 1.5, 0), props=dict(fixture="gate"))

def cart_object(name, coll_, xy, rot, M):
    bm = bmesh.new()
    # basket: open box made of a wire grid
    w, d, h, z0 = 0.55, 0.9, 0.45, 0.55
    grid = bmesh.new()
    _, faces = bm_box(w, d, h, z0=z0, cy=0.05, bm=grid)
    for f in faces:
        if f.normal.z > 0.5: grid.faces.remove(f)   # open top
    bmesh.ops.subdivide_edges(grid, edges=grid.edges[:], cuts=7, use_grid_fill=True)
    bmesh.ops.wireframe(grid, faces=grid.faces[:], thickness=0.006, use_even_offset=True, use_replace=True)
    for f in grid.faces: f.material_index = 0
    grid_me = bpy.data.meshes.new("tmp_grid"); grid.to_mesh(grid_me); grid.free()
    bm.from_mesh(grid_me); bpy.data.meshes.remove(grid_me)
    # chassis, handle, wheels
    for (bw, bd, bh, bz, cx, cy) in [(0.5, 0.85, 0.02, 0.15, 0, 0.05), (0.03, 0.03, 0.55, 0.15, -0.24, 0.45), (0.03, 0.03, 0.55, 0.15, 0.24, 0.45),
                                     (0.03, 0.03, 0.55, 0.15, -0.24, -0.35), (0.03, 0.03, 0.55, 0.15, 0.24, -0.35), (0.6, 0.03, 0.03, 1.0, 0, -0.45)]:
        _, f = bm_box(bw, bd, bh, z0=bz, cx=cx, cy=cy, bm=bm)
        for ff in f: ff.material_index = 0
    for cx, cy in [(-0.22, 0.4), (0.22, 0.4), (-0.22, -0.3), (0.22, -0.3)]:
        wb = bmesh.new(); bm_revolve([(0, 0), (0.06, 0), (0.06, 0.03), (0, 0.03)], [1, 1, 1], n=16, bm=wb)
        for v in wb.verts: v.co = (cx + v.co.z - 0.015, cy + v.co.x, 0.06 + v.co.y)   # axle along X, wheel touching the floor
        wm = bpy.data.meshes.new("tmp_wheel"); wb.to_mesh(wm); wb.free(); bm.from_mesh(wm); bpy.data.meshes.remove(wm)
    for f in bm.faces:
        if f.calc_center_median().z < 0.14: f.material_index = 1
    return add_obj(name, bm_to_mesh(name, bm, [M["chrome"], M["rubber"]]), coll_, loc=(xy[0], xy[1], 0), rot=(0, 0, rot), props=dict(prop="cart"))

def pallet_object(name, coll_, xy, sid, M, pcoll):
    bm = bmesh.new()
    for k in range(7):
        bm_box(0.14, 1.2, 0.02, z0=0.12, cx=-0.5 + k * 0.166, bm=bm)
    for cy in (-0.55, 0, 0.55):
        bm_box(1.0, 0.1, 0.1, z0=0.02, cy=cy, bm=bm); bm_box(1.0, 0.1, 0.02, z0=0, cy=cy, bm=bm)
    add_obj(name, bm_to_mesh(name, bm, [M["wood_light"]]), coll_, loc=(xy[0], xy[1], 0), props=dict(fixture="pallet"))
    s = BY_ID[sid]
    # stacked shrink-wrapped cases: a box with the product label on its long sides
    cw, cd, chh = (s["w"] * 3 + 0.02, s["d"] * 2 + 0.02, s["h"] + 0.01) if s["form"] != "bag" else (0.5, 0.35, 0.3)
    label = image_mat(f"MAT_{sid}_label", label_path(s))
    case = bpy.data.meshes.get(f"M_case_{sid}")
    if case is None:
        cb, faces = bm_box(cw, cd, chh); box_uv(cb, faces, cw, cd, chh)
        for f in faces: f.material_index = 1 if abs(f.normal.y) > 0.5 else 0
        case = bm_to_mesh(f"M_case_{sid}", cb, [principled(f"MAT_{sid}_case", (0.85, 0.85, 0.85, 1), 0.2, coat=0.6), label])
    nx, ny = max(1, int(1.0 // cw)), max(1, int(1.2 // cd))
    layers = 3 if s["form"] != "bag" else 2
    n = 0
    for lz in range(layers):
        for ix in range(nx):
            for iy in range(ny):
                if lz == layers - 1 and RNG.random() < 0.3: continue
                add_instance(f"case_{sid}.{name}_{n}", case, pcoll, loc=(xy[0] - (nx - 1) * cw / 2 + ix * cw, xy[1] - (ny - 1) * cd / 2 + iy * cd, 0.14 + lz * chh),
                        rot=(0, 0, RNG.uniform(-0.02, 0.02)), props=dict(sku=sid, category=s["cat"], slot=name, case=1))
                n += 1

def build_produce_bakery(M):
    fc = coll("Fixtures_Produce", coll("Fixtures")); pc = coll("Products_Produce", coll("Products"))
    fruit = [("apple_red", M["fruit_apple_red"], 0.038), ("apple_green", M["fruit_apple_green"], 0.037), ("orange", M["fruit_orange"], 0.04),
             ("lemon", M["fruit_lemon"], 0.032), ("lime", M["fruit_lime"], 0.028), ("tomato", M["fruit_tomato"], 0.035)]
    fruit_mesh = {}
    for nm, mat, r in fruit:
        bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=2, radius=r)
        for v in bm.verts: v.co.z *= 0.92
        me = bm_to_mesh(f"M_fruit_{nm}", bm, [mat])
        for p in me.polygons: p.use_smooth = True
        fruit_mesh[nm] = me
    crate_me = None
    for ti, (tx, ty) in enumerate([(7.5, -D2 + 2.6), (10.2, -D2 + 2.6), (7.5, -D2 + 4.8), (10.2, -D2 + 4.8)]):
        bm = bmesh.new()
        bm_box(2.0, 1.2, 0.04, z0=0.82, bm=bm)
        for cx, cy in [(-0.9, -0.5), (0.9, -0.5), (-0.9, 0.5), (0.9, 0.5)]: bm_box(0.05, 0.05, 0.82, z0=0, cx=cx, cy=cy, bm=bm)
        bm_box(2.0, 1.2, 0.03, z0=0.25, bm=bm)
        add_obj(f"ProduceTable_{ti}", bm_to_mesh(f"ProduceTable_{ti}", bm, [M["wood"]]), fc, loc=(tx, ty, 0), props=dict(fixture="produce_table"))
        for ci, (cx, cy) in enumerate([(-0.5, -0.3), (0.5, -0.3), (-0.5, 0.3), (0.5, 0.3)]):
            if crate_me is None:
                cb = bmesh.new()
                bm_box(0.6, 0.4, 0.02, z0=0, bm=cb)
                for (w, d, ox, oy) in [(0.6, 0.02, 0, -0.19), (0.6, 0.02, 0, 0.19), (0.02, 0.4, -0.29, 0), (0.02, 0.4, 0.29, 0)]: bm_box(w, d, 0.15, z0=0, cx=ox, cy=oy, bm=cb)
                crate_me = bm_to_mesh("M_crate", cb, [M["wood_light"]])
            add_obj(f"Crate_{ti}_{ci}", crate_me, fc, loc=(tx + cx, ty + cy, 0.86), props=dict(fixture="crate"))
            nm, mat, r = fruit[(ti * 4 + ci) % len(fruit)]
            k = 0
            for layer in range(2):
                for ix in range(int(0.56 // (2 * r))):
                    for iy in range(int(0.36 // (2 * r))):
                        if layer == 1 and RNG.random() < 0.35: continue
                        add_instance(f"{nm}.crate{ti}_{ci}_{k}", fruit_mesh[nm], pc,
                                loc=(tx + cx - 0.28 + r + ix * 2 * r + RNG.uniform(-0.004, 0.004), ty + cy - 0.18 + r + iy * 2 * r + RNG.uniform(-0.004, 0.004), 0.88 + r * 0.92 + layer * 1.7 * r),
                                rot=(RNG.uniform(0, 6.28), RNG.uniform(0, 6.28), RNG.uniform(0, 6.28)), props=dict(sku=nm, category="produce", slot=f"crate{ti}_{ci}"))
                        k += 1
    # bakery counter (glass display) in the back-left corner
    bm = bmesh.new(); mats = [M["counter"], M["glass"], M["wood"]]
    _, f = bm_box(2.4, 0.8, 0.9, z0=0, bm=bm); [setattr(x, "material_index", 0) for x in f]
    _, f = bm_box(2.4, 0.7, 0.5, z0=0.9, cy=0.05, bm=bm); [setattr(x, "material_index", 1) for x in f]
    _, f = bm_box(2.4, 0.6, 0.02, z0=1.1, bm=bm); [setattr(x, "material_index", 2) for x in f]
    add_obj("BakeryCounter", bm_to_mesh("BakeryCounter", bm, mats), fc, loc=(-7.5, 8.4, 0), props=dict(fixture="bakery_counter"))
    bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.05)
    for v in bm.verts: v.co.x *= 1.6; v.co.z *= 0.7
    roll = bm_to_mesh("M_bread_roll", bm, [M["bread"]])
    for p in roll.polygons: p.use_smooth = True
    for k in range(40):
        add_instance(f"bread_roll.{k}", roll, pc, loc=(-7.5 - 1.0 + (k % 10) * 0.22, 8.4 - 0.2 + (k // 10) * 0.16, 0.95 + (0.05 if k // 10 in (1, 2) else 0.04)),
                rot=(0, 0, RNG.uniform(0, 3.14)), props=dict(sku="bread_roll", category="bakery"))

# ----------------------------------------------------------------------------- lighting, cameras, render settings
def build_lighting(M):
    lc = coll("Lighting")
    xs = [-12 + 3 * i for i in range(9)]; ys = [-D2 + 1.5 + 3 * j for j in range(int((2 * D2 - 1.5) // 3) + 1)]
    panel = None
    ld = bpy.data.lights.new("LED_panel_light", 'AREA'); ld.shape = 'RECTANGLE'; ld.size = 1.2; ld.size_y = 0.6; ld.energy = 45; ld.color = (1.0, 0.97, 0.93)
    ld.shadow_soft_size = 0.5
    for x in xs:
        for y in ys:
            if panel is None:
                bm = bmesh.new(); _, f = bm_box(1.2, 0.6, 0.02, z0=-0.02, bm=bm)
                panel = bm_to_mesh("M_led_panel", bm, [M["led"]])
            add_obj(f"LEDPanel_{x}_{y}", panel, lc, loc=(x, y, H_CEIL - 0.001), props=dict(fixture="light_panel"))
            add_obj(f"LEDLight_{x}_{y}", ld, lc, loc=(x, y, H_CEIL - 0.03))
    # world already holds the Poly Haven HDRI; keep it modest so it does not overpower the interior
    w = bpy.context.scene.world
    if w and w.node_tree:
        bgn = next((n for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"), None)
        if bgn: bgn.inputs["Strength"].default_value = 1.0

def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

def build_cameras():
    cc = coll("Cameras")
    def cam(name, loc, target, lens=24):
        cd = bpy.data.cameras.new(name); cd.lens = lens; cd.clip_end = 300
        ob = add_obj(name, cd, cc, loc=loc); look_at(ob, target); return ob
    cam("Cam_Entrance", (9.5, -D2 + 1.0, 1.65), (2.0, 2.0, 1.2), 22)
    cam("Cam_Aisle", (-1.525, -5.6, 1.6), (-1.525, 6.0, 1.1), 28)
    cam("Cam_Shelf", (-3.05 - 0.6 - 1.4, 0.3, 1.45), (-3.05 - 0.6, 0.5, 1.0), 35)
    cam("Cam_Checkout", (3.0, -4.2, 1.7), (-3.0, -D2 + 2.5, 1.0), 26)
    cam("Cam_Produce", (5.0, -D2 + 1.2, 1.7), (9.5, -D2 + 4.5, 0.9), 26)
    cam("Cam_Overview", (-12.3, -D2 + 0.5, 3.7), (3.0, 5.0, 0.4), 14)
    cam("Cam_Exterior", (17.0, -24.0, 9.0), (0.0, 0.5, 1.0), 24)
    td = cam("Cam_TopDown", (0.0, 0.0, 60.0), (0.0, 0.0, 0.0), 24)
    td.data.type = 'ORTHO'; td.data.ortho_scale = 2 * W2 + 2; td.rotation_euler = (0, 0, 0)
    # walkthrough camera: entrance -> front aisle -> aisle 4 -> back aisle -> aisle 6 -> checkouts -> exit
    yf, yb = -5.0, 6.2
    wp = [(9.5, -D2 + 1.0), (9.0, -D2 + 3.2), (4.0, yf), (-1.525, yf), (-1.525, 0.0), (-1.525, yb), (2.0, yb + 0.5), (4.575, yb), (4.575, 0.0), (4.575, yf), (0.0, yf - 0.6), (-2.6, -D2 + 4.0), (-2.6, -D2 + 1.6), (-4.5, -D2 + 0.9), (-8.5, -D2 + 0.9), (-9.5, -D2 - 0.6), (-9.5, -D2 - 3.0)]
    walk = cam("Cam_Walk", (wp[0][0], wp[0][1], 1.62), (wp[1][0], wp[1][1], 1.3), 24)
    sc = bpy.context.scene; sc.frame_start = 1; sc.frame_end = 30 * len(wp)
    for i, (x, y) in enumerate(wp):
        f = 1 + i * 30
        walk.location = (x, y, 1.62 + 0.01 * math.sin(i))
        nx, ny = wp[min(i + 1, len(wp) - 1)]
        px, py = wp[max(i - 1, 0)]
        dx, dy = (nx - x, ny - y) if i < len(wp) - 1 else (x - px, y - py)
        L = math.hypot(dx, dy) or 1.0
        look_at(walk, (x + dx / L * 4.0, y + dy / L * 4.0, 1.38))   # level gaze 4 m ahead
        walk.keyframe_insert("location", frame=f); walk.keyframe_insert("rotation_euler", frame=f)
    for fc in walk.animation_data.action.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = 'BEZIER'; kp.easing = 'AUTO'
    sc.camera = bpy.data.objects["Cam_Entrance"]

def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'GPU'; sc.cycles.samples = 256; sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPTIX'
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
        for d in prefs.devices: d.use = d.type != 'CPU'
    except Exception as e: print("cycles prefs:", e)
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.fps = 24
    sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'; sc.view_settings.exposure = -0.3
    try:
        sc.eevee.use_raytracing = True; sc.eevee.use_shadows = True; sc.eevee.taa_samples = 16
        sc.eevee.ray_tracing_options.resolution_scale = '2'
    except Exception as e: print("eevee:", e)
    sc.render.film_transparent = False
    # walk navigation defaults for roaming in the viewport
    try:
        wn = bpy.context.preferences.inputs.walk_navigation
        wn.view_height = 1.65; wn.walk_speed = 2.5; wn.use_gravity = True; wn.jump_height = 0.4; wn.mouse_speed = 1.0
    except Exception as e: print("walk prefs:", e)
    # viewport: material preview, far clip
    try:
        screens = [w.screen for w in bpy.context.window_manager.windows] if bpy.context.window_manager else []
        areas = [a for scr in screens for a in scr.areas]
        for area in areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D':
                        sp.shading.type = 'MATERIAL'; sp.clip_end = 400; sp.clip_start = 0.02
                        sp.shading.use_scene_lights = True; sp.shading.use_scene_world = True
                        sp.overlay.show_relationship_lines = False; sp.overlay.show_floor = False; sp.overlay.show_extras = False
    except Exception as e: print("viewport:", e)

def build_hierarchy():
    """Parent every object to an Empty named after its collection so the USD export gets grouped Xforms."""
    hc = coll("Hierarchy")
    groups = {}
    skip = lambda c: c.name in ("Hierarchy", "Prototypes") or c.name.startswith("P_")
    for c in bpy.data.collections:
        if skip(c): continue
        g = bpy.data.objects.new(f"GRP_{c.name}", None); g.empty_display_size = 0.1; hc.objects.link(g); groups[c.name] = g
        for o in c.objects:
            if o.parent is None: o.parent = g
    # nest group empties like the collection tree (Products_G1 under Products, ...)
    for c in bpy.data.collections:
        for ch in c.children:
            if ch.name in groups and c.name in groups: groups[ch.name].parent = groups[c.name]
    print(f"hierarchy: {len(groups)} group empties")

def finalize_view_layer():
    try:
        lc = bpy.context.view_layer.layer_collection.children.get("Prototypes")
        if lc: lc.exclude = True
    except Exception as e: print("view layer:", e)

def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for blk in (bpy.data.meshes, bpy.data.lights, bpy.data.cameras, bpy.data.materials, bpy.data.images):
        for b in list(blk):
            if b.users == 0 and not b.name.startswith(("floor_tiles_08", "white_plaster_02", "wide_street")) and b.name not in ("Render Result", "Viewer Node"):
                blk.remove(b)
    _MATS.clear(); LIB.clear(); TAGMESH.clear(); PROTO.clear()

STAGES = ["clear", "structure", "library", "gondolas", "walls", "chillers", "front", "produce", "signage", "lighting", "cameras", "hierarchy", "render"]

def build(stages=("all",), save=None):
    t0 = time.time()
    stages = STAGES if "all" in stages else list(stages)
    M = fixture_mats()
    for stg in stages:
        t = time.time()
        if stg == "clear": clear_scene(); M = fixture_mats()
        elif stg == "structure": build_structure(M)
        elif stg == "library": build_library()
        elif stg == "gondolas": build_gondolas(M)
        elif stg == "walls": build_wall_units(M)
        elif stg == "chillers": build_chillers(M)
        elif stg == "front": build_front(M)
        elif stg == "produce": build_produce_bakery(M)
        elif stg == "signage": build_signage(M)
        elif stg == "lighting": build_lighting(M)
        elif stg == "cameras": build_cameras()
        elif stg == "hierarchy": build_hierarchy()
        elif stg == "render": setup_render()
        print(f"[stage {stg}] {round(time.time() - t, 1)} s")
    finalize_view_layer()
    n_obj = len(bpy.data.objects); n_prod = sum(1 for o in bpy.data.objects if "sku" in o.keys() and not o.name.startswith("tag_"))
    STATS.update(objects=n_obj, products=n_prod, total_s=round(time.time() - t0, 1))
    print(f"DONE: {n_obj} objects, {n_prod} product instances, {round(time.time() - t0, 1)} s")
    if save:
        bpy.ops.wm.save_as_mainfile(filepath=save)
        print("saved", save)
    return STATS

if __name__ == "__main__" and "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1:]
    stages = ["all"] if "--all" in args else [a for a in args if not a.startswith("--") and a in STAGES]
    save = args[args.index("--save") + 1] if "--save" in args else None
    build(stages or ["all"], save=save)
