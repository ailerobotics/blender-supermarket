"""Realise a :class:`Scenario` as a Blender scene. Runs inside Blender's Python.

One object per entity (unique ``pass_index`` -> IndexOB mask channel); products are
coloured boxes from the catalog, people are capsule mannequins, spills are thin
floor meshes with class-appropriate materials (clear specular water, opaque milk,
dark glossy oil, ...). Per-frame visibility and motion come straight from
``Scenario.state_at`` so pixels match the projected ground truth.

This module is only imported by ``generate.py`` under ``blender -b``; it is never
part of the offline test path.
"""
from __future__ import annotations

import math
import os
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # scenarios/
import catalog_geom as geom           # noqa: E402

_MAT_CACHE = {}


def _hexrgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _srgb_lin(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def _principled(name, rgb, rough=0.5, metal=0.0, transmission=0.0, alpha=1.0, emit=0.0):
    if name in _MAT_CACHE:
        return _MAT_CACHE[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    r, g, b = _srgb_lin(rgb)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    for key in ("Transmission Weight", "Transmission"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = transmission
            break
    if emit:
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (r, g, b, 1.0)
                break
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = 'BLEND' if hasattr(m, "blend_method") else m.blend_method
    _MAT_CACHE[name] = m
    return m


def _material_for(e):
    """Pick a material from the entity's class."""
    if e.kind == "product":
        s = geom.BY_ID.get(e.cls)
        rgb = _hexrgb(s["bg"]) if s else (0.6, 0.6, 0.6)
        return _principled(f"prod_{e.cls}", rgb, rough=0.45)
    if e.kind == "person":
        return _principled("person", (0.25, 0.3, 0.45), rough=0.7)
    if e.kind == "fixture":
        return _principled("fixture", (0.55, 0.56, 0.58), rough=0.6, metal=0.2)
    if e.cls == "water":
        return _principled("haz_water", (0.8, 0.85, 0.9), rough=0.02, transmission=0.9, alpha=0.5)
    if e.cls == "milk":
        return _principled("haz_milk", (0.96, 0.96, 0.93), rough=0.35)
    if e.cls == "juice":
        return _principled("haz_juice", (0.85, 0.4, 0.08), rough=0.15, transmission=0.3)
    if e.cls == "oil":
        return _principled("haz_oil", (0.12, 0.1, 0.04), rough=0.05, metal=0.0)
    if e.cls == "debris":
        return _principled("haz_debris", (0.4, 0.35, 0.3), rough=0.8)
    if e.cls == "fallen_product":
        s = geom.BY_ID.get(e.meta.get("sku", ""))
        rgb = _hexrgb(s["bg"]) if s else (0.7, 0.2, 0.2)
        return _principled(f"fallen_{e.cls}", rgb, rough=0.4)
    if e.cls == "floor_marking":
        return _principled("dist_floormark", (0.95, 0.85, 0.1), rough=0.6)
    if e.cls == "shadow":
        return _principled("dist_shadow", (0.05, 0.05, 0.05), rough=0.9)
    if e.cls == "paper_sign":
        return _principled("dist_paper", (0.95, 0.95, 0.95), rough=0.7)
    return _principled("misc", (0.5, 0.5, 0.5))


def _box_object(name, size, mat, pass_index):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= size[0]; v.co.y *= size[1]; v.co.z *= size[2]
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.pass_index = pass_index
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _person_object(name, size, mat, pass_index):
    """A capsule mannequin (body cylinder + head sphere) fitting the entity box."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    r = min(size[0], size[1]) / 2
    body_h = size[2] * 0.78
    bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r, radius2=r * 0.85,
                          depth=body_h)
    for v in bm.verts:
        v.co.z += body_h / 2 - size[2] / 2
    head = bmesh.new()
    bmesh.ops.create_uvsphere(head, u_segments=12, v_segments=8, radius=r * 1.1)
    hz = body_h + r * 1.1 - size[2] / 2
    for v in head.verts:
        v.co.z += hz
    head_me = bpy.data.meshes.new("h")
    head.to_mesh(head_me); head.free()
    bm.from_mesh(head_me)
    bpy.data.meshes.remove(head_me)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.pass_index = pass_index
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _add_shelves(scn, plan):
    """Draw cheap shelf boards + back panels for fixtures carrying product entities,
    so facings sit on something that reads as a gondola (pass_index 0, not annotated)."""
    fixture_ids = sorted({e.meta.get("fixture_id") for e in scn.entities
                          if e.kind == "product" and e.meta.get("fixture_id")})
    # some builders only store slot_id; recover fixture id from the slot prefix
    slot_ids = {e.meta.get("slot_id") for e in scn.entities if e.kind == "product"}
    board_mat = _principled("shelf_board", (0.78, 0.78, 0.8), rough=0.35, metal=0.4)
    back_mat = _principled("shelf_back", (0.62, 0.63, 0.66), rough=0.6)
    seen_back = set()
    for slot in geom.iter_slots(plan, fixtures=fixture_ids or None):
        if slot.slot_id not in slot_ids:
            continue
        (rx, ry), (fxv, fyv) = geom.shelf_frame(slot.facing_dir)
        ox, oy = slot.origin_xy
        # board: centred half the shelf depth in front of the back edge
        bcx = ox + fxv * slot.depth / 2
        bcy = oy + fyv * slot.depth / 2
        size = (abs(rx) * slot.width + abs(fxv) * slot.depth,
                abs(ry) * slot.width + abs(fyv) * slot.depth, 0.03)
        b = _box_object(f"shelf_{slot.slot_id}", size, board_mat, 0)
        b.location = (bcx, bcy, slot.shelf_z - 0.015)
        # one back panel per (fixture,bay) occludes the grey void behind
        key = (slot.fixture_id, slot.facing_dir, slot.bay_index)
        if key not in seen_back:
            seen_back.add(key)
            back_size = (abs(rx) * slot.width + abs(fxv) * 0.03,
                         abs(ry) * slot.width + abs(fyv) * 0.03, 2.0)
            bp = _box_object(f"back_{slot.slot_id}", back_size, back_mat, 0)
            bp.location = (ox - fxv * 0.03, oy - fyv * 0.03, 1.0)


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    _MAT_CACHE.clear()


def build(scn, plan):
    """Build floor, lighting, cameras and one object per entity. Returns id->object."""
    clear_scene()
    sc = bpy.context.scene
    store = plan["store"]

    # floor
    floor_mat = _principled("floor", (0.62, 0.62, 0.6), rough=0.25, metal=0.0)
    floor = _box_object("Floor", (store["width"], store["depth"], 0.02), floor_mat, 0)
    floor.location = (0, 0, -0.01)

    # lighting: a soft sun plus a filled world so interiors are legible and cheap
    sun_d = bpy.data.lights.new("Sun", "SUN"); sun_d.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_d); sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(35), math.radians(15), math.radians(40))
    world = sc.world or bpy.data.worlds.new("W")
    sc.world = world; world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg:
        bg.inputs["Color"].default_value = (0.7, 0.72, 0.75, 1.0)
        bg.inputs["Strength"].default_value = 0.6

    _add_shelves(scn, plan)

    objects = {}
    for e in scn.entities:
        mat = _material_for(e)
        if e.kind == "person":
            ob = _person_object(e.id, e.size, mat, e.pass_index)
        else:
            ob = _box_object(e.id, e.size, mat, e.pass_index)
        (cx, cy, cz), rot = e.pose_at(0)
        ob.location = (cx, cy, cz)
        ob.rotation_euler = (0, 0, rot)
        objects[e.id] = ob

    # cameras
    cam_objs = {}
    for cp in scn.cameras:
        cd = bpy.data.cameras.new(cp.name)
        cd.sensor_fit = 'HORIZONTAL'
        cd.sensor_width = cp.K()["sensor_mm"]
        cd.lens = cp.lens_mm
        cd.clip_end = 300
        ob = bpy.data.objects.new(cp.name, cd)
        ob.matrix_world = Matrix(cp.c2w())
        sc.collection.objects.link(ob)
        cam_objs[cp.name] = ob
    return objects, cam_objs


def apply_frame(scn, objects, frame):
    """Set per-frame visibility and pose for every entity object."""
    for e in scn.entities:
        ob = objects[e.id]
        present = e.present_at(frame)
        ob.hide_render = not present
        if present:
            (cx, cy, cz), rot = e.pose_at(frame)
            ob.location = (cx, cy, cz)
            ob.rotation_euler = (0, 0, rot)
