# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Created Date: Thursday, April 15th 2021, 4:38:48 pm
# Copyright: Tommi Hyppänen
#
# Modified 2025 Romain Guimbal
#
# Modified 2026 by Peak-Design:
#   - Ported to Blender 5.0 API
#   - Added failed parts tracking and popup warnings
#   - Import summary improvements

import json
import ntpath
import os
import time
from collections import Counter, OrderedDict

import numpy as np
import bmesh
import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

from .utils import get_addon_prefs
from .importer import NativeMeshData
from .mesh_utils import (
    add_material,
    obj_unlink_all,
    choose_hierarchy_types,
    create_new_obj_with_mesh,
    transform_to_up,
    set_obj_matrix_world,
    calculate_detail_level,
    bpy_update_object_data,
)

import material_db
from .material_db import (
    get_active_matdb_path,
    ensure_matdb_materials,
    apply_matdb_to_objects,
    cleanup_unused_step_materials,
    matdb_enum_items,
    list_matdb_files,
    PG_MaterialMapping,
)

# LRU file cache with max entry limit
MAX_FILE_CACHE = 10
global_file_cache = OrderedDict()

# Color quantization precision for material merging (~1.5% tolerance)
_COLOR_MERGE_PRECISION = 64


def _quantize_color(col):
    """Round color components to merge near-identical materials."""
    return tuple(
        round(c * _COLOR_MERGE_PRECISION) / _COLOR_MERGE_PRECISION for c in col
    )


def _cache_put(filepath, step_reader):
    """Add to file cache with LRU eviction."""
    if filepath in global_file_cache:
        global_file_cache.move_to_end(filepath)
    global_file_cache[filepath] = step_reader
    while len(global_file_cache) > MAX_FILE_CACHE:
        evicted_path, _ = global_file_cache.popitem(last=False)
        print(f"Cache evicted: {os.path.basename(evicted_path)}")


def _cache_get(filepath):
    """Get from file cache, updating LRU order. Returns None if not found."""
    if filepath in global_file_cache:
        global_file_cache.move_to_end(filepath)
        return global_file_cache[filepath]
    return None


# Debug timing flag — set from addon settings at import start
_debug_timing = False

# Cumulative timing accumulators for profiling Phase 2
_phase2_times = {
    "build_trimesh": 0.0,
    "fuse_verts": 0.0,
    "filter_zero_area": 0.0,
    "filter_same_face": 0.0,
    "fill_empty_color": 0.0,
    "get_all_loop_data": 0.0,
    "add_to_bm": 0.0,
    "bpy_update": 0.0,
}


def _reset_phase2_times():
    for k in _phase2_times:
        _phase2_times[k] = 0.0


def _print_phase2_times():
    print("\n--- Phase 2 timing breakdown ---")
    total = sum(_phase2_times.values())
    for k, v in sorted(_phase2_times.items(), key=lambda x: -x[1]):
        pct = (v / total * 100) if total > 0 else 0
        print(f"  {k:20s}: {v:7.2f}s  ({pct:4.1f}%)")
    print(f"  {'TOTAL':20s}: {total:7.2f}s")


def precompute_mesh_data(step_reader, shp, lind, angd, hacks, part_name=""):
    """Compute mesh + loop data from OCC shape.

    Returns (mesh, colors, mat_names, norms, uvs).
    mesh is either TriMesh (Python fallback) or NativeMeshData (C++ path).
    """
    if _debug_timing:
        t0 = time.time()

    mesh = step_reader.build_trimesh(
        shp, lin_def=lind, ang_def=angd, hacks=hacks, part_name=part_name
    )

    if _debug_timing:
        t1 = time.time()

    if isinstance(mesh, NativeMeshData):
        # Native path: numpy vectorized operations
        mesh.fuse_verts()
        if _debug_timing:
            t2 = time.time()
        mesh.filter_zero_area()
        if _debug_timing:
            t3 = time.time()
        mesh.filter_same_face()
        if _debug_timing:
            t4 = time.time()
        mesh.fill_empty_color()
        if _debug_timing:
            t5 = time.time()
        # Data stays in numpy arrays — no conversion needed
        colors = mesh.get_loop_colors()
        mat_names = mesh.get_loop_mat_names()
        norms = mesh.get_loop_norms()
        uvs = None  # not used currently
        if _debug_timing:
            t6 = time.time()
    else:
        # Python fallback: TriMesh path
        mesh.fuse_verts()
        if _debug_timing:
            t2 = time.time()
        mesh.filter_zero_area()
        if _debug_timing:
            t3 = time.time()
        mesh.filter_same_face()
        if _debug_timing:
            t4 = time.time()
        mesh.fill_empty_color()
        if _debug_timing:
            t5 = time.time()
        colors, mat_names, norms, uvs = mesh.get_all_loop_data()
        if _debug_timing:
            t6 = time.time()

    if _debug_timing:
        _phase2_times["build_trimesh"] += t1 - t0
        _phase2_times["fuse_verts"] += t2 - t1
        _phase2_times["filter_zero_area"] += t3 - t2
        _phase2_times["filter_same_face"] += t4 - t3
        _phase2_times["fill_empty_color"] += t5 - t4
        _phase2_times["get_all_loop_data"] += t6 - t5

    return mesh, colors, mat_names, norms, uvs


def apply_mesh_to_blender(
    obj, mesh, colors, mat_names, norms, uvs, vcol_name="Colors", build_materials=True
):
    """Main-thread only: push precomputed mesh data into a Blender object."""
    if isinstance(mesh, NativeMeshData):
        return _apply_native_mesh(
            obj, mesh, colors, mat_names, norms, vcol_name, build_materials
        )
    else:
        return _apply_trimesh(
            obj, mesh, colors, mat_names, norms, uvs, vcol_name, build_materials
        )


def _apply_native_mesh(obj, mesh, colors, mat_names, norms, vcol_name, build_materials):
    """Fast path: from_pydata + foreach_set, no bmesh for geometry."""
    n_verts = len(mesh.verts)
    n_faces = len(mesh.faces)
    if _debug_timing:
        print(f"[bm] {n_verts}", end="")
        t0 = time.time()

    me = obj.data

    # Ensure OBJECT mode for mesh updates
    active = bpy.context.object
    prev_mode = active.mode if active else "OBJECT"
    if prev_mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # Clear existing geometry and material slots (needed for Rebuild Selected)
    me.clear_geometry()
    me.materials.clear()

    # Direct mesh creation via foreach_set (avoids numpy→list conversion)
    me.vertices.add(n_verts)
    me.vertices.foreach_set("co", mesh.verts.astype(np.float32).ravel())
    me.loops.add(n_faces * 3)
    me.loops.foreach_set("vertex_index", mesh.faces.astype(np.int32).ravel())
    me.polygons.add(n_faces)
    loop_starts = np.arange(0, n_faces * 3, 3, dtype=np.int32)
    me.polygons.foreach_set("loop_start", loop_starts)
    loop_totals = np.full(n_faces, 3, dtype=np.int32)
    me.polygons.foreach_set("loop_total", loop_totals)
    me.update()
    me.validate()

    if _debug_timing:
        t1 = time.time()

    # -- Vertex colors via foreach_set --
    if n_faces > 0 and colors is not None and len(colors) > 0:
        color_attr = me.color_attributes.new(
            name=vcol_name, type="FLOAT_COLOR", domain="CORNER"
        )
        # colors is (T*3, 3) float32 — need RGBA (T*3, 4)
        n_loops = len(colors)
        rgba = np.ones((n_loops, 4), dtype=np.float32)
        rgba[:, :3] = colors
        color_attr.data.foreach_set("color", rgba.ravel())

    # -- Materials --
    if build_materials and n_faces > 0:
        # Build per-face material names (use tri_mat_names directly, not per-loop)
        face_mat_names = mesh.tri_mat_names  # list[str|None] len=n_faces
        face_colors = mesh.tri_colors  # (n_faces, 3) float32

        # Resolve None names to auto-generated names
        # Pre-compute quantized color hex for unnamed faces using numpy
        q = np.round(face_colors * _COLOR_MERGE_PRECISION) / _COLOR_MERGE_PRECISION
        q_bytes = np.clip((q * 255).astype(np.int32), 0, 255)
        resolved_names = []
        for fi in range(n_faces):
            mn = face_mat_names[fi]
            if mn is None:
                r, g, b = q_bytes[fi]
                mn = f"STEP_{r:02x}{g:02x}{b:02x}"
            resolved_names.append(mn)

        # Get unique material names and assign indices
        unique_names = list(dict.fromkeys(resolved_names))  # preserves order
        name_to_idx = {n: i for i, n in enumerate(unique_names)}

        # Ensure materials exist and attach to object
        for mn in unique_names:
            if mn not in bpy.data.materials:
                # Find the color for this material
                fi = resolved_names.index(mn)
                col = tuple(float(x) for x in face_colors[fi])
                add_material(mn, col[:3], link_vertex_color=False)
            me.materials.append(bpy.data.materials[mn])

        # Vectorized index assignment
        mat_indices = np.array([name_to_idx[n] for n in resolved_names], dtype=np.int32)
        me.polygons.foreach_set("material_index", mat_indices)

    # -- Seam/sharp edges: numpy computation + direct attribute API --
    # IMPORTANT: we must NOT use a bmesh round-trip here because
    # bm.from_mesh → bm.to_mesh can reorder loops, which breaks the
    # normals_split_custom_set mapping (norms follow from_pydata order).
    sharp_keys, seam_keys, max_v = _compute_edge_attributes(mesh)

    n_edges = len(me.edges)
    if n_edges > 0 and (len(sharp_keys) > 0 or len(seam_keys) > 0):
        # Get edge vertex pairs via foreach_get
        edge_verts = np.zeros(n_edges * 2, dtype=np.int32)
        me.edges.foreach_get("vertices", edge_verts)
        edge_verts = edge_verts.reshape(-1, 2)

        v0 = np.minimum(edge_verts[:, 0], edge_verts[:, 1])
        v1 = np.maximum(edge_verts[:, 0], edge_verts[:, 1])
        edge_keys = v0.astype(np.int64) * max_v + v1.astype(np.int64)

        if len(seam_keys) > 0:
            seam_arr = np.isin(edge_keys, seam_keys).astype(bool)
            me.edges.foreach_set("use_seam", seam_arr)

        if len(sharp_keys) > 0:
            sharp_arr = np.isin(edge_keys, sharp_keys).astype(bool)
            sharp_attr = me.attributes.get("sharp_edge")
            if sharp_attr is None:
                sharp_attr = me.attributes.new(
                    name="sharp_edge", type="BOOLEAN", domain="EDGE"
                )
            sharp_attr.data.foreach_set("value", sharp_arr)

    if norms is not None and len(norms) > 0:
        me.normals_split_custom_set(norms)

    if prev_mode != "OBJECT":
        bpy.ops.object.mode_set(mode=prev_mode)

    if _debug_timing:
        t2 = time.time()
        _phase2_times["add_to_bm"] += t1 - t0
        _phase2_times["bpy_update"] += t2 - t1

    return mesh.matrix


def _compute_edge_attributes(mesh):
    """Vectorized computation of seam and sharp edge sets.

    Returns (sharp_keys_packed, seam_keys_packed, max_v) where keys are
    int64-packed edge keys: min_v * max_v + max_v_of_edge.
    Caller uses these to look up Blender edges efficiently.
    """
    margin = 0.02
    faces = mesh.faces  # (T, 3) int32
    loop_norms = mesh.get_loop_norms()  # (T*3, 3) float32
    verts = mesh.verts  # (V, 3) float32
    batches = mesh.tri_batches  # (T,) int32

    T = len(faces)
    if T == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 1

    max_v = int(np.max(faces)) + 1

    # Build half-edge table: 3 edges per face
    fi = np.arange(T, dtype=np.int32)

    va = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    vb = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    corner_a = np.concatenate([fi * 3, fi * 3 + 1, fi * 3 + 2])
    corner_b = np.concatenate([fi * 3 + 1, fi * 3 + 2, fi * 3])
    face_of = np.concatenate([fi, fi, fi])

    edge_min = np.minimum(va, vb)
    edge_max = np.maximum(va, vb)
    edge_keys = edge_min.astype(np.int64) * max_v + edge_max.astype(np.int64)

    unique_keys, inverse, counts = np.unique(
        edge_keys, return_inverse=True, return_counts=True
    )

    # Interior edges only (shared by exactly 2 faces)
    interior_idx = np.where(counts == 2)[0]
    if len(interior_idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), max_v

    group_starts = np.zeros(len(unique_keys) + 1, dtype=np.int64)
    np.cumsum(counts, out=group_starts[1:])
    order = np.argsort(edge_keys)

    int_starts = group_starts[interior_idx]
    he0 = order[int_starts]
    he1 = order[int_starts + 1]

    # --- Normal discontinuity test (cross-batch edges only) ---
    cross_batch = batches[face_of[he0]] != batches[face_of[he1]]
    int_edge_keys = unique_keys[interior_idx]

    ev0 = edge_min[he0]
    ev1 = edge_max[he0]

    # Map half-edge corners to sorted edge vertices
    he0_a_is_ev0 = va[he0] == ev0
    n0_ev0_c = np.where(he0_a_is_ev0, corner_a[he0], corner_b[he0])
    n0_ev1_c = np.where(he0_a_is_ev0, corner_b[he0], corner_a[he0])
    he1_a_is_ev0 = va[he1] == ev0
    n1_ev0_c = np.where(he1_a_is_ev0, corner_a[he1], corner_b[he1])
    n1_ev1_c = np.where(he1_a_is_ev0, corner_b[he1], corner_a[he1])

    # Get normal vectors (N_interior, 3)
    norm0_ev0 = loop_norms[n0_ev0_c]
    norm1_ev0 = loop_norms[n1_ev0_c]
    norm0_ev1 = loop_norms[n0_ev1_c]
    norm1_ev1 = loop_norms[n1_ev1_c]

    # Edge direction as projection plane normal
    plane = verts[ev0] - verts[ev1]
    plane_len = np.linalg.norm(plane, axis=1, keepdims=True)
    plane = plane / np.maximum(plane_len, 1e-12)

    def _batch_prjtest(plane, n0, n1):
        dot0 = np.sum(plane * n0, axis=1, keepdims=True)
        prj0 = n0 - plane * dot0
        prj0 = prj0 / np.maximum(np.linalg.norm(prj0, axis=1, keepdims=True), 1e-12)
        dot1 = np.sum(plane * n1, axis=1, keepdims=True)
        prj1 = n1 - plane * dot1
        prj1 = prj1 / np.maximum(np.linalg.norm(prj1, axis=1, keepdims=True), 1e-12)
        return np.sum(prj0 * prj1, axis=1) < (1.0 - margin)

    sharp_ev0 = _batch_prjtest(plane, norm0_ev0, norm1_ev0)
    sharp_ev1 = _batch_prjtest(plane, norm0_ev1, norm1_ev1)
    # Sharp/seam only between different OCC faces WITH normal discontinuity.
    # Smooth boundaries (e.g. cylinder halves) get neither sharp nor seam.
    discontinuous = cross_batch & sharp_ev0 & sharp_ev1
    sharp_keys = int_edge_keys[discontinuous]
    seam_keys = sharp_keys  # seams only where normals actually split

    return sharp_keys, seam_keys, max_v


def _apply_trimesh(
    obj, mesh, colors, mat_names, norms, uvs, vcol_name, build_materials
):
    """Original bmesh path for TriMesh objects."""
    if _debug_timing:
        print(f"[bm] {len(mesh.verts)}", end="")
        t0 = time.time()
    bm = bmesh.new()
    mesh.add_to_bm(bm, edges_as_seams=True, discontinuity_as_sharp=True)
    if _debug_timing:
        t1 = time.time()
    bpy_update_object_data(
        obj.data,
        bm,
        vcol_name,
        colors,
        uvs,
        norms,
        mat_names,
        build_materials=build_materials,
    )
    if _debug_timing:
        t2 = time.time()
        _phase2_times["add_to_bm"] += t1 - t0
        _phase2_times["bpy_update"] += t2 - t1

    return mesh.matrix


def build_mesh(step_reader, obj, shp, lind, angd, vcol_name="Colors"):
    hacks = set([])
    if get_addon_prefs().hack_skip_zero_solids:
        hacks.add("skip_solids")

    mesh, colors, mat_names, norms, uvs = precompute_mesh_data(
        step_reader, shp, lind, angd, hacks
    )

    return apply_mesh_to_blender(
        obj,
        mesh,
        colors,
        mat_names,
        norms,
        uvs,
        vcol_name,
        build_materials=get_addon_prefs().build_materials,
    )


def _show_import_issues_popup(failed_parts, recovered_parts):
    """Show a Blender popup dialog listing parts that had import problems."""

    def draw(self, context):
        layout = self.layout
        if recovered_parts:
            layout.label(
                text=f"{len(recovered_parts)} part(s) had corrupted geometry and were recovered:"
            )
            col = layout.column(align=True)
            for name in recovered_parts[:20]:
                col.label(text=f"    {name}", icon="FILE_REFRESH")
            if len(recovered_parts) > 20:
                col.label(text=f"    ... and {len(recovered_parts) - 20} more")
            col.label(
                text="    Recovered parts may have missing faces or appearance.",
                icon="INFO",
            )
            layout.separator()
        if failed_parts:
            layout.label(text=f"{len(failed_parts)} part(s) produced no geometry:")
            col = layout.column(align=True)
            for name in failed_parts[:20]:
                col.label(text=f"    {name}", icon="ERROR")
            if len(failed_parts) > 20:
                col.label(text=f"    ... and {len(failed_parts) - 20} more")
            col.label(
                text="    Usually caused by unresolved references in the STEP file.",
                icon="INFO",
            )

    icon = "ERROR" if failed_parts else "INFO"
    bpy.context.window_manager.popup_menu(
        draw, title="STEPper NEXT Import Warning", icon=icon
    )


def load_step(
    context,
    filepath,
    custom_scale=None,
    lin_deflection=0.8,
    ang_deflection=0.5,
    # merge_distance=0.001,
    up_as="Y",
    htypes="TREE",
    apply_scale=True,
    material_database="NONE",
):
    from . import importer

    global _debug_timing
    _debug_timing = get_addon_prefs().debug_timing

    hierarchy_flat, hierarchy_tree, hierarchy_empties = choose_hierarchy_types(htypes)

    filename = "".join(ntpath.basename(filepath).split(".")[:-1])

    cached = _cache_get(filepath)
    if cached is None:
        try:
            step_reader = importer.ReadSTEP(filepath)
            _cache_put(filepath, step_reader)
        except AssertionError as e:
            print(e)
            return False
    else:
        step_reader = cached
        print("Loaded file from cache")

    tree = step_reader.tree
    scale = step_reader.scale
    if custom_scale is not None:
        scale = custom_scale

    # divide by Blender unit length
    scale /= context.scene.unit_settings.scale_length
    print("Current Blender scale set at:", context.scene.unit_settings.scale_length)

    wm = bpy.context.window_manager

    created_objs = []
    created_names = {}
    created_uuid = {}

    # traverse shapes, render in "face" mode
    start_time = time.time()
    all_shapes = tree.get_shapes()
    total = len(all_shapes)

    # Tessellation is done on-demand inside build_trimesh (per shape).
    # Pre-tessellation was removed: the threading it used caused race
    # conditions leading to costly [re-tess] retessellations later.

    # Eagerly build recovery compounds (only needed for corrupt STEP files
    # with unresolved references).  Running it here makes the cost visible
    # instead of hiding it inside the first build_trimesh call.
    if step_reader.import_problems.get("Unresolved refs", 0) > 0 or True:
        # Always try — the reader checks internally if recovery is needed
        rec_start = time.time()
        step_reader._build_recovery_compound()
        rec_dt = time.time() - rec_start
        if rec_dt > 0.5:
            n_compounds = (
                len(step_reader._recovery_compounds)
                if step_reader._recovery_compounds
                else 0
            )
            print(
                f"\n--- Recovery compounds built in {rec_dt:.2f}s ({n_compounds} compounds) ---"
            )

    # Pre-compute mesh data for all unique shapes before the bpy loop.
    # This lets us batch-create materials upfront and keeps the Phase 2
    # loop focused on the fast bmesh/bpy work.
    _reset_phase2_times()
    hacks = set()
    if get_addon_prefs().hack_skip_zero_solids:
        hacks.add("skip_solids")
    build_materials = get_addon_prefs().build_materials

    # Identify unique shapes (first occurrence per shape_name)
    unique_shapes = {}  # shape_name -> (shp, part_name)
    for shp, node_index in all_shapes:
        if shp is None:
            continue
        _, _, tag, part_name, _, _, _ = tree.nodes[node_index].get_values()
        shape_name = "tt_" + repr(tag)
        if shape_name not in unique_shapes:
            unique_shapes[shape_name] = (shp, part_name)

    precomputed = {}  # shape_name -> (mesh, colors, mat_names, norms, uvs)
    n_unique = len(unique_shapes)
    if n_unique > 0:
        print(f"\n--- Phase 1/3: Pre-computing {n_unique} unique meshes ---")
        precomp_start = time.time()

        last_pct_10 = 0
        for si, (sname, (shp, part_name)) in enumerate(unique_shapes.items()):
            try:
                if _debug_timing:
                    t_shape = time.time()
                precomputed[sname] = precompute_mesh_data(
                    step_reader,
                    shp,
                    lin_deflection,
                    ang_deflection,
                    hacks,
                    part_name=part_name,
                )
                if _debug_timing:
                    dt_shape = time.time() - t_shape
                    if dt_shape > 2.0:
                        pname = part_name or sname
                        print(f"\n  [{pname}: {dt_shape:.1f}s]", end="", flush=True)
            except Exception as e:
                print(f"\nWarning: precompute failed for {sname} ({part_name}): {e}")

            pct_10 = (100 * (si + 1) // n_unique) // 10 * 10
            if pct_10 > last_pct_10:
                print(f"\n  Phase 1: {pct_10}%", end="", flush=True)
                last_pct_10 = pct_10

        print(f"\nPre-compute done in {time.time() - precomp_start:.2f}s")

    # Pre-create all materials so bpy_update_object_data doesn't do it per-face
    if build_materials and precomputed:
        all_mat_info = {}  # mat_name -> color
        for mesh, colors, mat_names, norms, uvs in precomputed.values():
            for ci, mname in enumerate(mat_names):
                if mname:
                    if mname not in all_mat_info:
                        c = colors[ci]
                        all_mat_info[mname] = tuple(float(x) for x in c)
                else:
                    c = colors[ci]
                    col = (
                        tuple(float(x) for x in c)
                        if ci < len(colors) and c[0] >= 0.0
                        else (0.5, 0.5, 0.5)
                    )
                    col = _quantize_color(col)
                    auto_name = "STEP_" + "".join(
                        "{0:0{1}x}".format(int(col[i] * 255), 2) for i in range(3)
                    )
                    if auto_name not in all_mat_info:
                        all_mat_info[auto_name] = col
        for mname, col in all_mat_info.items():
            if mname not in bpy.data.materials:
                add_material(mname, col, link_vertex_color=False)

    print(f"\n--- Phase 2/3: Building {total} Blender objects ---")
    wm.progress_begin(0, total)
    for i, (shp, node_index) in enumerate(all_shapes):
        parent_uuid, self_uuid, tag, name, _, local_t, global_t = tree.nodes[
            node_index
        ].get_values()

        if name == "root":
            name = filename + ".empties"

        shape_name = "tt_" + repr(tag)
        wm.progress_update(i)
        obj = None

        # Shape found in leaf
        if shp:
            if _debug_timing:
                print(
                    "\nBuilding ({}/{}): {} ".format(i + 1, total, name),
                    end="",
                    flush=True,
                )
                print("[T" + repr(shp.ShapeType()) + "]", end="", flush=True)

            # If object already built, just copy it using linked mesh data
            if shape_name in created_names:
                if _debug_timing:
                    print("[Link]", end="", flush=True)

                source_obj = created_names[shape_name]
                obj = source_obj.copy()
                created_objs.append(obj)
            else:
                if _debug_timing:
                    print("[Build]", end="", flush=True)

                obj = create_new_obj_with_mesh(name)

                if shape_name in precomputed:
                    # Use pre-computed data — only bpy/bmesh work on main thread
                    mesh, colors, mat_names, norms, uvs = precomputed[shape_name]
                    apply_mesh_to_blender(
                        obj,
                        mesh,
                        colors,
                        mat_names,
                        norms,
                        uvs,
                        build_materials=build_materials,
                    )
                else:
                    # Fallback for shapes that failed precompute
                    build_mesh(step_reader, obj, shp, lin_deflection, ang_deflection)

                # Track parts that produced no geometry
                if obj.data is not None and len(obj.data.vertices) == 0:
                    step_reader.failed_parts.append(name)
                    if get_addon_prefs().skip_empty_objects:
                        mesh_data = obj.data
                        bpy.data.objects.remove(obj)
                        bpy.data.meshes.remove(mesh_data)
                        obj = None

                if obj is not None:
                    created_objs.append(obj)
                    created_names[shape_name] = obj

        # No shape in leaf, empty creation enabled, do this
        elif hierarchy_empties:
            # Create empty
            obj = bpy.data.objects.new(name, None)
            obj.empty_display_size = 2
            obj.empty_display_type = "PLAIN_AXES"
            created_objs.append(obj)
            # set_obj_matrix_world(obj, global_t)

        # Object has been created
        if obj:
            # assign property to obj
            obj["STEP_tag"] = tag
            obj["STEP_parent"] = parent_uuid
            obj["STEP_uuid"] = self_uuid
            obj["STEP_file"] = filepath
            obj["STEP_name"] = name
            obj["STEP_tree_location"] = node_index
            obj["STEP_applied_scale"] = scale if apply_scale else 0.0
            # Store original STEP material names for material database feature
            if (
                obj.data is not None
                and hasattr(obj.data, "materials")
                and obj.data.materials
            ):
                obj["STEP_materials"] = json.dumps(
                    [m.name if m else "" for m in obj.data.materials]
                )
            created_uuid[self_uuid] = obj

    # assert len(created_objs) == len(shapes_labels)
    if _debug_timing:
        _print_phase2_times()
    print("\n" + repr(step_reader.import_problems))

    # remove all temporary links
    for tobj in created_objs:
        obj_unlink_all(tobj)

    # build flat collection
    if hierarchy_flat:
        flat_collection = bpy.data.collections.new(filename + ".flat")
        bpy.context.scene.collection.children.link(flat_collection)

        created_collections = {}
        for obj in created_objs:
            group_name = obj["STEP_name"]

            # max collection name len = 61
            if len(group_name) > 50:
                group_name = group_name[:25] + "_" + group_name[-25:]

            # TODO: check dupe collections for dupe imports
            if group_name not in created_collections:
                group_collection = bpy.data.collections.new(group_name)
                created_collections[group_name] = group_collection
                flat_collection.children.link(group_collection)
            else:
                group_collection = created_collections[group_name]

            global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
            set_obj_matrix_world(obj, global_t)
            group_collection.objects.link(obj)

    # build tree of collections
    if hierarchy_tree:
        tree_collection = bpy.data.collections.new(filename + ".hierarchy")
        bpy.context.scene.collection.children.link(tree_collection)
        hierarchy_collections = {}
        hierarchy_collections[-1] = tree_collection

        root = tree.nodes[0]
        # Map root node to tree collection so objects with STEP_parent=0 resolve
        hierarchy_collections[root.index] = tree_collection

        if len(root.children) > 0:
            # Iterative tree traversal (avoids recursion limit on deep assemblies)
            stack = list(reversed([(c, 0, tree_collection) for c in root.children]))
            while stack:
                node_idx, level, parent_collection = stack.pop()
                node = tree.nodes[node_idx]
                if len(node.children) > 0:
                    collection_node = bpy.data.collections.new(node.name)
                    assert node.index not in hierarchy_collections
                    hierarchy_collections[node.index] = collection_node
                    parent_collection.children.link(collection_node)
                    for c in reversed(node.children):
                        stack.append((c, level + 1, collection_node))

            # link objects to tree
            if len(hierarchy_collections.items()) > 0:
                for obj in created_objs:
                    parent_id = obj.get("STEP_parent", -1)
                    if parent_id in hierarchy_collections:
                        hierarchy_collections[parent_id].objects.link(obj)
                    elif -1 in hierarchy_collections:
                        hierarchy_collections[-1].objects.link(obj)
                    else:
                        bpy.context.scene.collection.objects.link(obj)
                    global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
                    set_obj_matrix_world(obj, global_t)

    # build hierarchy with empties
    if hierarchy_empties:
        for obj in created_objs:
            global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
            set_obj_matrix_world(obj, global_t)
            bpy.context.scene.collection.objects.link(obj)

            # Parent objs
            parent_id = obj["STEP_parent"]
            if parent_id in created_uuid:
                parent = created_uuid[parent_id]
                obj.parent = parent
                obj.matrix_parent_inverse = parent.matrix_world.inverted()

    # Apply material database replacements (before transforms)
    db_path = get_active_matdb_path(material_database)
    if db_path:
        mappings = ensure_matdb_materials(db_path)
        apply_matdb_to_objects(created_objs, mappings)
        cleanup_unused_step_materials()

    print(f"\n--- Phase 3/3: Applying transforms ---")
    transform_to_up(up_as[0], created_objs, scale, apply_scale=apply_scale)

    wm.progress_end()
    elapsed = time.time() - start_time

    # Import summary report
    n_objects = len(created_objs)
    n_unique = len(created_names)
    n_linked = n_objects - n_unique
    print(f"\n{'='*50}")
    print(f"  STEPper NEXT Import Summary")
    print(f"{'='*50}")
    print(f"  File:    {filename}")
    print(f"  Objects: {n_objects} ({n_unique} unique, {n_linked} linked copies)")
    print(f"  Scale:   {scale:.6f} m/unit")
    print(f"  Time:    {elapsed:.2f}s")
    has_problems = False
    for k, v in step_reader.import_problems.items():
        if v > 0:
            print(f"  Warning - {k}: {v}")
            has_problems = True
    if step_reader.skipped_shapes:
        print(f"  Skipped: {len(step_reader.skipped_shapes)} shapes")
        has_problems = True
    if step_reader.recovered_parts:
        print(f"  Recovered parts ({len(step_reader.recovered_parts)}):")
        for rp_name in step_reader.recovered_parts:
            print(f"    - {rp_name}")
        has_problems = True
    if step_reader.failed_parts:
        print(f"  Failed parts ({len(step_reader.failed_parts)}):")
        for fp_name in step_reader.failed_parts:
            print(f"    - {fp_name}")
        has_problems = True
    if not has_problems:
        print(f"  No warnings")
    print(f"{'='*50}")

    # Return lists of failed/recovered part names (empty = full success)
    return step_reader.failed_parts, step_reader.recovered_parts


class PG_Stepper(bpy.types.PropertyGroup):
    """Per-scene properties (import parameters, file paths).

    Persistent settings (build_materials, debug_timing, etc.) live on
    STEP_AddonPreferences and are accessed via get_addon_prefs().
    """

    detail_level: bpy.props.IntProperty(
        name="Mesh detail",
        description="How detailed you want the mesh to be",
        default=100,
        min=1,
    )

    lin_deflection: bpy.props.FloatProperty(
        name="Linear deflection",
        description="Smaller values increase polygon count. Higher values lower polygon count.",
        default=0.8,
        min=0.002,
        # max=2.0,
    )

    ang_deflection: bpy.props.FloatProperty(
        name="Angular deflection",
        description="Smaller values increase polygon count. Higher values lower polygon count.",
        default=0.5,
        min=0.002,
        # max=2.0,
    )

    fix_ascii_file: bpy.props.StringProperty(
        name="File",
        description="Path to problematic STEP file",
        default="",
        maxlen=1024,
        subtype="FILE_PATH",
    )

    # Material database UI state
    mat_db_mappings: bpy.props.CollectionProperty(type=PG_MaterialMapping)
    mat_db_active_index: bpy.props.IntProperty(default=0)
    mat_db_apply_selection_only: bpy.props.BoolProperty(
        name="Selection only",
        description="Apply material mappings only to selected objects",
        default=False,
    )


class ImportStepCADOperator(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.occ_import_step"
    bl_label = "Import STEP"
    bl_description = "Import a STEP file"
    bl_options = {"PRESET"}

    filter_glob: StringProperty(default="*.step;*.stp;*.st", options={"HIDDEN"})
    files: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)
    # files: bpy.props.CollectionProperty(type=idprop.types.IDPropertyGroup)
    override_file: StringProperty(default="", options={"HIDDEN"})

    fw_as: bpy.props.EnumProperty(
        items=[
            ("XPOS", "X", "", 0),
            # ("XNEG", "X-", "", 1),
            ("YPOS", "Y", "", 2),
            # ("YNEG", "Y-", "", 3),
            ("ZPOS", "Z", "", 4),
            # ("ZNEG", "Z-", "", 5),
        ],
        name="Forward",
        default="ZPOS",
        description="Forward axis of the imported model",
    )

    up_as: bpy.props.EnumProperty(
        items=[
            ("XPOS", "X", "", 0),
            # ("XNEG", "X-", "", 1),
            ("YPOS", "Y", "", 2),
            # ("YNEG", "Y-", "", 3),
            ("ZPOS", "Z", "", 4),
            # ("ZNEG", "Z-", "", 5),
        ],
        name="Up",
        default="YPOS",
        description="Up axis of the imported model",
    )

    hierarchy_types: bpy.props.EnumProperty(
        items=[
            ("FLAT", "Flat collection", "", 2),
            ("TREE", "Tree collection", "", 4),
            ("EMPTIES", "Parented empties", "", 6),
            # ("FLAT_AND_TREE", "Flat and tree collection", "", 0),
        ],
        name="Tree hierarchy",
        default="EMPTIES",
        description="Organization styles of objects",
    )

    user_scale: bpy.props.FloatProperty(
        name="Scale", description="Set object scale", default=0.01, min=0.00001
    )

    lin_deflection: bpy.props.FloatProperty(
        name="Linear deflection",
        description="Smaller values increase polygon count. Higher values lower polygon count.",
        default=0.8,
        min=0.002,
        max=2.0,
    )

    ang_deflection: bpy.props.FloatProperty(
        name="Angular deflection",
        description="Smaller values increase polygon count. Higher values lower polygon count.",
        default=0.5,
        min=0.002,
        max=2.0,
    )

    detail_level: bpy.props.IntProperty(
        name="Mesh detail",
        description="How detailed you want the mesh to be",
        default=100,
        min=1,
    )

    custom_scale: bpy.props.BoolProperty(
        name="Custom scale",
        description="Instead of loading the unit information from the file, determine it manually",
        default=False,
    )

    apply_scale: bpy.props.BoolProperty(
        name="Apply scale",
        description="Bake scale into mesh vertices so object scale is (1, 1, 1)",
        default=True,
    )

    material_database: bpy.props.EnumProperty(
        items=matdb_enum_items,
        name="Material Database",
        description="Replace STEP materials using a material database",
    )

    def draw(self, context):
        layout = self.layout

        def spacer(inpl):
            row = inpl.row()
            row.ui_units_y = 0.5
            row.label(text="")
            return row

        row = layout.row()

        row.label(text="STEPper NEXT import options:")

        col = layout.box()
        col = col.column(align=True)

        row = col.row()
        row.prop(self, "custom_scale")
        if self.custom_scale:
            row = col.row()
            row.prop(self, "user_scale")

        row = col.row()
        row.prop(self, "apply_scale")

        if get_addon_prefs().simpler_parameters:
            row = col.row()
            row.prop(self, "detail_level")

        else:
            row = col.row()
            row.prop(self, "lin_deflection")

            row = col.row()
            row.prop(self, "ang_deflection")

        # row = col.row()
        # row.prop(prg, "fw_as")

        row = col.row()
        row.prop(self, "up_as")

        row = col.row()
        row.prop(self, "hierarchy_types", text="Hierarchy")

        spacer(col)
        row = col.row()
        row.prop(self, "material_database", text="Material DB")

    def invoke(self, context, event):
        # Default the material database dropdown to the addon preference
        prefs = get_addon_prefs()
        if prefs.active_matdb and prefs.active_matdb != "NONE":
            # Verify it's still a valid option
            valid = {name for name, _ in list_matdb_files()}
            if prefs.active_matdb in valid:
                self.material_database = prefs.active_matdb
        return super().invoke(context, event)

    def execute(self, context):
        folder = os.path.dirname(self.filepath)

        # print(type(self.files))
        # print(dir(self.files))
        l_def, a_def = self.lin_deflection, self.ang_deflection
        if get_addon_prefs().simpler_parameters:
            a_def, l_def = calculate_detail_level(self.detail_level)

        import_files = [i.name for i in self.files]

        if self.override_file != "":
            import_files = [self.override_file]

        # iterate through the selected files
        all_failed_parts = []
        all_recovered_parts = []
        for j, i in enumerate(import_files):
            # generate full path to file
            path_to_file = os.path.join(folder, i)
            print("Opening file:", path_to_file)
            result = load_step(
                context,
                path_to_file,
                custom_scale=self.user_scale if self.custom_scale else None,
                lin_deflection=l_def,
                ang_deflection=a_def,
                up_as=self.up_as,
                htypes=self.hierarchy_types,
                apply_scale=self.apply_scale,
                material_database=self.material_database,
            )
            if result is False:
                self.report(
                    {"ERROR"}, "STEP file could not be opened. Possibly damaged file."
                )
                return {"CANCELLED"}
            failed, recovered = result
            all_failed_parts.extend(failed)
            all_recovered_parts.extend(recovered)

        if all_failed_parts or all_recovered_parts:
            _show_import_issues_popup(all_failed_parts, all_recovered_parts)
            if all_failed_parts:
                msg = f"{len(all_failed_parts)} part(s) imported with no geometry."
                self.report({"WARNING"}, msg)
            if all_recovered_parts:
                msg = f"{len(all_recovered_parts)} part(s) recovered from corrupted geometry."
                self.report({"INFO"}, msg)

        return {"FINISHED"}


class STEP_OT_ClearCache(bpy.types.Operator):
    bl_idname = "object.occ_clear_cache"
    bl_label = "Clear STEP cache"
    bl_description = "Clear STEP cache, enabling the reload of a file"

    def execute(self, context):
        # utils.memorytrace_print()
        # global global_file_cache
        # items = list(global_file_cache.values())
        # for entry in items:
        #     for i, shp in enumerate(entry):
        #         label, color, tag = entry[shp]
        #         # shp.Nullify()

        global_file_cache.clear()
        return {"FINISHED"}


class STEP_OT_FixASCII(bpy.types.Operator):
    bl_idname = "object.occ_fix_ascii"
    bl_label = "Attempt STEP ASCII fix"
    bl_description = (
        "Attempt repairing invalid STEP characters.\n"
        "For files that crash the program when trying to load.\n"
        "A new file with _fix post-fix is created into the folder."
    )

    def execute(self, context):
        from pathlib import Path
        import unicodedata

        print("Attempting to format STEP file as ASCII")
        i_file = context.scene.stepper.fix_ascii_file
        p = Path(i_file)
        if i_file == "" or not p.exists():
            self.report(
                {"ERROR"},
                "File does not exist.",
            )
            return {"FINISHED"}
        print(p.stat().st_size // 1024, "kB")

        outf = Path(p.parent, Path(p.stem.replace(" ", "_") + "_fix.step"))
        with outf.open("w", encoding="ASCII") as fo:
            with p.open("rb") as f:
                content = bytearray(f.read())
                content = content.replace(b"\r\n", b"\n")
                content = content.replace(b",\n", b",")
                content = content.replace(b"(\n", b"(")
                fo.write(content.decode("ASCII", errors="ignore"))

        self.report(
            {"INFO"},
            "Operation finished.",
        )
        return {"FINISHED"}


class STEP_OT_PrintDebug(bpy.types.Operator):
    bl_idname = "object.occ_print_debug"
    bl_label = "Print STEP debug info"
    bl_description = "Print STEP debug info"

    def execute(self, context):
        from pathlib import Path

        print("Attempting to format STEP file as ASCII")
        i_file = context.scene.stepper.print_debug
        p = Path(i_file)
        if i_file == "" or not p.exists():
            self.report(
                {"ERROR"},
                "File does not exist.",
            )
            return {"FINISHED"}

        print(p.stat().st_size // 1024, "kB")

        from . import stepanalyzer

        SA = stepanalyzer.StepAnalyzer(filename=p)
        print(SA.dump())

        self.report(
            {"INFO"},
            "Operation finished.",
        )
        return {"FINISHED"}


class STEP_OT_ReloadSTEP(bpy.types.Operator):
    bl_idname = "object.occ_reload_step"
    bl_label = "Reload STEP"
    bl_description = "Reload STEP file"

    @classmethod
    def poll(cls, context):
        return context.object is not None and "STEP_file" in context.object

    def execute(self, context):
        from . import importer

        filepath = context.object["STEP_file"]
        step_reader = importer.ReadSTEP(filepath)
        _cache_put(filepath, step_reader)
        return {"FINISHED"}


class STEP_OT_ClearFileCache(bpy.types.Operator):
    bl_idname = "object.occ_clear_file_cache"
    bl_label = "Clear this file from cache"
    bl_description = "Remove the selected object's STEP file from cache (next import re-reads from disk)"

    @classmethod
    def poll(cls, context):
        return context.object is not None and "STEP_file" in context.object

    def execute(self, context):
        filepath = context.object["STEP_file"]
        if filepath in global_file_cache:
            del global_file_cache[filepath]
            self.report({"INFO"}, f"Cleared cache for: {os.path.basename(filepath)}")
        else:
            self.report({"WARNING"}, "File not in cache")
        return {"FINISHED"}


class STEP_OT_RebuildSelected(bpy.types.Operator):
    bl_idname = "object.occ_rebuild_selected"
    bl_label = "Rebuild selected objects from the STEP file"
    bl_description = "Experimental: Causes issues on some shapes\n" + bl_label

    @classmethod
    def poll(cls, context):
        return context.object is not None and "STEP_file" in context.object

    def execute(self, context):
        meshes = {}
        prevname = ""
        curname = ""
        build_tags = set()
        rebuilt_meshes = set()
        my_selection = list(context.selected_objects)

        ang_def = context.scene.stepper.ang_deflection
        lin_def = context.scene.stepper.lin_deflection
        # merge_distance = context.scene.stepper.merge_distance
        if get_addon_prefs().simpler_parameters:
            ang_def, lin_def = calculate_detail_level(
                bpy.context.scene.stepper.detail_level
            )

        # select all objs with the same meshes
        for obj in my_selection:
            for other_obj in context.scene.objects:
                if obj.data == other_obj.data:
                    other_obj.select_set(True)

        # go through all selected and rebuild the meshes
        wm = bpy.context.window_manager
        wm.progress_begin(0, len(my_selection))
        for progress_count, obj in enumerate(my_selection):
            if obj.data.name not in meshes:
                meshes[obj.data.name] = obj.data
                sel_tag = obj["STEP_tag"]
                prevname = curname
                curname = obj["STEP_file"]
            else:
                assert meshes[obj.data.name] == obj.data

            if sel_tag in rebuilt_meshes:
                continue

            if prevname != curname:
                cached_reader = _cache_get(curname)
                if cached_reader is not None:
                    step_reader = cached_reader
                    tree = step_reader.tree
                else:
                    self.report(
                        {"ERROR"},
                        'STEP loader: Object "{}" not found in cache for file {}. '
                        "Please reload STEP file".format(obj.name, curname),
                    )
                    break

            for shp, node_index in tree.get_shapes():
                _, _, tag, name, _, _, _ = tree.nodes[node_index].get_values()
                if tag == sel_tag:
                    rebuilt_meshes.add(sel_tag)
                    print("Rebuilding:", sel_tag, obj.data.name)
                    # Reset pre-tessellation flag so shapes get re-tessellated
                    # with the new deflection values
                    step_reader._pre_tessellated = False
                    build_mesh(step_reader, obj, shp, lin_def, ang_def)
                    # Re-apply baked scale if it was applied during import
                    applied_scale = obj.get("STEP_applied_scale", 0.0)
                    if applied_scale and applied_scale != 1.0:
                        mesh = obj.data
                        vert_count = len(mesh.vertices)
                        if vert_count > 0:
                            verts = np.empty(vert_count * 3, dtype=np.float32)
                            mesh.vertices.foreach_get("co", verts)
                            verts *= applied_scale
                            mesh.vertices.foreach_set("co", verts)
                            mesh.update()
                    obj.display_type = "TEXTURED"
                    build_tags.add(obj["STEP_tag"])
                    break

            wm.progress_update(progress_count)

        wm.progress_end()

        for obj in context.selected_objects:
            obj.display_type = "TEXTURED"

        # Re-apply active material database if one is set
        db_path = get_active_matdb_path()
        if db_path:
            mappings = ensure_matdb_materials(db_path)
            apply_matdb_to_objects(my_selection, mappings)

        return {"FINISHED"}


class STEP_PT_STEPper(bpy.types.Panel):
    bl_label = "STEPper NEXT: Build"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STEPper NEXT"

    def draw(self, context):
        prg = context.scene.stepper

        layout = self.layout

        # def spacer(inpl):
        #     row = inpl.row()
        #     row.ui_units_y = 0.5
        #     row.label(text="")
        #     return row

        row = layout.row()
        col = row.column(align=True)

        # row = col.row()
        # row.prop(prg, "merge_distance")

        if get_addon_prefs().simpler_parameters:
            row = col.row()
            row.prop(prg, "detail_level")

        else:
            row = col.row()
            row.prop(prg, "lin_deflection")

            row = col.row()
            row.prop(prg, "ang_deflection")

        layout = self.layout
        # layout.label(text="Used memory: {}".format(total_size(global_file_cache)))
        row = layout.row()
        row.operator(STEP_OT_RebuildSelected.bl_idname, text="Rebuild selected")


class STEP_PT_STEPper_Reload(bpy.types.Panel):
    bl_label = "STEPper NEXT: File"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STEPper NEXT"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator(STEP_OT_ReloadSTEP.bl_idname, text="Reload STEP file")
        row = layout.row()
        row.operator(
            STEP_OT_ClearFileCache.bl_idname, text="Clear this file from cache"
        )
        row = layout.row()
        row.operator(STEP_OT_ClearCache.bl_idname, text="Clear all cache")
        row = layout.row()
        row.label(text=f"Cached files: {len(global_file_cache)}/{MAX_FILE_CACHE}")


class STEP_PT_STEPper_Debug(bpy.types.Panel):
    bl_label = "STEPper NEXT: Debug"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STEPper NEXT"

    def draw(self, context):
        layout = self.layout
        prg = context.scene.stepper

        bxp = layout.box()
        bxp.label(text="Enforce ASCII")
        col = bxp.row().column(align=True)

        row = col.row()
        row.prop(prg, "fix_ascii_file")
        row = col.row()
        row.operator("object.occ_fix_ascii", text="Attempt fix STEP charset")

        # row = layout.row()
        # row.label(text="Error messages:")

        if (
            context.object is not None
            and "STEP_file" in context.object
            and context.object["STEP_file"] in global_file_cache
        ):
            bxp = layout.box()
            bxp.label(text="Reported problems:")

            row = bxp.row()
            col = row.column(align=True)
            step_reader = global_file_cache[context.object["STEP_file"]]
            for k, v in step_reader.import_problems.items():
                row = col.row()
                row.label(text=k + ": " + repr(v))

            bxs = layout.box()
            bxs.label(text="Skipped shapes:")

            row = bxs.row()
            col = row.column(align=True)
            if len(step_reader.skipped_shapes) > 0:
                for v in step_reader.skipped_shapes:
                    row = col.row()
                    row.label(text=repr(v))
            else:
                row = col.row()
                row.label(text="No skipped shapes")

        else:
            bxp = layout.box()
            row = bxp.row()
            row.label(text="Select active STEP object")


class STEP_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    build_materials: bpy.props.BoolProperty(
        name="Build materials",
        description="Build materials from STEP file colors",
        default=True,
    )

    hack_skip_zero_solids: bpy.props.BoolProperty(
        name="Skip faulty solids",
        description="Skip corrupted/empty parts entirely (no healing or recovery attempts)",
        default=False,
    )

    simpler_parameters: bpy.props.BoolProperty(
        name="Artist friendly parameters",
        description="Instead of linear and angle deflection values, use only detail setting",
        default=True,
    )

    skip_empty_objects: bpy.props.BoolProperty(
        name="Skip empty objects",
        description="Don't create objects for parts that produce no geometry",
        default=True,
    )

    debug_timing: bpy.props.BoolProperty(
        name="Debug timing",
        description="Print detailed timing information during import",
        default=False,
    )

    active_matdb: bpy.props.EnumProperty(
        items=matdb_enum_items,
        name="Material Database",
        description="Active material database for import",
    )

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(self, "build_materials")

        row = layout.row()
        row.prop(self, "hack_skip_zero_solids")

        row = layout.row()
        row.prop(self, "simpler_parameters")

        row = layout.row()
        row.prop(self, "skip_empty_objects")

        row = layout.row()
        row.prop(self, "debug_timing")

        row = layout.row()
        row.prop(self, "active_matdb")


def menu_func_import(self, context):
    self.layout.operator(
        ImportStepCADOperator.bl_idname, text="STEP (.step, .stp) [STEPper NEXT]"
    )


classes = (
    STEP_AddonPreferences,
    PG_Stepper,
    ImportStepCADOperator,
    STEP_OT_ClearCache,
    STEP_OT_ClearFileCache,
    STEP_OT_RebuildSelected,
    STEP_OT_ReloadSTEP,
    STEP_OT_FixASCII,
    STEP_OT_PrintDebug,
    STEP_PT_STEPper,
    STEP_PT_STEPper_Reload,
    STEP_PT_STEPper_Debug,
)


def register():
    material_db.register()

    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.stepper = bpy.props.PointerProperty(type=PG_Stepper)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    for c in classes[::-1]:
        bpy.utils.unregister_class(c)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    del bpy.types.Scene.stepper

    material_db.unregister()
