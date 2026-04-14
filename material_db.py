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

"""Material database: file I/O helpers, operators, and UI panel."""

import json
import os
from collections import Counter
import bpy
from .utils import get_addon_prefs

_MATDB_TEXT_NAME = "STEPper_MaterialDB"
_ADDON_DIR = os.path.dirname(os.path.realpath(__file__))


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------


def _get_matdb_dir():
    """Return the MaterialDB folder inside the addon directory, creating it if needed."""
    d = os.path.join(_ADDON_DIR, "MaterialDB")
    os.makedirs(d, exist_ok=True)
    return d


def list_matdb_files():
    """Return list of (filename_without_ext, full_path) for all .blend files in MaterialDB/."""
    d = _get_matdb_dir()
    result = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".blend"):
                result.append((f[:-6], os.path.join(d, f)))
    return result

def matdb_enum_items(self, context):
    """Dynamic enum items for material database selection."""
    items = [("NONE", "None", "Do not use a material database")]
    for name, path in list_matdb_files():
        items.append((name, name, f"Use material database: {name}"))
    return items


def get_active_matdb_path(db_name=None):
    """Return the full path of the active material database, or empty string."""
    if db_name is None:
        db_name = get_addon_prefs().active_matdb
    if not db_name or db_name == "NONE":
        return ""
    path = os.path.join(_get_matdb_dir(), db_name + ".blend")
    return path if os.path.isfile(path) else ""


def _write_material_database(filepath, mappings_dict):
    """Write material mappings and replacement materials to a .blend database file."""
    # Remove any leftover temporary text datablock
    while True:
        text = bpy.data.texts.get(_MATDB_TEXT_NAME)
        if text:
            bpy.data.texts.remove(text)
        else:
            break

    text = bpy.data.texts.new(_MATDB_TEXT_NAME)
    text.write(json.dumps(mappings_dict, indent=2))

    # Collect datablocks to write: the text + all replacement materials.
    # Linked/library materials (e.g. from asset browser) cannot be written
    # directly — make temporary local copies for those.
    datablocks = {text}
    temp_copies = []
    for replacement_name in set(mappings_dict.values()):
        mat = bpy.data.materials.get(replacement_name)
        if not mat:
            print(f"STEPper MatDB: Material '{replacement_name}' not found, skipping")
            continue
        if mat.library or mat.override_library:
            # Create a full local copy so it can be written to the database
            local_copy = mat.copy()
            local_copy.name = replacement_name  # keep the expected name
            datablocks.add(local_copy)
            temp_copies.append(local_copy)
        else:
            datablocks.add(mat)

    print(f"STEPper MatDB: Writing {len(datablocks)} datablocks to {filepath}")
    bpy.data.libraries.write(filepath, datablocks, fake_user=True)

    # Clean up temporary text datablock and any temporary material copies
    bpy.data.texts.remove(text)
    for tmp in temp_copies:
        bpy.data.materials.remove(tmp)


def _read_matdb_mappings(filepath):
    """Read ONLY the JSON mappings from a database .blend file.

    Does NOT append materials.  Returns dict {original_name: replacement_name}.
    """
    abs_path = bpy.path.abspath(filepath)
    if not abs_path or not os.path.isfile(abs_path):
        return {}

    # Clean up any leftover text block from a previous read
    while True:
        text = bpy.data.texts.get(_MATDB_TEXT_NAME)
        if text:
            bpy.data.texts.remove(text)
        else:
            break

    # Append only the text datablock
    with bpy.data.libraries.load(abs_path, link=False) as (data_from, data_to):
        if _MATDB_TEXT_NAME in data_from.texts:
            data_to.texts = [_MATDB_TEXT_NAME]

    mappings = {}
    text = bpy.data.texts.get(_MATDB_TEXT_NAME)
    if text:
        try:
            mappings = json.loads(text.as_string())
        except json.JSONDecodeError:
            print("STEPper MatDB: Invalid JSON in database file")
        bpy.data.texts.remove(text)
    else:
        print("STEPper MatDB: No mapping text block found in database file")

    return mappings


def _append_matdb_materials(filepath):
    """Append all materials from a database .blend into the current file.

    Materials that already exist locally (by name) are skipped to avoid
    duplicates.  Returns the number of materials appended.
    """
    abs_path = bpy.path.abspath(filepath)
    if not abs_path or not os.path.isfile(abs_path):
        return 0

    with bpy.data.libraries.load(abs_path, link=False) as (data_from, data_to):
        to_append = [m for m in data_from.materials if m not in bpy.data.materials]
        data_to.materials = to_append

    count = len(to_append)
    if count:
        print(f"STEPper MatDB: Appended {count} material(s) from database")
    return count


# ---------------------------------------------------------------------------
# Scene scanning and applying
# ---------------------------------------------------------------------------

def _scan_scene_materials():
    """Scan scene for STEP objects and determine material mappings.

    For each original STEP material name, find what replacement material is
    currently assigned.  When an original name maps to multiple different
    replacements, pick the one with the highest instance count.

    Returns dict {original_name: replacement_name}.
    """
    original_to_replacements = {}  # {orig: Counter({repl: count})}

    for obj in bpy.data.objects:
        if obj.data is None or not hasattr(obj.data, "materials"):
            continue

        step_mats_json = obj.get("STEP_materials")
        if step_mats_json:
            try:
                original_names = json.loads(step_mats_json)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            # Backward compat: treat current materials as identity mapping
            original_names = [m.name for m in obj.data.materials if m]

        current_mats = [m.name if m else None for m in obj.data.materials]

        for i, orig_name in enumerate(original_names):
            if i < len(current_mats) and current_mats[i]:
                repl_name = current_mats[i]
                if orig_name not in original_to_replacements:
                    original_to_replacements[orig_name] = Counter()
                original_to_replacements[orig_name][repl_name] += 1

    # Resolve conflicts: majority wins
    mappings = {}
    for orig_name, counter in original_to_replacements.items():
        mappings[orig_name] = counter.most_common(1)[0][0]

    return mappings


def ensure_matdb_materials(db_path):
    """Ensure all materials from a database .blend exist locally.

    Call this before any operation that needs to reference database materials.
    Returns the mappings dict.
    """
    if not db_path:
        return {}
    _append_matdb_materials(db_path)
    return _read_matdb_mappings(db_path)


def apply_matdb_to_objects(objects, mappings):
    """Replace STEP materials on *objects* according to *mappings* dict.

    Uses the STEP_materials custom property to determine original material
    names so replacements work even when materials were already swapped by
    a previous database apply.

    Returns the number of material slots replaced.
    """
    if not mappings:
        return 0

    print(f"Applying material database ({len(mappings)} mappings)")

    replaced = 0
    processed_meshes = set()
    for obj in objects:
        if obj.data is None or not hasattr(obj.data, "materials"):
            continue
        mesh = obj.data
        if mesh in processed_meshes:
            continue
        processed_meshes.add(mesh)

        # Read original STEP material names from the custom property
        step_mats_json = obj.get("STEP_materials")
        if step_mats_json:
            try:
                original_names = json.loads(step_mats_json)
            except (json.JSONDecodeError, TypeError):
                original_names = None
        else:
            original_names = None

        for slot_idx in range(len(mesh.materials)):
            # Determine the original STEP name for this slot
            if original_names and slot_idx < len(original_names):
                orig_name = original_names[slot_idx]
            else:
                # Fallback: use current material name
                mat = mesh.materials[slot_idx]
                orig_name = mat.name if mat else None

            if orig_name and orig_name in mappings:
                replacement_name = mappings[orig_name]
                replacement_mat = bpy.data.materials.get(replacement_name)
                current_mat = mesh.materials[slot_idx]
                if replacement_mat and replacement_mat != current_mat:
                    mesh.materials[slot_idx] = replacement_mat
                    replaced += 1

    return replaced


def cleanup_unused_step_materials():
    """Remove materials with zero users that were generated by STEP import."""
    removed = 0
    # Iterate over a snapshot since we're modifying the collection
    for mat in list(bpy.data.materials):
        if mat.users == 0 and (
            mat.name.startswith("STEP_") or not mat.name.startswith(".")
        ):
            # Only remove materials that look like STEP-generated ones
            # (named colors like "GRAY", hex like "STEP_808080", etc.)
            # Safety: only remove if truly zero users
            bpy.data.materials.remove(mat)
            removed += 1
    if removed:
        print(f"STEPper MatDB: Removed {removed} unused material(s)")
    return removed


def _populate_ui_mappings(stepper, mappings):
    """Fill the UI CollectionProperty from a mappings dict."""
    stepper.mat_db_mappings.clear()
    for orig, repl in sorted(mappings.items()):
        item = stepper.mat_db_mappings.add()
        item.original_name = orig
        item.replacement_name = repl


# ---------------------------------------------------------------------------
# PropertyGroup
# ---------------------------------------------------------------------------

class PG_MaterialMapping(bpy.types.PropertyGroup):
    """A single original-name -> replacement-material mapping entry."""

    original_name: bpy.props.StringProperty(name="Original Name")
    replacement_name: bpy.props.StringProperty(name="Replacement")


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class STEP_OT_MatDBCreate(bpy.types.Operator):
    """Create a new material database from the current scene"""

    bl_idname = "stepper.mat_db_create"
    bl_label = "Create Material Database"

    db_name: bpy.props.StringProperty(
        name="Database Name",
        description="Name for the new material database",
        default="material_database",
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        name = self.db_name.strip()
        if not name:
            self.report({"ERROR"}, "Database name cannot be empty")
            return {"CANCELLED"}

        filepath = os.path.join(_get_matdb_dir(), name + ".blend")

        mappings = _scan_scene_materials()
        if not mappings:
            self.report({"WARNING"}, "No STEP materials found in scene")
            return {"CANCELLED"}

        _write_material_database(filepath, mappings)

        # Set as active database and load into UI
        prefs = get_addon_prefs()
        prefs.active_matdb = name
        _populate_ui_mappings(context.scene.stepper, mappings)

        self.report({"INFO"}, f"Created '{name}' with {len(mappings)} mapping(s)")
        return {"FINISHED"}


class STEP_OT_MatDBDuplicate(bpy.types.Operator):
    """Duplicate the active material database with a new name"""

    bl_idname = "stepper.mat_db_duplicate"
    bl_label = "Duplicate Material Database"

    db_name: bpy.props.StringProperty(
        name="New Name",
        description="Name for the duplicated database",
        default="",
    )

    @classmethod
    def poll(cls, context):
        return bool(get_active_matdb_path())

    def invoke(self, context, event):
        prefs = get_addon_prefs()
        self.db_name = prefs.active_matdb + "_copy"
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        import shutil

        name = self.db_name.strip()
        if not name:
            self.report({"ERROR"}, "Database name cannot be empty")
            return {"CANCELLED"}

        src = get_active_matdb_path()
        dst = os.path.join(_get_matdb_dir(), name + ".blend")
        if os.path.exists(dst):
            self.report({"ERROR"}, f"Database '{name}' already exists")
            return {"CANCELLED"}

        shutil.copy2(src, dst)

        prefs = get_addon_prefs()
        prefs.active_matdb = name
        # Reload mappings from the copy
        mappings = _read_matdb_mappings(dst)
        _populate_ui_mappings(context.scene.stepper, mappings)

        self.report({"INFO"}, f"Duplicated to '{name}'")
        return {"FINISHED"}


class STEP_OT_MatDBRefresh(bpy.types.Operator):
    """Reload mappings from the active database file and append its materials"""

    bl_idname = "stepper.mat_db_refresh"
    bl_label = "Load Material Database"

    @classmethod
    def poll(cls, context):
        return bool(get_active_matdb_path())

    def execute(self, context):
        db_path = get_active_matdb_path()

        # Append materials first so they show up in the UI prop_search
        _append_matdb_materials(db_path)

        mappings = _read_matdb_mappings(db_path)
        if not mappings:
            self.report({"WARNING"}, "No mappings found in database")
            return {"CANCELLED"}

        _populate_ui_mappings(context.scene.stepper, mappings)
        self.report({"INFO"}, f"Loaded {len(mappings)} mapping(s)")
        return {"FINISHED"}


class STEP_OT_MatDBUpdate(bpy.types.Operator):
    """Add new original material names from the scene to the database"""

    bl_idname = "stepper.mat_db_update"
    bl_label = "Update Material Database"

    @classmethod
    def poll(cls, context):
        return len(context.scene.stepper.mat_db_mappings) > 0

    def execute(self, context):
        stepper = context.scene.stepper
        scene_mappings = _scan_scene_materials()

        # Existing originals in the UI list
        existing = {item.original_name for item in stepper.mat_db_mappings}

        added = 0
        for orig, repl in scene_mappings.items():
            if orig not in existing:
                item = stepper.mat_db_mappings.add()
                item.original_name = orig
                item.replacement_name = repl
                added += 1

        if added:
            self.report(
                {"INFO"},
                f"Added {added} new mapping(s). Press Save to write to database.",
            )
        else:
            self.report({"INFO"}, "No new materials to add")
        return {"FINISHED"}


class STEP_OT_MatDBSave(bpy.types.Operator):
    """Save current mappings to the active database file"""

    bl_idname = "stepper.mat_db_save"
    bl_label = "Save Material Database"

    @classmethod
    def poll(cls, context):
        return (
            bool(get_active_matdb_path())
            and len(context.scene.stepper.mat_db_mappings) > 0
        )

    def execute(self, context):
        stepper = context.scene.stepper
        filepath = get_active_matdb_path()

        mappings = {}
        for item in stepper.mat_db_mappings:
            mappings[item.original_name] = item.replacement_name

        _write_material_database(filepath, mappings)
        self.report({"INFO"}, f"Saved {len(mappings)} mapping(s) to database")
        return {"FINISHED"}


class STEP_OT_MatDBDelete(bpy.types.Operator):
    """Delete the active material database file"""

    bl_idname = "stepper.mat_db_delete"
    bl_label = "Delete Material Database"

    @classmethod
    def poll(cls, context):
        return bool(get_active_matdb_path())

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        prefs = get_addon_prefs()
        db_path = get_active_matdb_path()
        name = prefs.active_matdb

        try:
            os.remove(db_path)
        except OSError as e:
            self.report({"ERROR"}, f"Could not delete database: {e}")
            return {"CANCELLED"}

        prefs.active_matdb = "NONE"
        context.scene.stepper.mat_db_mappings.clear()
        self.report({"INFO"}, f"Deleted database '{name}'")
        return {"FINISHED"}


class STEP_OT_MatDBApply(bpy.types.Operator):
    """Apply material mappings from the active database to objects in the scene"""

    bl_idname = "stepper.mat_db_apply"
    bl_label = "Apply Material Database"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(get_active_matdb_path())

    def invoke(self, context, event):
        # Append materials / read mappings BEFORE the undo step
        db_path = get_active_matdb_path()
        self._mappings = ensure_matdb_materials(db_path)
        if not self._mappings:
            self.report({"WARNING"}, "No mappings found in database")
            return {"CANCELLED"}
        return self.execute(context)

    def execute(self, context):
        stepper = context.scene.stepper

        if stepper.mat_db_apply_selection_only:
            objects = list(context.selected_objects)
        else:
            objects = [
                obj for obj in bpy.data.objects if obj.get("STEP_file") is not None
            ]

        if not objects:
            self.report({"WARNING"}, "No STEP objects found")
            return {"CANCELLED"}

        replaced = apply_matdb_to_objects(objects, self._mappings)
        cleanup_unused_step_materials()
        self.report(
            {"INFO"}, f"Replaced {replaced} material(s) across {len(objects)} object(s)"
        )
        return {"FINISHED"}



# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class STEP_UL_MaterialMappings(bpy.types.UIList):
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_property, index
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            split = layout.split(factor=0.45, align=True)
            split.label(text=item.original_name + "  \u2192", icon="MATERIAL")
            split.prop_search(item, "replacement_name", bpy.data, "materials", text="")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.original_name)


class STEP_PT_MaterialDB(bpy.types.Panel):
    bl_label = "STEPper NEXT: Material DB"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STEPper NEXT"

    def draw(self, context):
        layout = self.layout
        stepper = context.scene.stepper
        prefs = get_addon_prefs()

        # Database selector
        box = layout.box()
        box.label(text="Database")
        row = box.row(align=True)
        row.prop(prefs, "active_matdb", text="")
        sub = row.row(align=True)
        sub.enabled = prefs.active_matdb != "NONE"
        sub.operator("stepper.mat_db_delete", text="", icon="TRASH")

        row = box.row(align=True)
        row.operator("stepper.mat_db_create", text="New", icon="ADD")
        sub = row.row(align=True)
        sub.enabled = prefs.active_matdb != "NONE"
        sub.operator("stepper.mat_db_duplicate", text="Duplicate", icon="DUPLICATE")
        row.operator("stepper.mat_db_refresh", text="Load", icon="IMPORT")

        # Mappings list
        if len(stepper.mat_db_mappings) > 0:
            box = layout.box()
            box.label(text="Material Mappings")
            box.template_list(
                "STEP_UL_MaterialMappings",
                "",
                stepper,
                "mat_db_mappings",
                stepper,
                "mat_db_active_index",
                rows=5,
            )
            row = box.row(align=True)
            row.operator("stepper.mat_db_update", text="Update", icon="FILE_REFRESH")
            row.operator("stepper.mat_db_save", text="Save", icon="EXPORT")

            row = box.row(align=True)
            row.operator("stepper.mat_db_apply", text="Apply", icon="CHECKMARK")
            row.prop(stepper, "mat_db_apply_selection_only")


classes = (
    PG_MaterialMapping,
    STEP_OT_MatDBCreate,
    STEP_OT_MatDBDuplicate,
    STEP_OT_MatDBRefresh,
    STEP_OT_MatDBUpdate,
    STEP_OT_MatDBSave,
    STEP_OT_MatDBDelete,
    STEP_OT_MatDBApply,
    STEP_UL_MaterialMappings,
    STEP_PT_MaterialDB,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in classes[::-1]:
        bpy.utils.unregister_class(c)
