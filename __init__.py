# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2021 Tommi Hyppänen
#
# Modified 2025 Romain Guimbal
#
# Modified 2026 by Peak-Design:
#   - Fixed error in importing files with only single part with tree hierarchy option enabled
#   - Added failed parts popup and import diagnostics
#   - Fixed tessellation race conditions and corrupt STEP handling
#   - Added ShapeFix healing for shapes with corrupted/missing geometry
#   - Fixed crash: validate face triangulations before native C++ extraction
#   - Renamed to STEPper NEXT, auto-apply scale, skip empty objects (v2.1.3)
#   - Material database system, multi-user scale fix (v2.2.0)

INSIDE_BLENDER = True
try:
    import bpy
except ModuleNotFoundError:
    print("Stepper not running inside Blender.")
    INSIDE_BLENDER = False


if INSIDE_BLENDER:
    # Normally don't do import star, but here it's basically a file concatenation
    # File concatenation is because the test framework breaks on __init__.py import bpy
    from .main import *  # noqa: F403
