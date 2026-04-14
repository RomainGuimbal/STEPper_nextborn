import bpy

def get_addon_prefs():
    """Return the addon preferences instance."""
    return bpy.context.preferences.addons[__package__].preferences