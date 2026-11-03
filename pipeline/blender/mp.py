"""MPFB's services, reached the way MPFB's own test suite reaches them.

Blender 4.2+ extensions must import relatively, so the package's absolute name depends on the repository it
was installed into (bl_ext.user_default.mpfb here). dynamic_import finds the loaded module whose name ends
with the given package path and returns one of its attributes.
"""
import importlib
import sys

import addon_utils


def _ensure_loaded():
    if any(name.endswith(".mpfb") for name in sys.modules):
        return
    for mod in addon_utils.modules():
        if mod.__name__.endswith(".mpfb"):
            addon_utils.enable(mod.__name__, default_set=True, persistent=True)
            return
    raise RuntimeError("MPFB is not installed (blender --command extension install-file -r user_default -e ...)")


def dynamic_import(absolute_package_str, key):
    for amod in list(sys.modules):
        if amod.endswith(absolute_package_str):
            mod = importlib.import_module(amod)
            if not hasattr(mod, key):
                raise AttributeError(f"Module {amod} does not have attribute {key}")
            return getattr(mod, key)
    raise ValueError(f"No module found with name ending in {absolute_package_str}")


_ensure_loaded()
INFO = dynamic_import("mpfb", "MPFB_CONTEXTUAL_INFORMATION")
S = INFO["SERVICES"]
AssetService = S["AssetService"]
ExportService = S["ExportService"]
HumanService = S["HumanService"]
LocationService = S["LocationService"]
ObjectService = S["ObjectService"]
RigService = S["RigService"]
TargetService = S["TargetService"]
