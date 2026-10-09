"""Compatibility shim for running cashocs 1.3 on Ubuntu's FEniCS packages.

Ubuntu 24.04 ships legacy FEniCS (python3-dolfin) with UFL renamed to
``ufl_legacy``. cashocs 1.3 imports ``ufl`` and checks ``obj.__module__ ==
"ufl.form"``. This module exposes ``ufl_legacy`` under the name ``ufl``.
It does nothing if a regular ``ufl`` package is installed (e.g. conda FEniCS).
"""

import importlib
import inspect
import pkgutil
import sys

try:
    import ufl  # noqa: F401
except ImportError:
    import ufl_legacy
    sys.modules["ufl"] = ufl_legacy
    for info in pkgutil.walk_packages(ufl_legacy.__path__, "ufl_legacy."):
        try:
            sys.modules[info.name.replace("ufl_legacy", "ufl", 1)] = importlib.import_module(info.name)
        except Exception:
            pass
    for name, module in list(sys.modules.items()):
        if name.startswith("ufl_legacy"):
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__.startswith("ufl_legacy"):
                    cls.__module__ = cls.__module__.replace("ufl_legacy", "ufl", 1)
