"""Shim package to expose the inner pearl_pow_kernels module from the zip layout."""

from __future__ import annotations

import os as _os
from pkgutil import extend_path as _extend_path

# Allow importing modules from the nested pearl_pow_kernels/ directory.
__path__ = _extend_path(__path__, __name__)
_inner = _os.path.join(_os.path.dirname(__file__), "pearl_pow_kernels")
if _inner not in __path__:
    __path__.append(_inner)

# Re-export inner package symbols for convenience.
from .pearl_pow_kernels import *  # noqa: F401,F403
from .pearl_pow_kernels import __all__ as _inner_all

__all__ = list(_inner_all)
