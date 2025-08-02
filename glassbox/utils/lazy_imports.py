"""Utilities for optional imports."""
from importlib import import_module


def optional_import(module_name: str):
    """Attempt to import *module_name*.

    Raises an informative :class:`ImportError` if the module cannot be imported
    and hints how to install the optional dependency.
    """
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Install optional dependency via `pip install glassbox[{module_name}]`"
        ) from exc
