"""
Configuration loader for detector configurations.

This module provides functionality to load detector configurations from Python files
and build detector objects based on those configurations.

The ``detector_type`` key in the config dict (typically set in the YAML) selects
which builder to use:

- ``"wire"``  → ``build_larcube.build_detector``
- ``"pixel"`` → ``build_larcube_pixel.build_detector``

If ``detector_type`` is absent, ``"wire"`` is assumed for backward compatibility.
"""

import importlib.util
import sys
import os
from typing import Dict, Any

from .build_larcube import build_detector as build_wire_detector
from .build_larcube_pixel import build_detector as build_pixel_detector

_BUILDERS = {
    "wire": build_wire_detector,
    "pixel": build_pixel_detector,
}


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration from a Python file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file, or a module name under ``chroma_lar.config``.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    # try to import as module first
    try:
        config_module = importlib.import_module('chroma_lar.config.' + config_path)
    except ImportError:
        # fallback to loading from file path
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load configuration file: {config_path}")

        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config_module
        spec.loader.exec_module(config_module)

    if not hasattr(config_module, "get_config"):
        raise AttributeError(
            f"Configuration file {config_path} must define a get_config() function")

    return config_module.get_config()


def build_detector_from_config(config_path: str, **kwargs):
    """
    Build a detector from a configuration file.

    The ``detector_type`` key in the config (or overridden via *kwargs*)
    selects the builder: ``"wire"`` or ``"pixel"``.  Defaults to ``"wire"``
    if not specified.

    Parameters
    ----------
    config_path : str
        Path or module name for the configuration file.
    **kwargs
        Additional parameters to override configuration values.

    Returns
    -------
    detector.Detector
    """
    try:
        _config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", config_path + '.py')
        config = load_config_from_file(_config_path)
    except Exception:
        config = load_config_from_file(config_path)

    config.update(kwargs)

    detector_type = config.pop("detector_type", "wire")
    builder = _BUILDERS.get(detector_type)
    if builder is None:
        raise ValueError(
            f"Unknown detector_type '{detector_type}'. "
            f"Available types: {list(_BUILDERS.keys())}")
    return builder(**config)


def build_detector_from_dict(config: Dict[str, Any]):
    """
    Build a detector from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.  Must contain ``detector_type``
        (``"wire"`` or ``"pixel"``), or defaults to ``"wire"``.

    Returns
    -------
    detector.Detector
    """
    config = dict(config)  # copy to avoid mutating caller's dict
    detector_type = config.pop("detector_type", "wire")
    builder = _BUILDERS.get(detector_type)
    if builder is None:
        raise ValueError(
            f"Unknown detector_type '{detector_type}'. "
            f"Available types: {list(_BUILDERS.keys())}")
    return builder(**config)
