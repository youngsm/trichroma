"""Detector-specialized scene artifacts for the Triton optical backend.

The compiler deliberately lives outside :mod:`chroma.gpu`: it consumes the
reference Chroma geometry but emits plain NumPy structures with no CUDA or
PyCUDA dependency.
"""

from .compiler import (
    AnalyticBoxes,
    AnalyticWires,
    CanonicalPMT,
    CompiledReflect3WiresScene,
    OpticalTables,
    PMTInstances,
    Reachability,
    compile_reflect3wires_scene,
)
from .chroma_global_bvh import (
    TARGET_OPTICAL_SEMANTICS_SHA256,
    TARGET_TRAVERSAL_SHA256,
    ChromaGlobalBVHArtifact,
    ChromaGlobalBVHArtifactError,
    build_chroma_global_bvh_artifact,
    export_chroma_global_bvh_artifact,
    export_reflect3wires_chroma_global_bvh_artifact,
    load_chroma_global_bvh_artifact,
    load_reflect3wires_chroma_global_bvh_artifact,
    save_chroma_global_bvh_artifact,
)
from .chroma_global_traversal import (
    CHROMA_STACK_CAPACITY,
    DEFAULT_CERTIFICATE_RAY_TILE,
    ChromaGlobalBVHDevice,
    ChromaGlobalHit,
    ChromaGlobalTraversalUnavailable,
    ChromaGlobalTraversalWorkspace,
    nearest_chroma_global_hit,
    triton_chroma_global_available,
)

__all__ = [
    "AnalyticBoxes",
    "AnalyticWires",
    "CanonicalPMT",
    "CHROMA_STACK_CAPACITY",
    "DEFAULT_CERTIFICATE_RAY_TILE",
    "TARGET_OPTICAL_SEMANTICS_SHA256",
    "TARGET_TRAVERSAL_SHA256",
    "ChromaGlobalBVHArtifact",
    "ChromaGlobalBVHArtifactError",
    "ChromaGlobalBVHDevice",
    "ChromaGlobalHit",
    "ChromaGlobalTraversalUnavailable",
    "ChromaGlobalTraversalWorkspace",
    "CompiledReflect3WiresScene",
    "OpticalTables",
    "PMTInstances",
    "Reachability",
    "build_chroma_global_bvh_artifact",
    "compile_reflect3wires_scene",
    "export_chroma_global_bvh_artifact",
    "export_reflect3wires_chroma_global_bvh_artifact",
    "load_chroma_global_bvh_artifact",
    "load_reflect3wires_chroma_global_bvh_artifact",
    "nearest_chroma_global_hit",
    "save_chroma_global_bvh_artifact",
    "triton_chroma_global_available",
]
