"""Curved R5912 PMT camera fixture using repository optical calibration tables."""

import hashlib
from pathlib import Path

import numpy as np


def build_pmt_camera_scene():
    from chroma import make
    from chroma.detector import Detector
    from chroma.geometry import Solid, Surface
    from .geometry.pmt import build_r5912_pmt
    from .optical_calibration import OpticalCalibration
    from .photon_camera import CameraScene

    calibration_path = (
        Path(__file__).parents[1]
        / "benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json"
    )
    calibration = OpticalCalibration.load(calibration_path)
    wavelengths = np.arange(120, 741, 2, dtype=np.float32)
    # Existing synthetic calibration ends at 500nm. Preserve every tabulated
    # curve within that range and extend its endpoint value into the red fill.
    for material in calibration.materials.values():
        for field in (
            "refractive_index",
            "absorption_length",
            "scattering_length",
            "group_velocity",
        ):
            if not hasattr(material, field):
                continue
            table = getattr(material, field)
            material.set(field, np.interp(wavelengths, table[:, 0], table[:, 1]), wavelengths)
    for surface in calibration.surfaces.values():
        for field in (
            "detect",
            "absorb",
            "reflect_diffuse",
            "reflect_specular",
            "reemit",
            "reemission_cdf",
        ):
            table = getattr(surface, field)
            surface.set(field, np.interp(wavelengths, table[:, 0], table[:, 1]), wavelengths)
    lar, glass, vacuum = [
        calibration.materials[name] for name in ("liquid_argon", "glass", "vacuum")
    ]
    tpb = calibration.surfaces["validation_tpb"]
    pmt = build_r5912_pmt(
        glass_thickness=3,
        nzsteps=20,
        nsteps=48,
        diameter=203.2,
        outer_material=lar,
        glass=glass,
        vacuum=vacuum,
        photocathode_surface=calibration.surfaces["perfect_pmt_photocathode"],
        back_surface=calibration.surfaces["glossy_surface"],
        coating_surface=tpb,
    )
    geometry = Detector(lar)
    monitor = Surface("ideal_boundary_monitor")
    monitor.set("detect", 1.0, wavelengths)
    geometry.add_pmt(
        Solid(make.box(600, 400, 240), lar, vacuum, surface=monitor), displacement=(0, 0, 0)
    )
    geometry.add_pmt(
        pmt, rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), displacement=(20, 0, -15)
    )
    geometry.flatten()
    return CameraScene(
        name="pmt",
        title="TPB-coated PMT",
        geometry=geometry,
        wavelengths=wavelengths,
        source_center=(-260, 0, -15),
        source_band=(126, 130),
        outline=None,
        explanation="R5912 profile · VUV-to-visible TPB conversion · glass Fresnel transport · photocathode absorption",
        fluorescence_time=tpb.reemission_time_cdf,
        beam_width=60,
    )


def pmt_camera_metadata(scene):
    """Facet charts are exactly area-normalized on the exported triangle mesh."""
    from .geometry import pmt

    surface = scene.host.surface_index
    names = scene.host.optics.surfaces.names
    role = np.full(len(surface), 3, np.int32)
    for name, value in (
        ("ideal_boundary_monitor", 0),
        ("validation_tpb", 1),
        ("perfect_pmt_photocathode", 2),
    ):
        role[surface == list(names).index(name)] = value
    chart = np.full(len(surface), -1, np.int32)
    chart[role == 1] = np.arange(np.count_nonzero(role == 1))
    triangles = scene.bvh.triangle_vertices.astype(float)
    area = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        / 2
    )
    profile = Path(pmt.__file__).parents[2] / "data/pmt_contour_mm.csv"
    calibration = (
        Path(__file__).parents[1]
        / "benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json"
    )
    return dict(
        triangle_role=role.tolist(),
        triangle_chart=chart.tolist(),
        triangle_area_mm2=area.tolist(),
        roles={"wall": 0, "wls": 1, "photocathode": 2, "ordinary": 3},
        wls_charts=int(np.count_nonzero(role == 1)),
        chart_layout="(2*triangle_chart+outgoing_hemisphere)*wavelength_bins+bin",
        clip_plane_normal=[0, 1, 0],
        clip_plane_offset=0,
        keep_positive=True,
        cutaway_scope="Camera-only visibility cut; every forward photon sees the complete PMT. No luminous cut cap is added.",
        ultraviolet=dict(
            bin=0,
            effective_wavelength_nm=128,
            default_visible=False,
            optional_display="Explicitly labeled UV false color; not visible radiance",
        ),
        nominal_diameter_mm=203.2,
        glass_thickness_mm=3,
        profile_rings=20,
        azimuth_segments=48,
        tessellation_scope="The camera demonstration resamples the same R5912 profile with20 rings and48 azimuth segments. This changes its polygonal approximation; it is not exact subdivision of the coarse facets. Forward photons and camera queries use this same mesh, independently of the original detector geometry.",
        profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        calibration_status="synthetic",
        calibration_scope="Repository full-detector software-validation tables, not measured PMT calibration. Endpoint values above500nm are extended for the visible ceiling source.",
        source_scope="126–130nm narrowband VUV beam; this is not a full scintillation event",
    )
