import numpy as np
import pytest

from chroma.geometry import AngularProps, DichroicProps, Material, Surface
from chroma.triton.optics import (
    OpticalFeature,
    SURFACE_ANGULAR,
    SURFACE_COMPLEX,
    SURFACE_DEFAULT,
    SURFACE_DICHROIC,
    SURFACE_WLS,
    compile_optical_tables,
)


def _material(name="material", wavelengths=(300.0, 400.0, 500.0)):
    material = Material(name)
    material.set("refractive_index", (1.2, 1.3, 1.4), wavelengths)
    material.set("absorption_length", (10.0, 20.0, 30.0), wavelengths)
    material.set("scattering_length", (100.0, 200.0, 300.0), wavelengths)
    return material


def _property(coordinates, values):
    return np.asarray(list(zip(coordinates, values)), dtype=np.float32)


def test_compiler_matches_gpu_geometry_interpolation_and_clamping():
    material = _material(wavelengths=(300.0, 400.0, 500.0))
    wavelengths = np.arange(200.0, 601.0, 100.0, dtype=np.float32)
    times = np.arange(0.0, 3.0, 1.0, dtype=np.float32)

    tables = compile_optical_tables(
        [material], wavelengths=wavelengths, times=times
    )

    expected = np.interp(
        wavelengths,
        material.refractive_index[:, 0],
        material.refractive_index[:, 1],
    ).astype(np.float32)
    np.testing.assert_array_equal(tables.materials.refractive_index[0], expected)
    assert tables.materials.refractive_index[0, 0] == np.float32(1.2)
    assert tables.materials.refractive_index[0, -1] == np.float32(1.4)
    assert tables.wavelength_grid.start == np.float32(200.0)
    assert tables.wavelength_grid.step == np.float32(100.0)
    assert tables.wavelength_grid.count == 5
    assert tables.materials.refractive_index.flags.c_contiguous
    assert not tables.materials.refractive_index.flags.writeable
    with pytest.raises(ValueError):
        tables.materials.refractive_index[0, 0] = 2.0


def test_bulk_reemission_components_use_ragged_offsets_and_time_grid():
    wavelengths = np.arange(300.0, 501.0, 100.0, dtype=np.float32)
    times = np.arange(0.0, 3.0, 1.0, dtype=np.float32)
    first = _material("with-components")
    second = _material("without-components")

    first.comp_reemission_prob = [
        _property(wavelengths, (0.2, 0.3, 0.4)),
        _property(wavelengths, (0.8, 0.7, 0.6)),
    ]
    first.comp_reemission_wvl_cdf = [
        _property(wavelengths, (0.0, 0.4, 1.0)),
        _property(wavelengths, (0.0, 0.7, 1.0)),
    ]
    first.comp_reemission_time_cdf = [
        _property(times, (0.0, 0.6, 1.0)),
        _property(times, (0.0, 0.2, 1.0)),
    ]
    first.comp_absorption_length = [
        _property(wavelengths, (20.0, 30.0, 40.0)),
        _property(wavelengths, (40.0, 50.0, 60.0)),
    ]

    tables = compile_optical_tables(
        [first, second], wavelengths=wavelengths, times=times
    )
    materials = tables.materials

    np.testing.assert_array_equal(materials.component_offsets, (0, 2, 2))
    assert materials.component_reemission_prob.shape == (2, 3)
    assert materials.component_reemission_wavelength_cdf.shape == (2, 3)
    assert materials.component_reemission_time_cdf.shape == (2, 3)
    assert materials.component_absorption_length.shape == (2, 3)
    np.testing.assert_array_equal(
        materials.component_reemission_time_cdf[1],
        np.asarray((0.0, 0.2, 1.0), dtype=np.float32),
    )
    assert tables.feature_mask & OpticalFeature.BULK_REEMISSION
    assert tables.feature_mask & OpticalFeature.TIME_REEMISSION
    assert tables.feature_mask & OpticalFeature.WAVELENGTH_DEPENDENT


def test_all_surface_models_compile_to_common_and_ragged_tables():
    wavelengths = np.arange(300.0, 501.0, 100.0, dtype=np.float32)
    times = np.arange(0.0, 3.0, 1.0, dtype=np.float32)
    material = _material()

    default = Surface("default", model=SURFACE_DEFAULT)
    default.set("detect", 0.1, wavelengths)
    default.set("absorb", 0.2, wavelengths)
    default.set("reflect_diffuse", 0.3, wavelengths)
    default.set("reflect_specular", 0.4, wavelengths)

    complex_surface = Surface("complex", model=SURFACE_COMPLEX)
    complex_surface.set("detect", 0.25, wavelengths)
    complex_surface.set("eta", (1.4, 1.5, 1.6), wavelengths)
    complex_surface.set("k", (0.1, 0.2, 0.3), wavelengths)
    complex_surface.thickness = 12.5
    complex_surface.transmissive = 1

    wls = Surface("wls", model=SURFACE_WLS)
    wls.set("absorb", 0.6, wavelengths)
    wls.set("reemit", 0.8, wavelengths)
    wls.set("reflect_specular", 0.2, wavelengths)
    wls.set("reemission_cdf", (0.0, 0.5, 1.0), wavelengths)

    dichroic = Surface("dichroic", model=SURFACE_DICHROIC)
    dichroic.dichroic_props = DichroicProps(
        angles=(0.0, 0.5, 1.0),
        reflect=(
            _property(wavelengths, (0.1, 0.2, 0.3)),
            _property(wavelengths, (0.2, 0.3, 0.4)),
            _property(wavelengths, (0.3, 0.4, 0.5)),
        ),
        transmit=(
            _property(wavelengths, (0.7, 0.6, 0.5)),
            _property(wavelengths, (0.6, 0.5, 0.4)),
            _property(wavelengths, (0.5, 0.4, 0.3)),
        ),
    )

    angular = Surface("angular", model=SURFACE_ANGULAR)
    angular.angular_props = AngularProps(
        angles=(0.0, 0.5, 1.0),
        transmit=(0.2, 0.3, 0.4),
        reflect_specular=(0.3, 0.2, 0.1),
        reflect_diffuse=(0.1, 0.1, 0.1),
    )

    tables = compile_optical_tables(
        [material],
        [default, complex_surface, wls, dichroic, angular, None],
        wavelengths=wavelengths,
        times=times,
    )
    surfaces = tables.surfaces

    np.testing.assert_array_equal(surfaces.model, (0, 1, 2, 3, 4, 0))
    np.testing.assert_array_equal(
        surfaces.present, (True, True, True, True, True, False)
    )
    np.testing.assert_array_equal(
        surfaces.dichroic_offsets, (0, 0, 0, 0, 3, 3, 3)
    )
    np.testing.assert_array_equal(
        surfaces.angular_offsets, (0, 0, 0, 0, 0, 3, 3)
    )
    assert surfaces.dichroic_reflect.shape == (3, 3)
    assert surfaces.dichroic_transmit.shape == (3, 3)
    assert surfaces.angular_transmit.shape == (3,)
    assert surfaces.thickness[1] == np.float32(12.5)
    assert surfaces.transmissive[1] == 1
    assert surfaces.names[-1] == "<none>"

    expected_features = (
        OpticalFeature.DEFAULT_SURFACE
        | OpticalFeature.COMPLEX_SURFACE
        | OpticalFeature.WLS_SURFACE
        | OpticalFeature.DICHROIC_SURFACE
        | OpticalFeature.ANGULAR_SURFACE
        | OpticalFeature.SURFACE_REEMISSION
        | OpticalFeature.TRANSMISSIVE_SURFACE
        | OpticalFeature.WAVELENGTH_DEPENDENT
    )
    assert tables.feature_mask == expected_features


def test_compiler_rejects_ambiguous_or_invalid_optical_data():
    material = _material()
    with pytest.raises(ValueError, match="equally spaced"):
        compile_optical_tables(
            [material],
            wavelengths=np.asarray((300.0, 400.0, 550.0), dtype=np.float32),
            times=np.asarray((0.0, 1.0, 2.0), dtype=np.float32),
        )

    material.comp_reemission_prob = [
        _property((300.0, 500.0), (0.5, 0.5))
    ]
    with pytest.raises(ValueError, match="equal lengths"):
        compile_optical_tables(
            [material],
            wavelengths=np.asarray((300.0, 400.0, 500.0), dtype=np.float32),
            times=np.asarray((0.0, 1.0, 2.0), dtype=np.float32),
        )

    missing_props = Surface("missing-dichroic", model=SURFACE_DICHROIC)
    with pytest.raises(ValueError, match="requires dichroic_props"):
        compile_optical_tables(
            [_material()],
            [missing_props],
            wavelengths=np.asarray((300.0, 400.0, 500.0), dtype=np.float32),
            times=np.asarray((0.0, 1.0, 2.0), dtype=np.float32),
        )
