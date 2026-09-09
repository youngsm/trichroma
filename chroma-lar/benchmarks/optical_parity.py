"""Strict matched-seed comparisons with persistent failure evidence."""

import json
from pathlib import Path

import numpy as np


def compare_results(actual, reference, *, evidence, wavelength_atol=0.0):
    """Check terminal outcomes/hits; save states before raising on disagreement."""
    try:
        np.testing.assert_array_equal(actual.final_state["flags"], reference.final_state["flags"])
        for name in ("photon_ids", "channels"):
            np.testing.assert_array_equal(getattr(actual.hits, name), getattr(reference.hits, name))
        if wavelength_atol:
            np.testing.assert_allclose(
                actual.hits.wavelengths, reference.hits.wavelengths, rtol=1e-6, atol=wavelength_atol
            )
        else:
            np.testing.assert_array_equal(actual.hits.wavelengths, reference.hits.wavelengths)
        np.testing.assert_allclose(actual.hits.times, reference.hits.times, rtol=1e-5, atol=0.003)
        if actual.step_limit_count or reference.step_limit_count:
            raise AssertionError("comparison contains unfinished photons")
    except AssertionError as error:
        evidence = Path(evidence)
        evidence.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            evidence.with_suffix(".npz"),
            **{
                f"{prefix}_{key}": value
                for prefix, result in (("cpu", reference), ("gpu", actual))
                for key, value in result.final_state.items()
            },
        )
        evidence.with_suffix(".json").write_text(
            json.dumps(
                {
                    "error": str(error),
                    "flag_mismatch_indices": np.flatnonzero(
                        actual.final_state["flags"] != reference.final_state["flags"]
                    ).tolist(),
                    "cpu_detected": len(reference.hits),
                    "gpu_detected": len(actual.hits),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    return {
        "detected": len(actual.hits),
        "flag_mismatches": 0,
        "exact": "terminal flags, detected photon IDs, channels"
        + (" and wavelengths" if not wavelength_atol else ""),
        "hit_time_rtol": 1e-5,
        "hit_time_atol_ns": 0.003,
        "wavelength_rtol": 1e-6 if wavelength_atol else 0.0,
        "wavelength_atol_nm": wavelength_atol,
        "maximum_hit_time_difference_ns": float(
            np.max(np.abs(actual.hits.times - reference.hits.times), initial=0.0)
        ),
        "maximum_hit_wavelength_difference_nm": float(
            np.max(np.abs(actual.hits.wavelengths - reference.hits.wavelengths), initial=0.0)
        ),
    }
