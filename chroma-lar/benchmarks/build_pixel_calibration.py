"""Reproduce the synthetic pixel validation bundle from repository inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from chroma_lar.config.detector_config_pixel import get_config
from chroma_lar.geometry.pixelplane import compute_averaged_pixel_surface

DATA = Path(__file__).parent / "optical_validation"


def pixel_calibration():
    manifest = json.loads((DATA / "full_detector_synthetic_calibration_noise.json").read_text())
    config = get_config()
    surface = compute_averaged_pixel_surface(
        config["pixel_surface"],
        config["pcb_surface"],
        config["pixel_pad_size"],
        config["pixel_chamfer_radius"],
        config["pixel_pitch"],
        shield_surface=config["shield_surface"],
    )
    grid = manifest["wavelength_grid_nm"]
    wavelengths = np.arange(grid["start"], grid["stop"] + grid["step"], grid["step"])
    manifest["surfaces"]["averaged_pixel"] = {
        prop: np.column_stack(
            (wavelengths, np.interp(wavelengths, values[:, 0], values[:, 1]))
        ).tolist()
        for prop in ("detect", "absorb", "reflect_diffuse", "reflect_specular")
        for values in [getattr(surface, prop)]
    }
    manifest["provenance"] = (
        "Synthetic pixel-TPC validation. LAr, TPB, PMT and readout inputs inherited from "
        "full_detector_synthetic_calibration_noise.json; averaged pixel optical properties sampled "
        "from the repository pixel_simplified=True configuration. Not measured calibration."
    )
    manifest["pixel_model"] = {
        "simplified": bool(config["pixel_simplified"]),
        "pixels_per_face": config["n_pixels_y"] * config["n_pixels_z"],
        "pitch_mm": config["pixel_pitch"],
        "pad_size_mm": config["pixel_pad_size"],
        "chamfer_radius_mm": config["pixel_chamfer_radius"],
        "non_gold_fr4_fraction": 0.5,
        "non_gold_copper_fraction": 0.5,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DATA / "primitive_regions/pixel_synthetic_calibration.json"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pixel_calibration(), indent=2) + "\n")


if __name__ == "__main__":
    main()
