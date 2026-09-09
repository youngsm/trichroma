"""Compare an optical pipeline NPZ with matched single-PE measurements.

Observed NPZ keys: times_ns, channels, emitted_photons. The observation must
use the same source distribution, timing origin, channel map, acquisition
window, and signal definition. Background subtraction and uncertainty in the
emitted-light normalization belong to the experimental analysis. This script
reports descriptive differences, not a universal physics acceptance test.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import ks_2samp


def compare_samples(sim_times, sim_channels, sim_emitted, obs_times, obs_channels, obs_emitted):
    sim_times, obs_times = np.asarray(sim_times, float), np.asarray(obs_times, float)
    sim_channels, obs_channels = np.asarray(sim_channels), np.asarray(obs_channels)
    for times, channels, emitted in ((sim_times, sim_channels, sim_emitted), (obs_times, obs_channels, obs_emitted)):
        if times.ndim != 1 or channels.shape != times.shape or not np.isfinite(times).all():
            raise ValueError("time/channel arrays must be matching finite vectors")
        if not np.issubdtype(channels.dtype, np.integer) or np.any(channels < 0):
            raise ValueError("channel IDs must be nonnegative integers")
        if not np.isfinite(emitted) or emitted <= 0:
            raise ValueError("emitted-photon normalization must be finite and positive")
    result = {
        "simulation_pe_per_emitted_photon": len(sim_times)/sim_emitted,
        "observed_pe_per_emitted_photon": len(obs_times)/obs_emitted,
        "channels": {},
        "interpretation": "Descriptive comparison. No automatic acceptance threshold or fitted calibration.",
    }
    for channel in np.union1d(sim_channels, obs_channels):
        a, b = sim_times[sim_channels == channel], obs_times[obs_channels == channel]
        entry = {"simulation_count": len(a), "observed_count": len(b),
                 "simulation_pe_per_emitted_photon": len(a)/sim_emitted,
                 "observed_pe_per_emitted_photon": len(b)/obs_emitted}
        if len(a) and len(b):
            entry["time_ks_distance"] = float(ks_2samp(a, b).statistic)
            entry["time_quantiles"] = [.1, .5, .9, .99]
            entry["simulation_time_quantiles_ns"] = np.quantile(a, entry["time_quantiles"]).tolist()
            entry["observed_time_quantiles_ns"] = np.quantile(b, entry["time_quantiles"]).tolist()
        result["channels"][str(int(channel))] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.simulation, allow_pickle=False) as sim, np.load(args.observed, allow_pickle=False) as obs:
        metadata = json.loads(str(sim["metadata_json"]))
        result = compare_samples(sim["pe_times_ns"], sim["pe_channels"], metadata["photons"],
                                 obs["times_ns"], obs["channels"], float(obs["emitted_photons"]))
        result["simulation_metadata"] = metadata
        result["observation_file"] = args.observed
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
