"""Settings for the Chroma EventViewer:
photons_max will randomly select at most that many photons
photons_max_steps truncates all photon tracks to that number of steps
photons_only_type can be set to 'cher', 'scint', or 'reemit'
photons_detected_only will show only detected photon tracks
photons_track_size controls the track size of the photons"""

viewer_photons_max = 500
viewer_photons_max_steps = 20
viewer_photons_only_type = None
viewer_photons_detected_only = False
viewer_photons_track_size = 0.1

__exports__ = [
    "viewer_photons_max",
    "viewer_photons_max_steps",
    "viewer_photons_only_type",
    "viewer_photons_detected_only",
    "viewer_photons_track_size",
]