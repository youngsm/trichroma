#pragma once

#include "globals.hh"

class G4VPhysicalVolume;

namespace chroma_lar_geant4 {

struct GeometrySummary {
  G4int pmtCount = 0;
  G4int wireCount = 0;
  G4int wirePlaneCount = 0;
  G4int photocathodeChannels = 0;
};

// Build the full, two-sided detector represented by
// detector_config_reflect_reflect3wires.py.  The returned world owns the
// complete Geant4 geometry tree and all optical border/skin surfaces.
G4VPhysicalVolume* BuildReflect3WiresGeometry(GeometrySummary& summary);

}  // namespace chroma_lar_geant4

