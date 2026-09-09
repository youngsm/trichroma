#include "Reflect3WiresGeometry.hh"

#include "G4GDMLParser.hh"
#include "G4StateManager.hh"
#include "G4ios.hh"

#include <fstream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    G4cerr << "usage: generate_reflect3wires_gdml OUTPUT.gdml [METADATA.json]\n";
    return 2;
  }
  const std::string gdmlPath = argv[1];
  const std::string metadataPath = argc == 3 ? argv[2] : gdmlPath + ".json";

  chroma_lar_geant4::GeometrySummary summary;
  auto* world = chroma_lar_geant4::BuildReflect3WiresGeometry(summary);
  G4GDMLParser parser;
  parser.SetOutputFileOverwrite(true);
  parser.Write(gdmlPath, world, true);

  std::ofstream output(metadataPath);
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"source_config\": \"detector_config_reflect_reflect3wires.py\",\n"
         << "  \"wavelength_nm\": 450.0,\n"
         << "  \"pmt_count\": " << summary.pmtCount << ",\n"
         << "  \"photocathode_channels\": "
         << summary.photocathodeChannels << ",\n"
         << "  \"wire_plane_count\": " << summary.wirePlaneCount << ",\n"
         << "  \"wire_cylinder_count\": " << summary.wireCount << ",\n"
         << "  \"active_dimensions_mm\": [4320.0, 4320.0, 4320.0],\n"
         << "  \"cavity_scale\": 1.5,\n"
         << "  \"wire_pitch_mm\": 3.0,\n"
         << "  \"wire_diameter_mm\": 0.15,\n"
         << "  \"pmt_diameter_in\": 4.38,\n"
         << "  \"pmt_profile\": \"exact Chroma axial profile, analytic azimuth\"\n"
         << "}\n";

  G4cout << "wrote " << gdmlPath << " with " << summary.pmtCount
         << " PMTs and " << summary.wireCount << " clipped analytic cylinders\n";
  return output ? 0 : 1;
}

