#include "Reflect3WiresGeometry.hh"

#include "G4Box.hh"
#include "G4Element.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MultiUnion.hh"
#include "G4NistManager.hh"
#include "G4OpticalSurface.hh"
#include "G4PVPlacement.hh"
#include "G4PhysicalConstants.hh"
#include "G4Polycone.hh"
#include "G4RotationMatrix.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4Tubs.hh"
#include "G4VPhysicalVolume.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace chroma_lar_geant4 {
namespace {

constexpr G4double kActiveLength = 4320.0 * mm;
constexpr G4double kActiveHalf = kActiveLength / 2.0;
constexpr G4double kCavityLength = 6480.0 * mm;
constexpr G4double kPmtGap = 10.0 * mm;
constexpr G4double kPmtDiameter = 4.38 * 25.4 * mm;
constexpr G4double kPmtRadius = kPmtDiameter / 2.0;
constexpr G4double kPmtSpacing = 471.0 * mm;
constexpr G4double kWirePitch = 3.0 * mm;
constexpr G4double kWireRadius = 0.075 * mm;
constexpr G4double kCathodeHalfThickness = 3.0 * mm;
constexpr G4double kPmtBack = 125.414730 * mm;
constexpr G4double kActiveXHalf = 2160.0 * mm + kPmtGap + kPmtBack;
constexpr G4double kPhotonWavelength = 450.0 * nm;

struct ProfilePoint {
  G4double r;
  G4double z;
};

// This is the exact nzsteps=20, 4.38-inch R5912 profile produced by
// build_r5912_pmt().  G4Polycone revolves it analytically instead of using
// Chroma's 64-sided azimuthal tessellation.
const std::array<ProfilePoint, 22> kOuterProfile{{
    {0.000000, -125.414730}, {26.321610, -125.414730},
    {26.321610, -82.061490}, {31.037190, -78.452049},
    {35.876042, -75.603855}, {40.587076, -72.662419},
    {44.507562, -68.568956}, {48.358380, -64.669912},
    {51.338115, -59.920534}, {53.614518, -54.768797},
    {54.877320, -49.082065}, {55.297589, -43.216393},
    {55.076389, -37.440600}, {53.970426, -31.672851},
    {51.408172, -25.749956}, {46.563981, -19.578561},
    {41.161856, -14.176571}, {34.888394, -9.481185},
    {27.418289, -5.529764}, {18.231683, -2.433175},
    {10.425775, -0.692634}, {0.000000, 0.000000},
}};

const std::array<ProfilePoint, 22> kInnerProfile{{
    {0.000000, -123.772230}, {24.679110, -123.772230},
    {24.679110, -81.250277}, {30.118061, -77.087148},
    {35.024362, -74.199252}, {39.542942, -71.377979},
    {43.330071, -67.423757}, {47.062947, -63.644132},
    {49.883710, -59.148138}, {52.046362, -54.253832},
    {53.247662, -48.844059}, {53.652838, -43.189048},
    {53.439852, -37.627753}, {52.391295, -32.159380},
    {49.985015, -26.597037}, {45.332850, -20.670277},
    {40.083560, -15.421118}, {34.006947, -10.873065},
    {26.767738, -7.043779}, {17.789395, -4.017391},
    {10.191234, -2.323173}, {0.000000, -1.638887},
}};

constexpr G4double kPhotocathodeThreshold = -26.985180 * mm;

void SetConstProperty(G4MaterialPropertiesTable* table, const char* name,
                      G4double value) {
  std::array<G4double, 2> energy{{
      2.0 * pi * hbarc / (451.0 * nm),
      2.0 * pi * hbarc / (449.0 * nm),
  }};
  std::array<G4double, 2> values{{value, value}};
  table->AddProperty(name, energy.data(), values.data(), energy.size());
}

G4RotationMatrix* RotationFromLocalZ(const G4ThreeVector& requestedAxis) {
  const G4ThreeVector z = requestedAxis.unit();
  const G4ThreeVector reference =
      std::abs(z.z()) < 0.9 ? G4ThreeVector(0.0, 0.0, 1.0)
                            : G4ThreeVector(0.0, 1.0, 0.0);
  const G4ThreeVector x = reference.cross(z).unit();
  const G4ThreeVector y = z.cross(x).unit();
  return new G4RotationMatrix(x, y, z);
}

G4Material* MakeMaterial(const G4String& name, G4double density,
                         G4double refractiveIndex,
                         G4double absorptionLength,
                         G4double rayleighLength,
                         const std::vector<std::pair<G4Element*, G4double>>& massFractions) {
  auto* material = new G4Material(name, density, massFractions.size());
  for (const auto& component : massFractions) {
    material->AddElement(component.first, component.second);
  }
  auto* table = new G4MaterialPropertiesTable;
  SetConstProperty(table, "RINDEX", refractiveIndex);
  SetConstProperty(table, "ABSLENGTH", absorptionLength);
  SetConstProperty(table, "RAYLEIGH", rayleighLength);
  material->SetMaterialPropertiesTable(table);
  return material;
}

G4OpticalSurface* MakeMetalSurface(const G4String& name, G4double reflectivity,
                                   G4double efficiency,
                                   G4double specularSpike = 1.0) {
  auto* surface = new G4OpticalSurface(name, unified, polished, dielectric_metal);
  auto* table = new G4MaterialPropertiesTable;
  SetConstProperty(table, "REFLECTIVITY", reflectivity);
  SetConstProperty(table, "EFFICIENCY", efficiency);
  SetConstProperty(table, "SPECULARSPIKECONSTANT", specularSpike);
  SetConstProperty(table, "SPECULARLOBECONSTANT", 0.0);
  SetConstProperty(table, "BACKSCATTERCONSTANT", 0.0);
  surface->SetMaterialPropertiesTable(table);
  return surface;
}

G4OpticalSurface* MakeGlossyBackSurface() {
  // Chroma's glossy_surface is 50% ideal specular and 50% Lambertian, with no
  // absorption.  In the unified model the probability not assigned to spike,
  // lobe, or backscatter is Lambertian.
  auto* surface = new G4OpticalSurface(
      "glossy_surface", unified, ground, dielectric_metal);
  surface->SetSigmaAlpha(0.0);
  auto* table = new G4MaterialPropertiesTable;
  SetConstProperty(table, "REFLECTIVITY", 1.0);
  SetConstProperty(table, "EFFICIENCY", 0.0);
  SetConstProperty(table, "SPECULARSPIKECONSTANT", 0.5);
  SetConstProperty(table, "SPECULARLOBECONSTANT", 0.0);
  SetConstProperty(table, "BACKSCATTERCONSTANT", 0.0);
  surface->SetMaterialPropertiesTable(table);
  return surface;
}

template <typename Container>
G4Polycone* MakePolycone(const G4String& name, const Container& profile) {
  std::vector<G4double> z;
  std::vector<G4double> inner;
  std::vector<G4double> outer;
  z.reserve(profile.size());
  inner.assign(profile.size(), 0.0);
  outer.reserve(profile.size());
  for (const auto& point : profile) {
    z.push_back(point.z * mm);
    outer.push_back(point.r * mm);
  }
  return new G4Polycone(name, 0.0, twopi, z.size(), z.data(), inner.data(),
                        outer.data());
}

std::pair<std::vector<ProfilePoint>, std::vector<ProfilePoint>> SplitInnerProfile() {
  std::vector<ProfilePoint> back;
  std::vector<ProfilePoint> front;
  const G4double split = kPhotocathodeThreshold / mm;
  for (std::size_t i = 0; i + 1 < kInnerProfile.size(); ++i) {
    const auto& a = kInnerProfile[i];
    const auto& b = kInnerProfile[i + 1];
    if (a.z <= split) back.push_back(a);
    if (a.z >= split) front.push_back(a);
    if (a.z < split && b.z > split) {
      const G4double fraction = (split - a.z) / (b.z - a.z);
      const ProfilePoint crossing{
          a.r + fraction * (b.r - a.r), split};
      back.push_back(crossing);
      front.push_back(crossing);
    }
  }
  const auto& last = kInnerProfile.back();
  if (last.z <= split) back.push_back(last);
  if (last.z >= split) front.push_back(last);
  return {back, front};
}

struct ClippedWire {
  G4ThreeVector center;
  G4double halfLength;
};

bool ClipWireToActiveSquare(const G4ThreeVector& base,
                            const G4ThreeVector& axis,
                            ClippedWire& result) {
  G4double low = -std::numeric_limits<G4double>::infinity();
  G4double high = std::numeric_limits<G4double>::infinity();
  const auto clipAxis = [&](G4double coordinate, G4double direction) {
    if (std::abs(direction) < 1.0e-15) {
      return std::abs(coordinate) <= kActiveHalf;
    }
    G4double t0 = (-kActiveHalf - coordinate) / direction;
    G4double t1 = (kActiveHalf - coordinate) / direction;
    if (t0 > t1) std::swap(t0, t1);
    low = std::max(low, t0);
    high = std::min(high, t1);
    return low < high;
  };
  if (!clipAxis(base.y(), axis.y()) || !clipAxis(base.z(), axis.z())) {
    return false;
  }
  result.center = base + 0.5 * (low + high) * axis;
  result.halfLength = 0.5 * (high - low);
  return result.halfLength > 0.0;
}

std::string IndexedName(const char* prefix, G4int index, G4int width = 5) {
  std::ostringstream stream;
  stream << prefix << std::setfill('0') << std::setw(width) << index;
  return stream.str();
}

}  // namespace

G4VPhysicalVolume* BuildReflect3WiresGeometry(GeometrySummary& summary) {
  auto* nist = G4NistManager::Instance();
  auto* argon = nist->FindOrBuildElement("Ar");
  auto* silicon = nist->FindOrBuildElement("Si");
  auto* oxygen = nist->FindOrBuildElement("O");
  auto* iron = nist->FindOrBuildElement("Fe");

  auto* vacuum = MakeMaterial(
      "vacuum", universe_mean_density, 1.0, 1.0e6 * mm, 1.0e6 * mm,
      {{argon, 1.0}});
  auto* lar = MakeMaterial(
      "liquid_argon", 1.396 * g / cm3, 1.3784, 1.0e10 * mm, 950.0 * mm,
      {{argon, 1.0}});
  auto* glass = MakeMaterial(
      "glass", 1.0 * g / cm3, 1.525, 1.5e3 * mm, 1.0e6 * mm,
      {{silicon, 0.4675}, {oxygen, 0.5325}});
  auto* steel = MakeMaterial(
      "steel", 8.05 * g / cm3, 1.07, 1.0e-6 * mm, 1.0e-6 * mm,
      {{iron, 1.0}});

  auto* black = MakeMetalSurface("reflect00", 0.0, 0.0);
  auto* polishedSteel = MakeMetalSurface("polished_steel", 0.8, 0.0);
  auto* photocathode = MakeMetalSurface(
      "perfect_pmt_photocathode", 0.0, 1.0);
  auto* glossyBack = MakeGlossyBackSurface();

  auto* worldSolid = new G4Box("world_solid", 4000.0 * mm, 4000.0 * mm,
                               4000.0 * mm);
  auto* worldLogical = new G4LogicalVolume(worldSolid, vacuum, "world_logical");
  auto* worldPhysical = new G4PVPlacement(
      nullptr, {}, worldLogical, "world", nullptr, false, 0, false);

  auto* cavitySolid = new G4Box("cavity_solid", kCavityLength / 2.0,
                                kCavityLength / 2.0, kCavityLength / 2.0);
  auto* cavityLogical = new G4LogicalVolume(
      cavitySolid, lar, "cavity_logical");
  auto* cavityPhysical = new G4PVPlacement(
      nullptr, {}, cavityLogical, "cavity", worldLogical, false, 0, false);
  // GDML cannot reference the setup/world placement from a border surface.
  // A skin on the cavity gives the same two-way reflect00 boundary; the
  // explicit active/cavity border below takes precedence at the inner child.
  new G4LogicalSkinSurface("cavity_reflect00", cavityLogical, black);

  auto* activeSolid = new G4Box("active_enclosure_solid", kActiveXHalf,
                                kActiveHalf, kActiveHalf);
  auto* activeLogical = new G4LogicalVolume(
      activeSolid, lar, "active_enclosure_logical");
  auto* activePhysical = new G4PVPlacement(
      nullptr, {}, activeLogical, "active_enclosure", cavityLogical, false, 0,
      false);
  new G4LogicalBorderSurface("active_to_cavity_polished_steel", activePhysical,
                             cavityPhysical, polishedSteel);
  new G4LogicalBorderSurface("cavity_to_active_polished_steel", cavityPhysical,
                             activePhysical, polishedSteel);

  auto* cathodeSolid = new G4Box("cathode_solid", kCathodeHalfThickness,
                                 kActiveHalf, kActiveHalf);
  auto* cathodeLogical = new G4LogicalVolume(
      cathodeSolid, steel, "cathode_logical");
  new G4PVPlacement(nullptr, {}, cathodeLogical, "cathode", activeLogical,
                    false, 0, false);
  new G4LogicalSkinSurface("cathode_polished_steel", cathodeLogical,
                           polishedSteel);

  // Six periodic wire planes.  Within the optically reachable active box,
  // clipping each analytic infinite lattice line to the YZ square is exactly
  // equivalent to the specialized Chroma/Triton intersection bounds and also
  // respects Geant4's daughter-volume containment rule.
  const std::array<G4double, 3> angles{{halfpi, pi / 3.0, -pi / 3.0}};
  const std::array<G4double, 3> offsets{{0.0 * mm, -3.0 * mm, -6.0 * mm}};
  G4int wireCopy = 0;
  for (G4int side : {-1, 1}) {
    for (std::size_t plane = 0; plane < angles.size(); ++plane) {
      const G4double angle = angles[plane];
      const G4ThreeVector u(0.0, std::cos(angle), std::sin(angle));
      G4ThreeVector v;
      if (side > 0) {
        v = G4ThreeVector(0.0, -std::sin(angle), std::cos(angle));
      } else {
        v = G4ThreeVector(0.0, std::sin(angle), -std::cos(angle));
      }
      const G4double x = side > 0
          ? kActiveHalf + offsets[plane]
          : -kActiveHalf - offsets[plane];
      const G4int kmax = plane == 0 ? 720 : 983;
      const G4String solidName =
          "wire_solid_side" + std::to_string(side) + "_plane" +
          std::to_string(plane);
      auto* planeSolid = new G4MultiUnion(solidName);
      G4int planeWireCount = 0;
      for (G4int k = -kmax; k <= kmax; ++k) {
        const G4ThreeVector base(x, 0.0, 0.0);
        ClippedWire clipped;
        if (!ClipWireToActiveSquare(base + (k * kWirePitch) * v, u, clipped)) {
          continue;
        }
        auto* solid = new G4Tubs(
            solidName + "_" + std::to_string(k), 0.0, kWireRadius,
            clipped.halfLength, 0.0, twopi);
        auto* rotation = RotationFromLocalZ(u);
        planeSolid->AddNode(*solid, G4Transform3D(*rotation, clipped.center));
        delete rotation;
        ++wireCopy;
        ++planeWireCount;
      }
      planeSolid->Voxelize();
      auto* planeLogical = new G4LogicalVolume(
          planeSolid, steel, "wire_plane_logical_side" + std::to_string(side) +
                                 "_plane" + std::to_string(plane));
      new G4LogicalSkinSurface(
          "wire_plane_surface_side" + std::to_string(side) + "_plane" +
              std::to_string(plane),
          planeLogical, polishedSteel);
      new G4PVPlacement(
          nullptr, {}, planeLogical,
          "wire_plane_side" + std::to_string(side) + "_plane" +
              std::to_string(plane),
          activeLogical, false, static_cast<G4int>(summary.wirePlaneCount),
          false);
      if (planeWireCount == 0) {
        G4Exception("BuildReflect3WiresGeometry", "EmptyWirePlane",
                    FatalException, "wire plane contains no cylinders");
      }
      ++summary.wirePlaneCount;
    }
  }
  summary.wireCount = wireCopy;

  auto [backProfile, frontProfile] = SplitInnerProfile();
  auto* pmtOuterSolid = MakePolycone("pmt_outer_solid", kOuterProfile);
  auto* pmtBackSolid = MakePolycone("pmt_inner_back_solid", backProfile);
  auto* pmtFrontSolid = MakePolycone("pmt_inner_front_solid", frontProfile);
  auto* pmtOuterLogical = new G4LogicalVolume(
      pmtOuterSolid, glass, "pmt_outer_glass_logical");
  auto* pmtBackLogical = new G4LogicalVolume(
      pmtBackSolid, vacuum, "pmt_inner_back_logical");
  auto* pmtFrontLogical = new G4LogicalVolume(
      pmtFrontSolid, vacuum, "pmt_inner_front_logical");
  auto* pmtBackPhysical = new G4PVPlacement(
      nullptr, {}, pmtBackLogical, "pmt_inner_back", pmtOuterLogical, false, 0,
      false);
  auto* pmtFrontPhysical = new G4PVPlacement(
      nullptr, {}, pmtFrontLogical, "pmt_inner_front", pmtOuterLogical, false,
      0, false);

  G4int gridY = static_cast<G4int>(kActiveLength / kPmtSpacing) + 1;
  G4int gridZ = static_cast<G4int>(kActiveLength / kPmtSpacing) + 1;
  G4double bufferY =
      kActiveLength - (gridY - 1) * kPmtSpacing - 4.0 * kPmtRadius;
  G4double bufferZ =
      kActiveLength - (gridZ - 1) * kPmtSpacing - 4.0 * kPmtRadius;
  while (bufferY < kPmtSpacing / 2.0) {
    --gridY;
    bufferY = kActiveLength - (gridY - 1) * kPmtSpacing;
  }
  while (bufferZ < 0.0) {
    --gridZ;
    bufferZ = kActiveLength - (gridZ - 1) * kPmtSpacing;
  }

  std::vector<G4ThreeVector> wallPositions;
  wallPositions.reserve(gridY * gridZ);
  for (G4int iy = 0; iy < gridY; ++iy) {
    for (G4int iz = 0; iz < gridZ; ++iz) {
      G4double y = iy * kPmtSpacing - kActiveHalf +
                   (bufferY / 2.0 - kPmtSpacing / 4.0) + kPmtRadius;
      G4double z = iz * kPmtSpacing - kActiveHalf + bufferZ / 2.0 +
                   2.0 * kPmtRadius;
      if (iz % 2 == 1) y += kPmtSpacing / 2.0 - kPmtRadius;
      wallPositions.emplace_back(0.0, y, z);
    }
  }
  G4ThreeVector mean;
  for (const auto& position : wallPositions) mean += position;
  mean /= wallPositions.size();
  for (auto& position : wallPositions) position -= mean;

  G4int channel = 0;
  for (G4int side : {-1, 1}) {
    const G4ThreeVector inward(side < 0 ? 1.0 : -1.0, 0.0, 0.0);
    for (const auto& yz : wallPositions) {
      auto* rotation = RotationFromLocalZ(inward);
      const G4ThreeVector position(
          side * (kActiveHalf + kPmtGap), yz.y(), yz.z());
      auto* outerPhysical = new G4PVPlacement(
          rotation, position, pmtOuterLogical,
          IndexedName("pmt_outer_", channel, 3), activeLogical, false, channel,
          false);
      new G4LogicalBorderSurface(
          IndexedName("pmt_front_glass_to_vacuum_", channel, 3), outerPhysical,
          pmtFrontPhysical, photocathode);
      new G4LogicalBorderSurface(
          IndexedName("pmt_front_vacuum_to_glass_", channel, 3),
          pmtFrontPhysical, outerPhysical, photocathode);
      new G4LogicalBorderSurface(
          IndexedName("pmt_back_glass_to_vacuum_", channel, 3), outerPhysical,
          pmtBackPhysical, glossyBack);
      new G4LogicalBorderSurface(
          IndexedName("pmt_back_vacuum_to_glass_", channel, 3), pmtBackPhysical,
          outerPhysical, glossyBack);
      ++channel;
    }
  }
  summary.pmtCount = channel;
  summary.photocathodeChannels = channel;
  return worldPhysical;
}

}  // namespace chroma_lar_geant4
