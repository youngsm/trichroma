#include "G4Event.hh"
#include "G4GDMLParser.hh"
#include "G4MTRunManager.hh"
#include "G4OpAbsorption.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4OpRayleigh.hh"
#include "G4OpticalPhoton.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UserRunAction.hh"
#include "G4Version.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPhysicsList.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4ios.hh"
#include "Randomize.hh"
#include "globals.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr G4int kChannels = 162;
constexpr G4double kWavelength = 450.0 * nm;

struct Settings {
  std::string gdml;
  std::string json;
  G4long photons = 100000;
  G4int threads = 1;
  G4int photonsPerEvent = 256;
  G4int warmupEvents = 0;
  G4int maxSteps = 1000;
  std::uint64_t seed = 1;
  G4ThreeVector position{-1000.0 * mm, 0.0, 0.0};
};

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

G4double Uniform01(std::uint64_t seed, std::uint64_t photon,
                   std::uint64_t draw) {
  const auto bits = SplitMix64(seed ^ (photon * 0xd2b74407b1ce6e93ULL) ^
                               (draw * 0xca5a826395121157ULL));
  return static_cast<G4double>(bits >> 11U) * 0x1.0p-53;
}

G4ThreeVector Isotropic(std::uint64_t seed, std::uint64_t photon,
                        std::uint64_t firstDraw) {
  const G4double cosine = 2.0 * Uniform01(seed, photon, firstDraw) - 1.0;
  const G4double phi = CLHEP::twopi * Uniform01(seed, photon, firstDraw + 1U);
  const G4double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
  return {sine * std::cos(phi), sine * std::sin(phi), cosine};
}

class GDMLDetector final : public G4VUserDetectorConstruction {
 public:
  explicit GDMLDetector(const std::string& path) {
    parser_.Read(path, false);
    world_ = parser_.GetWorldVolume();
    if (world_ == nullptr) throw std::runtime_error("GDML contains no world");
  }
  G4VPhysicalVolume* Construct() override { return world_; }

 private:
  G4GDMLParser parser_;
  G4VPhysicalVolume* world_ = nullptr;
};

class OpticalPhysicsList final : public G4VUserPhysicsList {
 public:
  OpticalPhysicsList() {
    SetVerboseLevel(0);
  }
  void ConstructParticle() override { G4OpticalPhoton::Definition(); }
  void ConstructProcess() override {
    AddTransportation();
    auto* manager = G4OpticalPhoton::Definition()->GetProcessManager();
    manager->AddDiscreteProcess(new G4OpAbsorption);
    manager->AddDiscreteProcess(new G4OpRayleigh);
    manager->AddDiscreteProcess(new G4OpBoundaryProcess);
  }
  void SetCuts() override { SetCutsWithDefault(); }
};

class BenchmarkRun final : public G4Run {
 public:
  G4long generated = 0;
  G4long detected = 0;
  G4long bulkAbsorbed = 0;
  G4long surfaceAbsorbed = 0;
  G4long escaped = 0;
  G4long maxStepKilled = 0;
  G4long noRindexKilled = 0;
  G4long otherKilled = 0;
  G4long steps = 0;
  std::array<G4long, kChannels> channelHits{};

  void Merge(const G4Run* other) override {
    const auto* rhs = static_cast<const BenchmarkRun*>(other);
    generated += rhs->generated;
    detected += rhs->detected;
    bulkAbsorbed += rhs->bulkAbsorbed;
    surfaceAbsorbed += rhs->surfaceAbsorbed;
    escaped += rhs->escaped;
    maxStepKilled += rhs->maxStepKilled;
    noRindexKilled += rhs->noRindexKilled;
    otherKilled += rhs->otherKilled;
    steps += rhs->steps;
    for (G4int i = 0; i < kChannels; ++i) channelHits[i] += rhs->channelHits[i];
    G4Run::Merge(other);
  }

  G4long Terminal() const {
    return detected + bulkAbsorbed + surfaceAbsorbed + escaped +
           maxStepKilled + noRindexKilled + otherKilled;
  }
};

class RunAction final : public G4UserRunAction {
 public:
  G4Run* GenerateRun() override { return new BenchmarkRun; }
};

BenchmarkRun* CurrentRun() {
  return static_cast<BenchmarkRun*>(
      G4RunManager::GetRunManager()->GetNonConstCurrentRun());
}

class PrimaryGenerator final : public G4VUserPrimaryGeneratorAction {
 public:
  explicit PrimaryGenerator(const Settings& settings)
      : settings_(settings), gun_(1) {
    gun_.SetParticleDefinition(G4OpticalPhoton::Definition());
    gun_.SetParticleEnergy(CLHEP::twopi * CLHEP::hbarc / kWavelength);
    gun_.SetParticlePosition(settings_.position);
  }

  void GeneratePrimaries(G4Event* event) override {
    const G4long first =
        static_cast<G4long>(event->GetEventID()) * settings_.photonsPerEvent;
    const G4long count = std::min<G4long>(
        settings_.photonsPerEvent, settings_.photons - first);
    auto* run = CurrentRun();
    for (G4long offset = 0; offset < count; ++offset) {
      const auto photon = static_cast<std::uint64_t>(first + offset);
      const auto direction = Isotropic(settings_.seed, photon, 0);
      auto helper = Isotropic(settings_.seed, photon, 2);
      auto polarization = direction.cross(helper);
      if (polarization.mag2() < 1.0e-24) {
        helper = std::abs(direction.z()) < 0.9
                     ? G4ThreeVector(0.0, 0.0, 1.0)
                     : G4ThreeVector(0.0, 1.0, 0.0);
        polarization = direction.cross(helper);
      }
      gun_.SetParticleMomentumDirection(direction);
      gun_.SetParticlePolarization(polarization.unit());
      gun_.GeneratePrimaryVertex(event);
      ++run->generated;
    }
  }

 private:
  Settings settings_;
  G4ParticleGun gun_;
};

G4int PmtChannel(const G4Step* step) {
  const auto touchable = step->GetPreStepPoint()->GetTouchableHandle();
  for (G4int depth = 0; depth <= touchable->GetHistoryDepth(); ++depth) {
    const auto* volume = touchable->GetVolume(depth);
    if (volume != nullptr && volume->GetName().find("pmt_outer_") == 0) {
      return volume->GetCopyNo();
    }
  }
  const auto postTouchable = step->GetPostStepPoint()->GetTouchableHandle();
  for (G4int depth = 0; depth <= postTouchable->GetHistoryDepth(); ++depth) {
    const auto* volume = postTouchable->GetVolume(depth);
    if (volume != nullptr && volume->GetName().find("pmt_outer_") == 0) {
      return volume->GetCopyNo();
    }
  }
  return -1;
}

class SteppingAction final : public G4UserSteppingAction {
 public:
  explicit SteppingAction(G4int maxSteps) : maxSteps_(maxSteps) {}

  void UserSteppingAction(const G4Step* step) override {
    auto* run = CurrentRun();
    ++run->steps;
    auto* track = step->GetTrack();
    const auto* post = step->GetPostStepPoint();
    const auto* process = post->GetProcessDefinedStep();

    // G4Transportation normally defines a geometry-limited step even though
    // G4OpBoundaryProcess performs the forced optical interaction.  Querying
    // only GetProcessDefinedStep() therefore misclassified every terminal
    // surface interaction as an unexplained kill.
    auto* boundary = BoundaryProcess();
    if (boundary != nullptr) {
      if (boundary->GetStatus() == Detection) {
        ++run->detected;
        const G4int channel = PmtChannel(step);
        if (channel >= 0 && channel < kChannels) ++run->channelHits[channel];
        return;
      }
      if (boundary->GetStatus() == Absorption) {
        ++run->surfaceAbsorbed;
        return;
      }
      if (boundary->GetStatus() == NoRINDEX) {
        ++run->noRindexKilled;
        return;
      }
    }
    if (process != nullptr && process->GetProcessName() == "OpAbsorption") {
      ++run->bulkAbsorbed;
      return;
    }
    if (post->GetStepStatus() == fWorldBoundary) {
      ++run->escaped;
      return;
    }
    if (track->GetCurrentStepNumber() >= maxSteps_ &&
        track->GetTrackStatus() == fAlive) {
      track->SetTrackStatus(fStopAndKill);
      ++run->maxStepKilled;
      return;
    }
    if (track->GetTrackStatus() == fStopAndKill) ++run->otherKilled;
  }

 private:
  G4OpBoundaryProcess* BoundaryProcess() {
    if (boundary_ != nullptr) return boundary_;
    auto* manager = G4OpticalPhoton::Definition()->GetProcessManager();
    if (manager == nullptr) return nullptr;
    auto* processes = manager->GetProcessList();
    const G4int count = manager->GetProcessListLength();
    for (G4int index = 0; index < count; ++index) {
      boundary_ = dynamic_cast<G4OpBoundaryProcess*>((*processes)[index]);
      if (boundary_ != nullptr) return boundary_;
    }
    return nullptr;
  }

  G4int maxSteps_;
  G4OpBoundaryProcess* boundary_ = nullptr;
};

class Actions final : public G4VUserActionInitialization {
 public:
  explicit Actions(Settings settings) : settings_(std::move(settings)) {}
  void BuildForMaster() const override { SetUserAction(new RunAction); }
  void Build() const override {
    SetUserAction(new PrimaryGenerator(settings_));
    SetUserAction(new RunAction);
    SetUserAction(new SteppingAction(settings_.maxSteps));
  }

 private:
  Settings settings_;
};

G4long ParseLong(const char* value, const char* option) {
  char* end = nullptr;
  const auto parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + option + ": " + value);
  }
  return parsed;
}

Settings ParseArguments(int argc, char** argv) {
  Settings settings;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    const auto next = [&]() -> const char* {
      if (++i >= argc) throw std::runtime_error("missing value for " + option);
      return argv[i];
    };
    if (option == "--gdml") settings.gdml = next();
    else if (option == "--json") settings.json = next();
    else if (option == "--photons") settings.photons = ParseLong(next(), "--photons");
    else if (option == "--threads") settings.threads = ParseLong(next(), "--threads");
    else if (option == "--photons-per-event") settings.photonsPerEvent = ParseLong(next(), "--photons-per-event");
    else if (option == "--warmup-events") settings.warmupEvents = ParseLong(next(), "--warmup-events");
    else if (option == "--max-steps") settings.maxSteps = ParseLong(next(), "--max-steps");
    else if (option == "--seed") settings.seed = static_cast<std::uint64_t>(ParseLong(next(), "--seed"));
    else if (option == "--position") {
      const G4double x = std::stod(next());
      const G4double y = std::stod(next());
      const G4double z = std::stod(next());
      settings.position = {x * mm, y * mm, z * mm};
    } else if (option == "--help") {
      G4cout << "benchmark_reflect3wires --gdml FILE [--json FILE] "
                "[--photons N] [--threads N] [--photons-per-event N] "
                "[--warmup-events N] [--max-steps N] [--seed N] "
                "[--position X Y Z]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (settings.gdml.empty()) throw std::runtime_error("--gdml is required");
  if (settings.photons <= 0 || settings.threads <= 0 ||
      settings.photonsPerEvent <= 0 || settings.warmupEvents < 0 ||
      settings.maxSteps <= 0) {
    throw std::runtime_error("counts, threads, photons/event, and max steps must be positive");
  }
  return settings;
}

void WriteResult(std::ostream& output, const Settings& settings,
                 const BenchmarkRun& run, G4double setupElapsed,
                 G4double warmupElapsed, G4double elapsed,
                 G4int warmupEvents) {
  output << std::setprecision(12)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"engine\": \"Geant4\",\n"
         << "  \"geant4_version\": \"" << G4Version << "\",\n"
         << "  \"gdml\": \"" << settings.gdml << "\",\n"
         << "  \"threads\": " << settings.threads << ",\n"
         << "  \"photons\": " << settings.photons << ",\n"
         << "  \"photons_per_event\": " << settings.photonsPerEvent << ",\n"
         << "  \"warmup_events\": " << warmupEvents << ",\n"
         << "  \"events\": "
         << (settings.photons + settings.photonsPerEvent - 1) /
                settings.photonsPerEvent
         << ",\n"
         << "  \"seed\": " << settings.seed << ",\n"
         << "  \"wavelength_nm\": 450.0,\n"
         << "  \"source_position_mm\": [" << settings.position.x() / mm << ", "
         << settings.position.y() / mm << ", " << settings.position.z() / mm
         << "],\n"
         << "  \"setup_seconds\": " << setupElapsed << ",\n"
         << "  \"warmup_seconds\": " << warmupElapsed << ",\n"
         << "  \"elapsed_seconds\": " << elapsed << ",\n"
         << "  \"photons_per_second\": " << settings.photons / elapsed << ",\n"
         << "  \"generated\": " << run.generated << ",\n"
         << "  \"terminal\": " << run.Terminal() << ",\n"
         << "  \"detected\": " << run.detected << ",\n"
         << "  \"bulk_absorbed\": " << run.bulkAbsorbed << ",\n"
         << "  \"surface_absorbed\": " << run.surfaceAbsorbed << ",\n"
         << "  \"escaped\": " << run.escaped << ",\n"
         << "  \"max_step_killed\": " << run.maxStepKilled << ",\n"
         << "  \"no_rindex_killed\": " << run.noRindexKilled << ",\n"
         << "  \"other_killed\": " << run.otherKilled << ",\n"
         << "  \"steps\": " << run.steps << ",\n"
         << "  \"mean_steps_per_photon\": "
         << static_cast<G4double>(run.steps) / run.generated << ",\n"
         << "  \"channel_hits\": [";
  for (G4int i = 0; i < kChannels; ++i) {
    if (i != 0) output << ", ";
    output << run.channelHits[i];
  }
  output << "]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Settings settings = ParseArguments(argc, argv);
    const auto setupStarted = std::chrono::steady_clock::now();
    G4Random::setTheSeed(static_cast<long>(settings.seed));
    auto runManager = std::make_unique<G4MTRunManager>();
    runManager->SetNumberOfThreads(settings.threads);
    runManager->SetUserInitialization(new GDMLDetector(settings.gdml));
    runManager->SetUserInitialization(new OpticalPhysicsList);
    runManager->SetUserInitialization(new Actions(settings));
    runManager->Initialize();
    const G4double setupElapsed = std::chrono::duration<G4double>(
        std::chrono::steady_clock::now() - setupStarted).count();

    G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/event/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/tracking/verbose 0");
    const G4int events = static_cast<G4int>(
        (settings.photons + settings.photonsPerEvent - 1) /
        settings.photonsPerEvent);
    const G4int warmupEvents = settings.warmupEvents > 0
        ? std::min(settings.warmupEvents, events)
        : std::min(std::max(1, settings.threads), events);
    const auto warmupStarted = std::chrono::steady_clock::now();
    runManager->BeamOn(warmupEvents);
    const G4double warmupElapsed = std::chrono::duration<G4double>(
        std::chrono::steady_clock::now() - warmupStarted).count();
    const auto started = std::chrono::steady_clock::now();
    runManager->BeamOn(events);
    const G4double elapsed = std::chrono::duration<G4double>(
        std::chrono::steady_clock::now() - started).count();

    const auto* run = dynamic_cast<const BenchmarkRun*>(runManager->GetCurrentRun());
    if (run == nullptr) throw std::runtime_error("benchmark run result unavailable");
    WriteResult(std::cout, settings, *run, setupElapsed, warmupElapsed, elapsed,
                warmupEvents);
    if (!settings.json.empty()) {
      std::ofstream output(settings.json);
      WriteResult(output, settings, *run, setupElapsed, warmupElapsed, elapsed,
                  warmupEvents);
      if (!output) throw std::runtime_error("failed to write " + settings.json);
    }
    if (run->generated != settings.photons || run->Terminal() != settings.photons) {
      G4cerr << "terminal-accounting mismatch: generated=" << run->generated
             << " terminal=" << run->Terminal() << " expected="
             << settings.photons << G4endl;
      return 3;
    }
    return 0;
  } catch (const std::exception& error) {
    G4cerr << "error: " << error.what() << G4endl;
    return 2;
  }
}
