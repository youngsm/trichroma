cmake_minimum_required(VERSION 3.20)
project(chroma_lar_geant4_benchmark LANGUAGES CXX)

find_package(Geant4 11.4 REQUIRED gdml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(reflect3wires_geometry STATIC
  src/Reflect3WiresGeometry.cc
)
target_include_directories(reflect3wires_geometry PUBLIC src)
target_link_libraries(reflect3wires_geometry PUBLIC ${Geant4_LIBRARIES})
target_compile_options(reflect3wires_geometry PRIVATE -Wall -Wextra -Wpedantic)

add_executable(generate_reflect3wires_gdml src/GenerateGDML.cc)
target_link_libraries(generate_reflect3wires_gdml PRIVATE
  reflect3wires_geometry ${Geant4_LIBRARIES}
)

add_executable(benchmark_reflect3wires src/Benchmark.cc)
target_link_libraries(benchmark_reflect3wires PRIVATE ${Geant4_LIBRARIES})
target_compile_options(benchmark_reflect3wires PRIVATE -Wall -Wextra -Wpedantic)

