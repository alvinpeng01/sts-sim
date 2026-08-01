cmake_minimum_required(VERSION 3.19)
project(
        sts_lightspeed
        VERSION 0.1
        DESCRIPTION "High Performance, rng accurate implementation of Slay the Spire with tools for tree search and ml."
        LANGUAGES CXX)

file (GLOB_RECURSE sts_lightspeed_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

set(CMAKE_CXX_STANDARD 17)
# -DNDEBUG: assert() is a no-op in release. include/sts_common.h defines sts_asserts, which guards
# assert() calls sitting inside the hottest paths (ActionQueue::pushBack, monster move dispatch,
# BattleContext::endTurn) -- without NDEBUG every one of those is a live branch + call during search.
# -march=native: binary is host-specific, which is fine for a local research build (silverbot-
# reference exposes the same thing as an opt-in STS_MARCH_NATIVE option).
set(STS_PGO "" CACHE STRING "empty | generate | use")
set(CMAKE_CXX_FLAGS  "-Wno-shift-count-overflow -O3 -DNDEBUG -flto")
if(STS_PGO STREQUAL "generate")
    string(APPEND CMAKE_CXX_FLAGS " -fprofile-generate -fprofile-dir=${CMAKE_SOURCE_DIR}/pgo")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fprofile-generate")
elseif(STS_PGO STREQUAL "use")
    string(APPEND CMAKE_CXX_FLAGS " -fprofile-use -fprofile-correction -Wno-missing-profile -Wno-error=coverage-mismatch -Wno-coverage-mismatch -fprofile-dir=${CMAKE_SOURCE_DIR}/pgo")
endif()

add_subdirectory(json)
add_subdirectory(pybind11)

pybind11_add_module(slaythespire bindings/slaythespire.cpp bindings/bindings-util.cpp ${sts_lightspeed_SOURCES})
target_include_directories(slaythespire PUBLIC include)
target_include_directories(slaythespire PUBLIC json/single_include)
target_include_directories(slaythespire PUBLIC bindings)


add_executable(main apps/main.cpp ${sts_lightspeed_SOURCES})
target_link_directories(main PRIVATE json::nlohmann_json)
target_include_directories(main PUBLIC include)
target_include_directories(main PUBLIC json/include)

add_executable(test apps/test.cpp ${sts_lightspeed_SOURCES})
target_include_directories(test PUBLIC include)
target_link_directories(test PUBLIC json::nlohmann_json)
target_include_directories(test PUBLIC json/include)

add_executable(small-test apps/small-test.cpp bindings/bindings-util.cpp ${sts_lightspeed_SOURCES})
target_link_directories(small-test PRIVATE json::nlohmann_json)
target_include_directories(small-test PUBLIC include)
target_include_directories(small-test PUBLIC json/include)
target_include_directories(small-test PUBLIC bindings)
