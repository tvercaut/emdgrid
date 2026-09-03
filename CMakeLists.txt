cmake_minimum_required(VERSION 3.25)

project(emdgrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/GitVersion.cmake)
configure_file(
  include/emdgrid/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/emdgrid/version.hpp
  @ONLY)

find_package(ASan)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for tools like clang-tidy")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
    spdlog
    QUIET
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
    SYSTEM
)
FetchContent_MakeAvailable(spdlog)
if(TARGET spdlog)
  set_target_properties(spdlog PROPERTIES CXX_CLANG_TIDY "")
endif()

find_package(ortools REQUIRED)

if(TARGET ortools AND NOT TARGET ortools::ortools)
  add_library(ortools::ortools ALIAS ortools)
endif()

if(TARGET ortools)
  set_target_properties(ortools PROPERTIES SYSTEM TRUE CXX_CLANG_TIDY "")
elseif(TARGET ortools::ortools)
  get_target_property(is_alias ortools::ortools ALIASED_TARGET)
  if(NOT is_alias)
    set_target_properties(ortools::ortools PROPERTIES SYSTEM TRUE CXX_CLANG_TIDY "")
  endif()
endif()

find_package(LEMON QUIET)

if(NOT LEMON_FOUND AND NOT TARGET lemon)
  block()
    set(LEMON_ENABLE_COIN NO CACHE STRING "" FORCE)
    set(LEMON_ENABLE_GLPK NO CACHE STRING "" FORCE)
    set(LEMON_ENABLE_ILOG NO CACHE STRING "" FORCE)
    set(LEMON_ENABLE_SOPLEX NO CACHE STRING "" FORCE)
    FetchContent_Declare(
        lemon
        QUIET
        GIT_REPOSITORY https://github.com/The-OpenROAD-Project/lemon-graph
        GIT_TAG e9e3bc6dc0ffeda17eb2d2b4b296a68f03aec00d
        SYSTEM
    )
    FetchContent_MakeAvailable(lemon)
  endblock()
endif()

if(TARGET lemon)
  target_include_directories(lemon INTERFACE
    $<BUILD_INTERFACE:${LEMON_SOURCE_DIR}>
    $<BUILD_INTERFACE:${LEMON_BINARY_DIR}>
  )
  set_target_properties(lemon PROPERTIES SYSTEM TRUE CXX_CLANG_TIDY "")
  if(NOT TARGET lemon::lemon)
    add_library(lemon::lemon ALIAS lemon)
  endif()
elseif(TARGET lemon::lemon)
  get_target_property(is_alias lemon::lemon ALIASED_TARGET)
  if(NOT is_alias)
    set_target_properties(lemon::lemon PROPERTIES SYSTEM TRUE CXX_CLANG_TIDY "")
  endif()
endif()

find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
  set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
  message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}.")
else()
  message(STATUS "clang-tidy not found. Skipping corresponding checks.")
endif()

add_library(emdgrid src/emdgrid.cpp)
target_include_directories(
  emdgrid
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>)
target_compile_features(emdgrid PUBLIC cxx_std_23)

target_link_libraries(emdgrid PUBLIC spdlog::spdlog ortools::ortools lemon::lemon)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(emdgrid PUBLIC OpenMP::OpenMP_CXX)
endif()

option(EMDGRID_BUILD_EXAMPLES "Build example targets" ON)
option(EMDGRID_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

include(CTest)
if(EMDGRID_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(EMDGRID_BUILD_PYTHON_BINDINGS)
  add_subdirectory(bindings/python)
endif()
