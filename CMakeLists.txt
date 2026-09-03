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

find_package(ortools)

if(NOT TARGET ortools::ortools)
  block()
    set(FETCHCONTENT_QUIET ON)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    FetchContent_Declare(
        ortools
        QUIET
        SYSTEM
        GIT_REPOSITORY https://github.com/google/or-tools.git
        GIT_TAG "v9.15"
        GIT_SHALLOW TRUE
    )
    set(BUILD_DEPS ON CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_FLATZINC OFF CACHE BOOL "" FORCE)
    set(BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_CXX_SAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_CXX_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_MATH_OPT OFF CACHE BOOL "" FORCE)
    set(BUILD_LP_PARSER OFF CACHE BOOL "" FORCE)

    set(USE_BOP OFF CACHE BOOL "" FORCE)
    set(USE_COINOR OFF CACHE BOOL "" FORCE)
    set(USE_GLOP OFF CACHE BOOL "" FORCE)
    set(USE_GLPK OFF CACHE BOOL "" FORCE)
    set(USE_GUROBI OFF CACHE BOOL "" FORCE)
    set(USE_HIGHS OFF CACHE BOOL "" FORCE)
    set(USE_PDLP OFF CACHE BOOL "" FORCE)
    set(USE_SCIP OFF CACHE BOOL "" FORCE)
    set(USE_XPRESS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(ortools)
  endblock()
  if(TARGET ortools AND NOT TARGET ortools::ortools)
    add_library(ortools::ortools ALIAS ortools)
  endif()
endif()

if(TARGET ortools)
  set_target_properties(ortools PROPERTIES SYSTEM TRUE CXX_CLANG_TIDY "")
elseif(TARGET ortools::ortools)
  get_target_property(is_alias ortools::ortools ALIASED_TARGET)
  if(NOT is_alias)
    set_target_properties(ortools::ortools PROPERTIES SYSTEM TRUE CXX_CLANG_TIDY "")
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

target_link_libraries(emdgrid PUBLIC spdlog::spdlog ortools::ortools)

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
