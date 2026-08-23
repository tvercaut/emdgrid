cmake_minimum_required(VERSION 3.20)

project(emdgrid LANGUAGES CXX)

include(cmake/GitVersion.cmake)
configure_file(
  include/emdgrid/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/emdgrid/version.hpp
  @ONLY)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for tools like clang-tidy")

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
