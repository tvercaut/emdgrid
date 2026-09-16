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

find_package(spdlog QUIET)
if(spdlog_FOUND OR TARGET spdlog)
  message(STATUS "Found spdlog: ${spdlog_DIR}")
else()
  message(STATUS "Could NOT find spdlog: Building it")
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
    SYSTEM
  )
  FetchContent_MakeAvailable(spdlog)
endif()
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
if(LEMON_FOUND OR TARGET lemon)
  message(STATUS "Found LEMON: ${LEMON_DIR}")
else()
  message(STATUS "Could NOT find LEMON: Building it")
  block()
    set(LEMON_ENABLE_COIN NO CACHE STRING "" FORCE)
    set(LEMON_ENABLE_GLPK NO CACHE STRING "" FORCE)
    set(LEMON_ENABLE_ILOG NO CACHE STRING "" FORCE)
    set(LEMON_ENABLE_SOPLEX NO CACHE STRING "" FORCE)
    FetchContent_Declare(
        lemon
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

option(EMDGRID_USE_OPENMP "Use OpenMP parallelization in the potlemon network simplex solver" ON)
if(EMDGRID_USE_OPENMP)
  # Apple Clang does not locate Homebrew's libomp automatically.
  # Provide hints before find_package so CMake's FindOpenMP picks them up.
  if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
      execute_process(
        COMMAND ${BREW_EXECUTABLE} --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE BREW_LIBOMP_RESULT
      )
      if(BREW_LIBOMP_RESULT EQUAL 0 AND HOMEBREW_LIBOMP_PREFIX
          AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}/include/omp.h")
        # Use -isystem so omp.h is treated as a system header (no warnings).
        set(OpenMP_CXX_FLAGS
            "-Xpreprocessor -fopenmp -isystem ${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
      endif()
    endif()
  endif()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(emdgrid PUBLIC OpenMP::OpenMP_CXX)
    # Define POTLEMON_OPENMP so the header includes <omp.h> only when the
    # build system has set up the correct include paths (i.e. not just because
    # the compiler happens to define _OPENMP via an unrelated -fopenmp flag).
    target_compile_definitions(emdgrid PUBLIC POTLEMON_OPENMP=1)
    # On Apple/Homebrew, expose the libomp include path explicitly so all
    # consumers can find omp.h. The -isystem flag inside OpenMP_CXX_FLAGS is
    # in INTERFACE_COMPILE_OPTIONS and may not propagate reliably; an explicit
    # INTERFACE_INCLUDE_DIRECTORIES entry always propagates.
    if(APPLE AND HOMEBREW_LIBOMP_PREFIX)
      target_include_directories(emdgrid SYSTEM PUBLIC
        "${HOMEBREW_LIBOMP_PREFIX}/include")
    endif()
    # GCC puts omp.h in a private compiler include directory that clang-tidy
    # (which uses a Clang frontend) does not search. Expose it explicitly so
    # clang-tidy can parse headers that include <omp.h>.
    # Use -print-libgcc-file-name (always an absolute path) to locate the
    # GCC lib dir, then check for include/omp.h next to it.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-libgcc-file-name
        OUTPUT_VARIABLE _libgcc_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(_libgcc_path)
        get_filename_component(_gcc_lib_dir "${_libgcc_path}" DIRECTORY)
        if(EXISTS "${_gcc_lib_dir}/include/omp.h")
          target_include_directories(emdgrid SYSTEM PUBLIC
            "${_gcc_lib_dir}/include")
        endif()
      endif()
    endif()
    message(STATUS "OpenMP found and enabled (disable with -DEMDGRID_USE_OPENMP=OFF).")
  else()
    message(STATUS "OpenMP requested but not found on this system.")
  endif()
else()
  message(STATUS "OpenMP disabled (EMDGRID_USE_OPENMP=OFF).")
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
