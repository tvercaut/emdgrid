# Fetch a prebuilt OR-Tools release via tools/get_ortools.py when find_package
# can't locate an OR-Tools install, and point CMAKE_PREFIX_PATH at the result.
#
# This mirrors the spdlog/LEMON FetchContent fallbacks in CMakeLists.txt, but
# OR-Tools is consumed as a platform-specific prebuilt archive rather than
# built from source, so the platform/asset-selection logic lives in
# tools/get_ortools.py instead of CMake. Included only when find_package(ortools)
# has already failed once.

set(EMDGRID_ORTOOLS_FETCH_DIR "${CMAKE_BINARY_DIR}/_deps/ortools" CACHE PATH
    "Directory where a prebuilt OR-Tools release is downloaded and extracted")

function(_emdgrid_find_extracted_ortools_dir out_var)
  file(GLOB _dirs LIST_DIRECTORIES true "${EMDGRID_ORTOOLS_FETCH_DIR}/or-tools*")
  set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()

_emdgrid_find_extracted_ortools_dir(_emdgrid_ortools_dirs)

if(NOT _emdgrid_ortools_dirs)
  message(STATUS "Could NOT find ortools: fetching a prebuilt release via tools/get_ortools.py")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/get_ortools.py"
            --output-path "${EMDGRID_ORTOOLS_FETCH_DIR}"
            --show-progress off
    COMMAND_ERROR_IS_FATAL ANY)
  _emdgrid_find_extracted_ortools_dir(_emdgrid_ortools_dirs)
endif()

list(LENGTH _emdgrid_ortools_dirs _emdgrid_ortools_dirs_count)
if(NOT _emdgrid_ortools_dirs_count EQUAL 1)
  message(FATAL_ERROR
    "Expected exactly one extracted OR-Tools directory under "
    "${EMDGRID_ORTOOLS_FETCH_DIR}, found: ${_emdgrid_ortools_dirs}")
endif()

list(GET _emdgrid_ortools_dirs 0 _emdgrid_ortools_dir)
list(APPEND CMAKE_PREFIX_PATH "${_emdgrid_ortools_dir}")

# Record where the fetched OR-Tools shared libraries live so bindings/python/CMakeLists.txt
# can bundle them into the wheel (see the comment there for why that is necessary): this
# directory is under CMAKE_BINARY_DIR, which for a `pip install` is an ephemeral
# scikit-build-core temp dir that is deleted once the build finishes, so the extension
# module cannot rely on it still existing at import time and must ship its own copies.
set(EMDGRID_ORTOOLS_FETCHED_LIB_DIR "${_emdgrid_ortools_dir}/lib" CACHE PATH
    "Directory holding the fetched OR-Tools shared libraries, for bundling into the Python wheel" FORCE)
