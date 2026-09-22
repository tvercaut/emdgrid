find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.1.0
  GIT_SHALLOW TRUE
  SYSTEM
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(pyemdgrid pyemdgrid.cpp)
install(TARGETS pyemdgrid DESTINATION .)
target_link_libraries(pyemdgrid PRIVATE emdgrid)

if(EMDGRID_ORTOOLS_FETCHED_LIB_DIR)
  # OR-Tools was fetched as a prebuilt release by cmake/GetOrTools.cmake into a
  # directory under CMAKE_BINARY_DIR. For a normal `cmake --build` that
  # directory sticks around, and CMake's automatic build-tree RPATH lets
  # pyemdgrid find libortools.*, absl, protobuf, etc. there. But `pip install`
  # (scikit-build-core) builds in an ephemeral temp dir that is removed right
  # after the wheel is produced, and CMake does not propagate that build RPATH
  # to the installed/packaged module. Without this, the installed wheel fails
  # to import with e.g. "Library not loaded: @rpath/libortools.9.dylib". Ship
  # copies of the shared libraries next to the module instead, and point RPATH
  # at the module's own install directory so the wheel is self-contained.
  file(GLOB EMDGRID_ORTOOLS_RUNTIME_LIBS
       "${EMDGRID_ORTOOLS_FETCHED_LIB_DIR}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
  install(FILES ${EMDGRID_ORTOOLS_RUNTIME_LIBS} DESTINATION .)
  if(APPLE)
    set_target_properties(pyemdgrid PROPERTIES INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set_target_properties(pyemdgrid PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()
endif()

if(BUILD_TESTING)
  set(PYEMDGRID_TEST_ENV "PYTHONPATH=$<TARGET_FILE_DIR:pyemdgrid>")

  if(USE_ASAN)
    # Disable LeakSanitizer in python test as we get too many false positives from pybind
    list(APPEND PYEMDGRID_TEST_ENV "ASAN_OPTIONS=detect_leaks=0")
    if(ASAN_PRELOAD_LIBS)
      list(APPEND PYEMDGRID_TEST_ENV "LD_PRELOAD=${ASAN_PRELOAD_LIBS}")
    endif()
  endif()

  add_test(
    NAME pyemdgrid_import
    COMMAND
      ${CMAKE_COMMAND} -E env ${PYEMDGRID_TEST_ENV}
      ${Python3_EXECUTABLE} -c
      "import pyemdgrid; assert pyemdgrid.version() == '${EMDGRID_GIT_VERSION}'")

  find_program(PYTEST_EXE NAMES pytest pytest3)
  if(PYTEST_EXE)
    add_test(
      NAME pyemdgrid_pytest
      COMMAND
        ${CMAKE_COMMAND} -E env ${PYEMDGRID_TEST_ENV}
        ${Python3_EXECUTABLE} -m pytest
        ${CMAKE_SOURCE_DIR}/tests/test_emdgrid.py -v)
  endif()
endif()
