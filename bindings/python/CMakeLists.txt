find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.1.0)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(pyemdgrid pyemdgrid.cpp)
target_link_libraries(pyemdgrid PRIVATE emdgrid)

if(BUILD_TESTING)
  set(PYEMDGRID_TEST_ENV "PYTHONPATH=$<TARGET_FILE_DIR:pyemdgrid>")

  if(USE_ASAN AND ASAN_PRELOAD_LIBS)
    list(APPEND PYEMDGRID_TEST_ENV "LD_PRELOAD=${ASAN_PRELOAD_LIBS}")
  endif()

  add_test(
    NAME pyemdgrid_import
    COMMAND
      ${CMAKE_COMMAND} -E env ${PYEMDGRID_TEST_ENV}
      ${Python3_EXECUTABLE} -c
      "import pyemdgrid; assert pyemdgrid.version() == '${EMDGRID_GIT_VERSION}'")

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pytest"
    RESULT_VARIABLE PYTEST_MODULE_RESULT
    OUTPUT_QUIET ERROR_QUIET)
  if(PYTEST_MODULE_RESULT EQUAL 0)
    add_test(
      NAME pyemdgrid_pytest
      COMMAND
        ${CMAKE_COMMAND} -E env ${PYEMDGRID_TEST_ENV}
        ${Python3_EXECUTABLE} -m pytest
        ${CMAKE_SOURCE_DIR}/tests/test_emdgrid.py -v)
  endif()
endif()
