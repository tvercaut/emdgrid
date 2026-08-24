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

  if(USE_ASAN AND NOT MSVC AND NOT APPLE)
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
      OUTPUT_VARIABLE ASAN_LIB_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
      OUTPUT_VARIABLE STDCXX_LIB_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(PRELOAD_LIBS "")
    if(EXISTS "${ASAN_LIB_PATH}")
      list(APPEND PRELOAD_LIBS "${ASAN_LIB_PATH}")
    endif()
    if(EXISTS "${STDCXX_LIB_PATH}")
      list(APPEND PRELOAD_LIBS "${STDCXX_LIB_PATH}")
    endif()

    if(PRELOAD_LIBS)
      string(REPLACE ";" " " PRELOAD_STR "${PRELOAD_LIBS}")
      list(APPEND PYEMDGRID_TEST_ENV "LD_PRELOAD=${PRELOAD_STR}" "ASAN_OPTIONS=detect_leaks=0")
    else()
      list(APPEND PYEMDGRID_TEST_ENV "ASAN_OPTIONS=detect_leaks=0")
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
