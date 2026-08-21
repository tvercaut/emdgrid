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
  add_test(
    NAME pyemdgrid_import
    COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyemdgrid>
      ${Python3_EXECUTABLE} -c
      "import pyemdgrid; assert pyemdgrid.add(2, 3) == 5; assert pyemdgrid.version() == '${emdgrid_VERSION}'")
endif()
