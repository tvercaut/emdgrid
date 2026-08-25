include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.12
  SYSTEM
)
FetchContent_MakeAvailable(doctest)

if(TARGET doctest)
  set_target_properties(doctest PROPERTIES CXX_CLANG_TIDY "")
endif()

add_executable(emdgrid_tests test_emdgrid.cpp)
target_link_libraries(emdgrid_tests PRIVATE doctest::doctest emdgrid)
target_compile_definitions(emdgrid_tests PRIVATE EMDGRID_VERSION="${emdgrid_VERSION}")
set_target_properties(emdgrid_tests PROPERTIES CXX_CLANG_TIDY "")

add_executable(utils_tests test_utils.cpp)
target_link_libraries(utils_tests PRIVATE doctest::doctest emdgrid)
set_target_properties(utils_tests PROPERTIES CXX_CLANG_TIDY "")

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(emdgrid_tests)
doctest_discover_tests(utils_tests)
