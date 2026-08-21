include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.12)
FetchContent_MakeAvailable(doctest)

add_executable(emdgrid_tests test_emdgrid.cpp)
target_link_libraries(emdgrid_tests PRIVATE doctest::doctest emdgrid)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(emdgrid_tests)
