add_executable(emdgrid_tests test_emdgrid.cpp)
target_link_libraries(emdgrid_tests PRIVATE emdgrid)
target_compile_definitions(emdgrid_tests PRIVATE EMDGRID_VERSION="${PROJECT_VERSION}")
add_test(NAME emdgrid_tests COMMAND emdgrid_tests)
