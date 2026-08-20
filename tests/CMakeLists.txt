add_executable(emdgrid_tests test_emdgrid.cpp)
target_link_libraries(emdgrid_tests PRIVATE emdgrid)
add_test(NAME emdgrid_tests COMMAND emdgrid_tests)
