include(FetchContent)

find_package(cli11_proj QUIET)
if(cli11_proj_FOUND OR TARGET CLI11)
  message(STATUS "Found cli11_proj: ${cli11_proj_DIR}")
else()
  message(STATUS "Could NOT find cli11_proj: Building it")
  FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.7.2
    SYSTEM
  )

  FetchContent_MakeAvailable(cli11_proj)
endif()

add_executable(emdgrid_example emdgrid_example.cpp)
target_link_libraries(emdgrid_example PRIVATE emdgrid CLI11::CLI11)
