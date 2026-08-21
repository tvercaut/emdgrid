# Determine a version string from git describe, falling back to "NOTAVAILABLE"
# if git is not found or no tags/commits exist.
#
# Sets the variable EMDGRID_GIT_VERSION in the caller's scope.

find_package(Git QUIET)

set(EMDGRID_GIT_VERSION "NOTAVAILABLE")

if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE _git_describe
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result)
  if(_git_result EQUAL 0 AND _git_describe)
    set(EMDGRID_GIT_VERSION "${_git_describe}")
  endif()
endif()

message(STATUS "emdgrid version: ${EMDGRID_GIT_VERSION}")
