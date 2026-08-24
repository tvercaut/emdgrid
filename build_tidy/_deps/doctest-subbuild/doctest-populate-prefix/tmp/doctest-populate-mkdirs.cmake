# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/app/build_tidy/_deps/doctest-src"
  "/app/build_tidy/_deps/doctest-build"
  "/app/build_tidy/_deps/doctest-subbuild/doctest-populate-prefix"
  "/app/build_tidy/_deps/doctest-subbuild/doctest-populate-prefix/tmp"
  "/app/build_tidy/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp"
  "/app/build_tidy/_deps/doctest-subbuild/doctest-populate-prefix/src"
  "/app/build_tidy/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/app/build_tidy/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/app/build_tidy/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
