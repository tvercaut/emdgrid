option(USE_ASAN "Activate ASan compiler/linker options" OFF)

if(USE_ASAN)
  find_library(
    ASAN_LIBRARY
    NAMES
      asan
      libasan
      libclang_rt.asan-x86_64
      libclang_rt.asan-aarch64
  )

  if(NOT ASAN_LIBRARY)
    message(STATUS
      "find_library() could not locate an ASan runtime. "
      "Falling back to compiler-provided location."
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE ASAN_LIBRARY
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      # GCC returns the argument unchanged on failure.
      if(ASAN_LIBRARY STREQUAL "libasan.so")
        unset(ASAN_LIBRARY)
      endif()

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-runtime-dir
        OUTPUT_VARIABLE CLANG_RUNTIME_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      find_library(
        ASAN_LIBRARY
        NAMES
          libclang_rt.asan-x86_64
          libclang_rt.asan-aarch64
        PATHS ${CLANG_RUNTIME_DIR}
        NO_DEFAULT_PATH
      )
    endif()
  endif()

  if(NOT ASAN_LIBRARY)
    message(FATAL_ERROR
      "Unable to locate the AddressSanitizer runtime."
    )
  endif()

  message(STATUS "Using ASAN_LIBRARY: ${ASAN_LIBRARY}")
  
  if(MSVC)
    add_compile_options(/fsanitize=address)
    add_link_options(/fsanitize=address)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>"
        PARENT_SCOPE)
  else()
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
  endif()

  set(ASan_FOUND TRUE)

  if(NOT MSVC)
    set(PRELOAD_LIBS "")
    # Preload the ASan runtime to ensure it is loaded before any
    # instrumented shared libraries. This avoids the common
    # "ASan runtime does not come first" issue.
    if(ASAN_LIBRARY)
      list(APPEND PRELOAD_LIBS "${ASAN_LIBRARY}")
    endif()

    # Also preload the C++ runtime associated with the compiler that
    # built the project. This can help avoid runtime ABI mismatches
    # when Python extension modules are loaded under CI.
    #
    # We intentionally ask the compiler for the location instead of
    # using find_library(), as this guarantees we get the exact
    # libstdc++ selected by the active toolchain.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
        OUTPUT_VARIABLE LIBSTDCXX_LIBRARY
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      # GCC returns the argument unchanged if the library cannot
      # be located.
      if(LIBSTDCXX_LIBRARY STREQUAL "libstdc++.so.6")
        unset(LIBSTDCXX_LIBRARY)
      endif()

      if(LIBSTDCXX_LIBRARY)
        message(STATUS "Using LIBSTDCXX_LIBRARY: ${LIBSTDCXX_LIBRARY}")
        list(APPEND PRELOAD_LIBS "${LIBSTDCXX_LIBRARY}")
      endif()
    endif()

    if(PRELOAD_LIBS)
      string(REPLACE ";" " " PRELOAD_STR "${PRELOAD_LIBS}")
      message(STATUS "ASAN_PRELOAD_LIBS=${PRELOAD_STR}")
      set(ASAN_PRELOAD_LIBS "${PRELOAD_STR}" CACHE INTERNAL
          "Preload libraries for ASan")
    endif()
  endif()
endif()
