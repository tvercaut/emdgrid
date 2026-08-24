option(USE_ASAN "Activate ASan compiler/linker options" OFF)

if(USE_ASAN)
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

  if(NOT MSVC AND NOT APPLE)
    find_library(ASAN_LIBRARY NAMES asan libasan)
    if(NOT ASAN_LIBRARY)
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE ASAN_DRIVER_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(EXISTS "${ASAN_DRIVER_PATH}")
        set(ASAN_LIBRARY "${ASAN_DRIVER_PATH}")
      endif()
    endif()

    find_library(STDCXX_LIBRARY NAMES stdc++ libstdc++.so.6)
    if(NOT STDCXX_LIBRARY)
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
        OUTPUT_VARIABLE STDCXX_DRIVER_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(EXISTS "${STDCXX_DRIVER_PATH}")
        set(STDCXX_LIBRARY "${STDCXX_DRIVER_PATH}")
      endif()
    endif()

    set(PRELOAD_LIBS "")
    if(ASAN_LIBRARY)
      list(APPEND PRELOAD_LIBS "${ASAN_LIBRARY}")
    endif()
    if(STDCXX_LIBRARY)
      list(APPEND PRELOAD_LIBS "${STDCXX_LIBRARY}")
    endif()

    if(PRELOAD_LIBS)
      string(REPLACE ";" " " PRELOAD_STR "${PRELOAD_LIBS}")
      set(ASAN_PRELOAD_LIBS "${PRELOAD_STR}" CACHE INTERNAL
          "Preload libraries for ASan")
    endif()
  endif()
endif()
