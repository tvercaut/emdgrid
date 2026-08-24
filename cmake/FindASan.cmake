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
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
      OUTPUT_VARIABLE ASAN_LIB_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
      OUTPUT_VARIABLE STDCXX_LIB_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(PRELOAD_LIBS "")
    if(EXISTS "${ASAN_LIB_PATH}")
      list(APPEND PRELOAD_LIBS "${ASAN_LIB_PATH}")
    endif()
    if(EXISTS "${STDCXX_LIB_PATH}")
      list(APPEND PRELOAD_LIBS "${STDCXX_LIB_PATH}")
    endif()

    if(PRELOAD_LIBS)
      string(REPLACE ";" " " PRELOAD_STR "${PRELOAD_LIBS}")
      set(ASAN_PRELOAD_LIBS "${PRELOAD_STR}" CACHE INTERNAL
          "Preload libraries for ASan")
    endif()
  endif()
endif()
