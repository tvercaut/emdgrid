option(USE_ASAN "Activate ASan compiler/linker options" OFF)

if(USE_ASAN)
  find_library(ASAN_LIBRARY NAMES asan libasan)
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
    if(ASAN_LIBRARY)
      list(APPEND PRELOAD_LIBS "${ASAN_LIBRARY}")
    endif()

    if(PRELOAD_LIBS)
      string(REPLACE ";" " " PRELOAD_STR "${PRELOAD_LIBS}")
      set(ASAN_PRELOAD_LIBS "${PRELOAD_STR}" CACHE INTERNAL
          "Preload libraries for ASan")
    endif()
  endif()
endif()
