# Function to add all passed directories
function(add_subdirectories subdirectories)
  foreach(dir ${subdirectories})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
        add_subdirectory(${dir})
      endif()
    endif()
  endforeach(dir)
endfunction(add_subdirectories)

function(add_arguments_to_coverage_lib coverage_lib)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Add required flags (GCC & LLVM/Clang)
    target_compile_options(${coverage_lib} INTERFACE
      -O0        # no optimization
      -g         # generate debug info
      --coverage # sets all required flags
    )

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
      target_link_options(${coverage_lib} INTERFACE --coverage)
    else()
      target_link_libraries(${coverage_lib} INTERFACE --coverage)
    endif()
  endif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
endfunction(add_arguments_to_coverage_lib)
