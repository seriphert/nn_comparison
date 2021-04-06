cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

SET(PROJECT_NAME nn_comparison)
project($(PROJECT_NAME))

option(ENABLE_CODE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_TESTS "Build the testing tree." OFF)

# Application variables
SET(APP_NAME main)
SET(APP_LIB ${APP_NAME}_lib)
SET(TEST_APP_NAME test-${APP_NAME})

# Application directories
SET(DIRECTORIES
  datasets
  src
)

# Include useful functions
include(Common)

# Configure code coverage
SET(COVERAGE_LIB coverage_config)
add_library(${COVERAGE_LIB} INTERFACE)

if (ENABLE_CODE_COVERAGE)
  add_arguments_to_coverage_lib(${COVERAGE_LIB})
endif()

# Build application libraries
add_subdirectories("${DIRECTORIES}")

# Build test
if (ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
