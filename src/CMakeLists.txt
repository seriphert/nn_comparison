cmake_minimum_required(VERSION 3.12)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

SET(APP_SOURCES main.cpp)

SET(DIRECTORIES
  common
  layers
  swarms
  networks
)

include(Common)
add_subdirectories("${DIRECTORIES}")

# Build application executable
add_executable(${APP_NAME} ${APP_SOURCES})

target_link_libraries(${APP_NAME} PRIVATE ${DIRECTORIES})

# Build application library
add_library(${APP_LIB} INTERFACE)

target_link_libraries(${APP_LIB} INTERFACE ${DIRECTORIES})
