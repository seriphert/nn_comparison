cmake_minimum_required(VERSION 3.12)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

SET(LIB_NAME networks)

SET(DIRECTORIES
  base
  base_pso
  backprop
  pso
  clpso
  iclpso
)

SET(LIBRARIES ${DIRECTORIES})
list(TRANSFORM LIBRARIES APPEND "_network")

include(Common)
add_subdirectories("${DIRECTORIES}")

add_library(${LIB_NAME} INTERFACE)

target_link_libraries(${LIB_NAME} INTERFACE ${LIBRARIES})
