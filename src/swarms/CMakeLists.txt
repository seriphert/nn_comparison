cmake_minimum_required(VERSION 3.12)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

SET(LIB_NAME swarms)

SET(DIRECTORIES
  base
  pso
  clpso
  iclpso
)

SET(LIBRARIES ${DIRECTORIES})
list(REMOVE_ITEM LIBRARIES "base")
list(TRANSFORM LIBRARIES APPEND "_swarm")
list(APPEND LIBRARIES "swarm")

include(Common)
add_subdirectories("${DIRECTORIES}")

add_library(${LIB_NAME} INTERFACE)

target_link_libraries(${LIB_NAME} INTERFACE ${LIBRARIES})
