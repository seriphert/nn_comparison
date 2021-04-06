cmake_minimum_required(VERSION 3.12)

SET(LIB_NAME swarms-test)

SET(TARGETS
  pso_swarm_test
  clpso_swarm_test
  iclpso_swarm_test
)

SET(HEADERS ${TARGETS})
list(TRANSFORM HEADERS APPEND ".hpp")

SET(SOURCES ${TARGETS})
list(TRANSFORM SOURCES APPEND ".cpp")

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
