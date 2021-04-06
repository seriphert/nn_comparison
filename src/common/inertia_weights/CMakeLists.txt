cmake_minimum_required(VERSION 3.12)

SET(LIB_NAME inertia_weights)

SET(TARGETS
  inertia_weight
  linear_decreasing_inertia_weight
)

SET(HEADERS ${TARGETS})
list(TRANSFORM HEADERS APPEND ".hpp")

SET(SOURCES ${TARGETS})
list(TRANSFORM SOURCES APPEND ".cpp")

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
