cmake_minimum_required(VERSION 3.12)

SET(LIB_NAME params)

SET(TARGETS
  backprop_params
  base_pso_params
  pso_params
  clpso_params
)

SET(HEADERS ${TARGETS})
list(TRANSFORM HEADERS APPEND ".hpp")

SET(SOURCES ${TARGETS})
list(TRANSFORM SOURCES APPEND ".cpp")

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
