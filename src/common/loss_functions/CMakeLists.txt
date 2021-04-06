cmake_minimum_required(VERSION 3.12)

SET(LIB_NAME loss_functions)

SET(TARGETS
  loss_function
  mean_squared_error
)

SET(HEADERS ${TARGETS})
list(TRANSFORM HEADERS APPEND ".hpp")

SET(SOURCES ${TARGETS})
list(TRANSFORM SOURCES APPEND ".cpp")

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
