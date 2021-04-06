cmake_minimum_required(VERSION 3.12)

SET(LIB_NAME activation_functions)

SET(TARGETS
  activation_function
  sigmoid
)

SET(HEADERS ${TARGETS})
list(TRANSFORM HEADERS APPEND ".hpp")

SET(SOURCES ${TARGETS})
list(TRANSFORM SOURCES APPEND ".cpp")

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
