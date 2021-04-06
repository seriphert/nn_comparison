SET(LIB_NAME dataset)

SET(HEADERS
  dataset_type.hpp
  ${LIB_NAME}.hpp
)

SET(SOURCES ${LIB_NAME}.cpp)

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
