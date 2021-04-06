SET(LIB_NAME clpso_network)

SET(HEADERS ${LIB_NAME}.hpp)

SET(SOURCES ${LIB_NAME}.cpp)

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
