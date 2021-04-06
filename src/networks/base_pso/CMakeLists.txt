SET(LIB_NAME base_pso_network)

SET(HEADERS ${LIB_NAME}.hpp)

SET(SOURCES ${LIB_NAME}.cpp)

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
