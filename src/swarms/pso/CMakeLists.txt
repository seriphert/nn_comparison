SET(LIB_NAME pso_swarm)

SET(HEADERS ${LIB_NAME}.hpp)

SET(SOURCES ${LIB_NAME}.cpp)

add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})

target_link_libraries(${LIB_NAME} PRIVATE ${COVERAGE_LIB})
