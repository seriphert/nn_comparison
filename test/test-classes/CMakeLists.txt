SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

SET(LIB_NAME test-classes)

SET(DIRECTORIES swarms)

SET(LIBRARIES swarms-test)

include(Common)
add_subdirectories("${DIRECTORIES}")

add_library(${LIB_NAME} INTERFACE)

target_link_libraries(${LIB_NAME} INTERFACE ${LIBRARIES})
