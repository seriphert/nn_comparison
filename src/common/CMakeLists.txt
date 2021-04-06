SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

SET(LIB_NAME common)

SET(LIBRARIES
  activation_functions
  dataset
  generator
  inertia_weights
  loss_functions
  params
)

include(Common)
add_subdirectories("${LIBRARIES}")

add_library(${LIB_NAME} INTERFACE)

target_link_libraries(${LIB_NAME} INTERFACE ${LIBRARIES})

target_include_directories(${LIB_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${LIB_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/enum)

target_include_directories(${LIB_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
