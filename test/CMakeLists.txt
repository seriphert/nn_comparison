cmake_minimum_required(VERSION 3.12)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(GTest REQUIRED)
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

SET(SOURCES
  test-layer.cpp
  test-backprop-layer.cpp
  test-pso-layer.cpp
  test-swarm.cpp
  test-pso-swarm.cpp
  test-clpso-swarm.cpp
)

SET(LIBRARIES
  mocks
  test-classes
)

include(Common)
add_subdirectories("${LIBRARIES}")

add_executable(${TEST_APP_NAME} ${SOURCES})

target_link_libraries(${TEST_APP_NAME} PRIVATE GTest::GTest)
target_link_libraries(${TEST_APP_NAME} PRIVATE GTest::Main)
target_link_libraries(${TEST_APP_NAME} PRIVATE gmock)
target_link_libraries(${TEST_APP_NAME} PRIVATE ${APP_LIB})
target_link_libraries(${TEST_APP_NAME} PRIVATE ${LIBRARIES})

add_test(NAME MainTest COMMAND ${TEST_APP_NAME})
