#!/bin/bash
set -euo pipefail

# System variables
BROWSER=/usr/bin/firefox # path to executable
MKDIR="mkdir -p"

# App variables
APP_NAME="main"
TEST_APP_NAME="test-main"

# Target directories
BUILD_DIR="build"
TEST_DIR="test"

# Coverage variables 
COVERAGE_DIR="code_coverage"
COVERAGE_HTML="coverage.html"
COVERAGE_TITLE="Code Coverage Report"

$MKDIR $BUILD_DIR && cd $BUILD_DIR

# Configure and build project and its tests
cmake -DENABLE_CODE_COVERAGE=ON -DENABLE_TESTS=ON ..
cmake --build . -- -j 3
./$TEST_DIR/$TEST_APP_NAME
#ctest -j 3 --output-on-failure # alternative for manual tests execution

# [GCOVR] Coverage report generation
$MKDIR $COVERAGE_DIR && cd $COVERAGE_DIR
gcovr \
  --branches \
  --exclude-unreachable-branches \
  --exclude-throw-branches \
  --exclude-directories $TEST_DIR \
  -r ../.. \
  --html \
  --html-details \
  --html-title "$COVERAGE_TITLE" \
  -o $COVERAGE_HTML

$BROWSER $COVERAGE_HTML

# [LCOV] Coverage report generation
#lcov --rc lcov_branch_coverage=1 --capture --directory ./src --output-file coverage.info
#lcov --rc lcov_branch_coverage=1 --remove coverage.info '/usr/*' --output-file coverage.info
#lcov --rc lcov_branch_coverage=1 --list coverage.info
#genhtml --rc lcov_branch_coverage=1 coverage.info --output-directory out
#$BROWSER ./out/index.html
