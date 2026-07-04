include(FetchContent)

find_library(NETLIB_BLAS
  NAMES blas
  PATHS /usr/lib/x86_64-linux-gnu/blas
  NO_DEFAULT_PATH
  REQUIRED
)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_SOURCES
  level1_tests.cpp
  level2_tests.cpp
  level3_tests.cpp
)

set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(blas_tests ${TEST_SOURCES})

target_link_libraries(blas_tests PRIVATE
  blas_lib
  GTest::gtest_main
  ${NETLIB_BLAS}
)

include(GoogleTest)
gtest_discover_tests(blas_tests DISCOVERY_MODE PRE_TEST)
