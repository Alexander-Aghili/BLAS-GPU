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

enable_language(Fortran)

add_library(blat_shim STATIC netlib/blat_shim.cpp)
target_link_libraries(blat_shim PUBLIC blas_lib)

if(CMAKE_CUDA_FLAGS MATCHES "DOUBLE_PRECISION" OR CMAKE_CXX_FLAGS MATCHES "DOUBLE_PRECISION")
  set(BLAT_PRECISIONS d z)
else()
  set(BLAT_PRECISIONS s c)
endif()

foreach(prec IN LISTS BLAT_PRECISIONS)
  foreach(level IN ITEMS 2 3)
    set(tester xblat${level}${prec})
    add_executable(${tester} netlib/${prec}blat${level}.f)
    target_link_libraries(${tester} PRIVATE blat_shim ${NETLIB_BLAS} stdc++)
    add_test(NAME netlib_${prec}blat${level}
      COMMAND sh -c "exec '$<TARGET_FILE:${tester}>' < '${CMAKE_CURRENT_SOURCE_DIR}/netlib/${prec}blat${level}.in'")
    set_tests_properties(netlib_${prec}blat${level} PROPERTIES
      PASS_REGULAR_EXPRESSION "END OF TESTS"
      FAIL_REGULAR_EXPRESSION "FAILED|SUSPECT|FATAL ERROR")
  endforeach()
endforeach()
