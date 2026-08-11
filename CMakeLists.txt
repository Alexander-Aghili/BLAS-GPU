cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(BLAS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(blas_lib STATIC
  blas.cu
)

target_include_directories(blas_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(blas_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(blas_lib PUBLIC CUDA::cudart)

add_executable(blas
  main.cpp
  main.cu
)

target_link_libraries(blas PRIVATE
  blas_lib
  CUDA::cublas
)

add_executable(bench
  bench.cu
)

target_link_libraries(bench PRIVATE blas_lib)

enable_testing()
add_subdirectory(tests)
