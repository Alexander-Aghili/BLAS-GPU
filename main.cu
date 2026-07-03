#include "blas.cuh"

#include <cuda_runtime.h>

#include <cstdio>

__global__ void hello_kernel() {
    printf("Hello from the GPU (block %d, thread %d)\n", blockIdx.x, threadIdx.x);
}

int run_cublas(int /*argc*/, char** /*argv*/) {
    std::printf("Hello from the CPU\n");

    hello_kernel<<<1, 4>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
