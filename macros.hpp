#pragma once

#include <iostream>
#include <cuda_runtime.h>

//Macros for CUDA
//

//CUDA error checking
#define CUDA_ERROR_CHECK(call)                                                             \
    do {                                                                                  \
        cudaError_t status = call;                                                        \
        if (status != cudaSuccess) {                                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;       \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

#define GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, kernel)                                                   \
    do {										  \
	CUDA_ERROR_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel)); \
    } while (0)
