// One translation unit per historical revision of blas.cu.
//
// Compiled once per revision with -DREV=<code> -DREV_SRC="src/<hash>.cu".
// The revision's blas.cu is textually included inside its own namespace so
// that every revision's kernels, macros and static helpers stay separate.
// Headers are included first: their #pragma once makes the copies inside the
// namespace no-ops, so only the revision's own definitions land in it.
// Each launcher below reproduces that revision's host-side launch geometry.
#include <cmath>
#include <cuda_runtime.h>
#include "blas.hpp"
#include "launchers.hpp"

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define NS CAT(rev_, REV_TAG)

namespace NS {
#include REV_SRC
}

static Vector vec(const real_t* p, int n) { return Vector{const_cast<real_t*>(p), n, 1}; }
static Matrix mat(const real_t* p, int n) { return Matrix{const_cast<real_t*>(p), n, n, n}; }

#if defined(WANT_DOT)
void CAT(dot_, REV_TAG)(const real_t* x, const real_t* y, int n, real_t* result) {
#if REV_DOT_CFG == 0
    // 4da826e: occupancy API picks the block size.
    static int min_grid = 0, block = 0;
    if (block == 0) CUDA_ERROR_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, NS::dot_kernel));
    const int grid = (n + block - 1) / block;
    NS::dot_kernel<<<grid, block>>>(vec(x, n), vec(y, n), result);
#elif REV_DOT_CFG == 1
    // 3affd9b, 47a65d8: one element per thread.
    const int grid = (n + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE;
    NS::dot_kernel<<<grid, DOT_BLOCK_SIZE>>>(vec(x, n), vec(y, n), result);
#elif REV_DOT_CFG == 2
    // 7099f40, 98cc847: half the grid, two elements per thread.
    const int grid = (n + 2L * DOT_BLOCK_SIZE - 1) / (2L * DOT_BLOCK_SIZE);
    NS::dot_kernel<<<grid, DOT_BLOCK_SIZE>>>(vec(x, n), vec(y, n), result);
#endif
}
#endif

#if defined(WANT_GEMV)
void CAT(gemv_, REV_TAG)(const real_t* A, const real_t* x, real_t* y, int n) {
    const long m = n;
#if REV_GEMV_CFG == 0
    // 051d7c3: 256-thread blocks, grid = m / 512 (8 blocks at m = 4096).
    const int grid = (m + 2L * BLOCK_SIZE - 1) / (2L * BLOCK_SIZE);
    NS::gemv_kernel<<<grid, BLOCK_SIZE>>>(real_t(1), mat(A, n), vec(x, n), real_t(0), vec(y, n), m, (long)n, NoTransAt());
#elif REV_GEMV_CFG == 1
    // 15e6bc6 (unchanged through HEAD): 32 rows x 16 column splits.
    const dim3 block(GEMV_ROWS, GEMV_SPLITS);
    const int grid = (m + GEMV_ROWS - 1) / GEMV_ROWS;
    NS::gemv_kernel<<<grid, block>>>(real_t(1), mat(A, n), vec(x, n), real_t(0), vec(y, n), m, (long)n, NoTransAt());
#endif
}
#endif

#if defined(WANT_GEMM)
void CAT(gemm_, REV_TAG)(const real_t* A, const real_t* B, real_t* C, int n) {
    const long m = n, nn = n, k = n;
#if REV_GEMM_CFG == 0
    // 15e6bc6 and 89b175b: one output per thread, 16 x 16 blocks.
    const dim3 block(16, 16);
    long gx = (m + block.x - 1) / block.x;
    long gy = (nn + block.y - 1) / block.y;
#else
    // a44068e onward: 16 x 16 threads covering a GEMM_BLOCK_TILE square.
    const dim3 block(GEMM_TILE, GEMM_TILE);
    long gx = (m + GEMM_BLOCK_TILE - 1) / GEMM_BLOCK_TILE;
    long gy = (nn + GEMM_BLOCK_TILE - 1) / GEMM_BLOCK_TILE;
#endif
    if (gx > 65535) gx = 65535;
    if (gy > 65535) gy = 65535;
    const dim3 grid((unsigned)gx, (unsigned)gy);
    Matrix Cm = mat(C, n);
    NS::gemm_kernel<<<grid, block>>>(real_t(1), mat(A, n), mat(B, n), real_t(0), Cm, m, nn, k, NoTransAt(), NoTransAt());
}
#endif
