#include <vector>

#include <cublas_v2.h>

#include "blas.hpp"
#include "macros.hpp"

#define DOT_N (1 << 24)
#define GEMV_N 4096
#define GEMM_N 2048
#define ITERS 4

#ifdef DOUBLE_PRECISION
#define CUBLAS_DOT cublasDdot
#define CUBLAS_GEMV cublasDgemv
#define CUBLAS_GEMM cublasDgemm
#else
#define CUBLAS_DOT cublasSdot
#define CUBLAS_GEMV cublasSgemv
#define CUBLAS_GEMM cublasSgemm
#endif

static void bench_cublas_dot(const std::vector<real_t>& hx, const std::vector<real_t>& hy) {
    real_t* dx = nullptr;
    real_t* dy = nullptr;
    real_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&dx, DOT_N * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&dy, DOT_N * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemcpy(dx, hx.data(), DOT_N * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(dy, hy.data(), DOT_N * sizeof(real_t), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    for (int i = 0; i < ITERS; ++i) CUBLAS_DOT(handle, DOT_N, dx, 1, dy, 1, d_result);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cublasDestroy(handle);

    CUDA_ERROR_CHECK(cudaFree(dx));
    CUDA_ERROR_CHECK(cudaFree(dy));
    CUDA_ERROR_CHECK(cudaFree(d_result));
}

static void bench_cublas_gemv(const std::vector<real_t>& hA, const std::vector<real_t>& hv) {
    real_t* dA = nullptr;
    real_t* dv = nullptr;
    real_t* dw = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&dA, hA.size() * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&dv, GEMV_N * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&dw, GEMV_N * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(dv, hv.data(), GEMV_N * sizeof(real_t), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const real_t alpha = 1, beta = 0;
    for (int i = 0; i < ITERS; ++i)
        CUBLAS_GEMV(handle, CUBLAS_OP_N, GEMV_N, GEMV_N, &alpha, dA, GEMV_N, dv, 1, &beta, dw, 1);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cublasDestroy(handle);

    CUDA_ERROR_CHECK(cudaFree(dA));
    CUDA_ERROR_CHECK(cudaFree(dv));
    CUDA_ERROR_CHECK(cudaFree(dw));
}

static void bench_cublas_gemm(const std::vector<real_t>& hA, const std::vector<real_t>& hB) {
    real_t* dA = nullptr;
    real_t* dB = nullptr;
    real_t* dC = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&dA, hA.size() * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&dB, hB.size() * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&dC, (size_t)GEMM_N * GEMM_N * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(real_t), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const real_t alpha = 1, beta = 0;
    for (int i = 0; i < ITERS; ++i)
        CUBLAS_GEMM(handle, CUBLAS_OP_N, CUBLAS_OP_N, GEMM_N, GEMM_N, GEMM_N, &alpha, dA, GEMM_N, dB, GEMM_N, &beta, dC, GEMM_N);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cublasDestroy(handle);

    CUDA_ERROR_CHECK(cudaFree(dA));
    CUDA_ERROR_CHECK(cudaFree(dB));
    CUDA_ERROR_CHECK(cudaFree(dC));
}

int main() {
    std::vector<real_t> hx(DOT_N, 1), hy(DOT_N, 2);
    const Vector x{hx.data(), DOT_N, 1};
    const Vector y{hy.data(), DOT_N, 1};
    for (int i = 0; i < ITERS; ++i) dot(x, y);

    bench_cublas_dot(hx, hy);

    std::vector<real_t> hA((size_t)GEMV_N * GEMV_N, 1), hv(GEMV_N, 1), hw(GEMV_N, 0);
    const Matrix A{hA.data(), GEMV_N, GEMV_N, GEMV_N};
    const Vector v{hv.data(), GEMV_N, 1};
    Vector w{hw.data(), GEMV_N, 1};
    for (int i = 0; i < ITERS; ++i) gemv("N", (real_t)1, A, v, (real_t)0, w);

    bench_cublas_gemv(hA, hv);

    std::vector<real_t> hGA((size_t)GEMM_N * GEMM_N, 1), hGB((size_t)GEMM_N * GEMM_N, 1), hGC((size_t)GEMM_N * GEMM_N, 0);
    const Matrix GA{hGA.data(), GEMM_N, GEMM_N, GEMM_N};
    const Matrix GB{hGB.data(), GEMM_N, GEMM_N, GEMM_N};
    Matrix GC{hGC.data(), GEMM_N, GEMM_N, GEMM_N};
    for (int i = 0; i < ITERS; ++i) gemm("N", "N", (real_t)1, GA, GB, (real_t)0, GC);

    bench_cublas_gemm(hGA, hGB);

    return 0;
}
