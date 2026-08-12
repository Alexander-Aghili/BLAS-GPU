#include <vector>

#include "blas.hpp"

#define DOT_N (1 << 24)
#define GEMV_N 4096
#define GEMM_N 2048
#define ITERS 4

int main() {
    std::vector<real_t> hx(DOT_N, 1), hy(DOT_N, 2);
    const Vector x{hx.data(), DOT_N, 1};
    const Vector y{hy.data(), DOT_N, 1};
    for (int i = 0; i < ITERS; ++i) dot(x, y);

    std::vector<real_t> hA((size_t)GEMV_N * GEMV_N, 1), hv(GEMV_N, 1), hw(GEMV_N, 0);
    const Matrix A{hA.data(), GEMV_N, GEMV_N, GEMV_N};
    const Vector v{hv.data(), GEMV_N, 1};
    Vector w{hw.data(), GEMV_N, 1};
    for (int i = 0; i < ITERS; ++i) gemv("N", (real_t)1, A, v, (real_t)0, w);

    std::vector<real_t> hGA((size_t)GEMM_N * GEMM_N, 1), hGB((size_t)GEMM_N * GEMM_N, 1), hGC((size_t)GEMM_N * GEMM_N, 0);
    const Matrix GA{hGA.data(), GEMM_N, GEMM_N, GEMM_N};
    const Matrix GB{hGB.data(), GEMM_N, GEMM_N, GEMM_N};
    Matrix GC{hGC.data(), GEMM_N, GEMM_N, GEMM_N};
    for (int i = 0; i < ITERS; ++i) gemm("N", "N", (real_t)1, GA, GB, (real_t)0, GC);

    return 0;
}
