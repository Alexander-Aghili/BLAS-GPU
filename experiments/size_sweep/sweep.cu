// Size sweep: kernel-only timings of the historical DOT/GEMV/GEMM kernels and
// cuBLAS on device-resident FP32 data (inc = 1, ld = n, alpha = 1, beta = 0).
//
// At each size, every implementation (cuBLAS included) is timed round-robin:
// rep r of impl 0, rep r of impl 1, ... so that the laptop GPU's shifting
// clock/P-states affect all implementations alike. Each timed rep: flush L2
// (read a 96 MiB scratch buffer, leaving only clean lines so no write-backs
// land inside the timed kernel), reset the DOT accumulator, record a start
// event, launch, record a stop event. Cold L2 mirrors Nsight Compute's default
// cache control; clocks are not locked (ncu locks them to base by default).
//
// Usage: sweep [passes=3] [anchors]
// Output (stdout): sweep,routine,impl,rev,size,time_us_median,time_us_min,reps,rel_err
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include "launchers.hpp"
#include "macros.hpp"

#define CUBLAS_CHECK(call)                                                                   \
    do {                                                                                     \
        cublasStatus_t s = call;                                                             \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                    \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

static float* flush_buf = nullptr;
static float* flush_sink = nullptr;
static const size_t FLUSH_BYTES = 96ull << 20;  // RTX 4060 Laptop L2 is 32 MiB
static cudaEvent_t ev0, ev1;

__global__ void flush_kernel(const float4* __restrict__ p, size_t n, float* sink) {
    float acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5f) *sink = acc;  // never true; keeps the loads alive
}

static void flush_l2() {
    flush_kernel<<<24 * 8, 256>>>(reinterpret_cast<const float4*>(flush_buf), FLUSH_BYTES / sizeof(float4), flush_sink);
}

struct Job {
    const char* impl;
    const char* rev;
    std::function<void()> launch;     // enqueues exactly the measured work
    std::function<void()> reset;      // untimed setup before each launch
    std::function<double()> check;    // relative error of the last result vs cuBLAS
};

static double timed_once(const Job& j, bool flush) {
    if (flush) flush_l2();
    j.reset();
    CUDA_ERROR_CHECK(cudaEventRecord(ev0));
    j.launch();
    CUDA_ERROR_CHECK(cudaEventRecord(ev1));
    CUDA_ERROR_CHECK(cudaEventSynchronize(ev1));
    float ms;
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    return ms * 1e3;
}

// jobs[0] must be cuBLAS: it is run last before the checks so the reference
// buffers hold its result.
static void run_size(int sweep, const char* routine, long size, std::vector<Job>& jobs,
                     double budget_ms, int min_reps, int max_reps) {
    const size_t J = jobs.size();
    std::vector<int> reps(J);
    for (size_t i = 0; i < J; ++i) {
        double est = 0;
        for (int w = 0; w < 3; ++w) est = timed_once(jobs[i], false);
        reps[i] = (int)std::clamp(budget_ms * 1e3 / std::max(est, 1.0), (double)min_reps, (double)max_reps);
    }
    const int R = *std::max_element(reps.begin(), reps.end());
    std::vector<std::vector<double>> t(J);
    for (int r = 0; r < R; ++r)
        for (size_t i = 0; i < J; ++i)
            if (r < reps[i]) t[i].push_back(timed_once(jobs[i], true));
    CUDA_ERROR_CHECK(cudaGetLastError());

    timed_once(jobs[0], false);  // reference result
    for (size_t i = 0; i < J; ++i) {
        double err = 0;
        if (i > 0) {
            timed_once(jobs[i], false);
            err = jobs[i].check();
        }
        auto& v = t[i];
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        const double med = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        std::printf("%d,%s,%s,%s,%ld,%.3f,%.3f,%zu,%.3e\n", sweep, routine, jobs[i].impl, jobs[i].rev, size, med, v[0],
                    n, err);
    }
    std::fflush(stdout);
}

static double rel_err_vec(const real_t* d, const real_t* d_ref, size_t count) {
    std::vector<real_t> a(count), b(count);
    CUDA_ERROR_CHECK(cudaMemcpy(a.data(), d, count * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(b.data(), d_ref, count * sizeof(real_t), cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < count; ++i) {
        const double e = (double)a[i] - (double)b[i];
        num += e * e;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

static real_t* random_device(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<real_t> h(count);
    for (auto& v : h) v = u(rng);
    real_t* d = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d, count * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemcpy(d, h.data(), count * sizeof(real_t), cudaMemcpyHostToDevice));
    return d;
}

static real_t* zeros_device(size_t count) {
    real_t* d = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d, count * sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemset(d, 0, count * sizeof(real_t)));
    return d;
}

int main(int argc, char** argv) {
    const int sweeps = argc > 1 ? std::atoi(argv[1]) : 3;
    const bool anchors = argc > 2 && std::string(argv[2]) == "anchors";  // article sizes only
    const double budget_ms = 250.0;  // per (impl, size, pass)

    CUDA_ERROR_CHECK(cudaEventCreate(&ev0));
    CUDA_ERROR_CHECK(cudaEventCreate(&ev1));
    CUDA_ERROR_CHECK(cudaMalloc(&flush_buf, FLUSH_BYTES));
    CUDA_ERROR_CHECK(cudaMemset(flush_buf, 0, FLUSH_BYTES));
    CUDA_ERROR_CHECK(cudaMalloc(&flush_sink, sizeof(float)));
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    const real_t one = 1, zero = 0;
    auto none = [] {};

    std::vector<long> dot_sizes, gemv_sizes, gemm_sizes;
    for (int p = 12; p <= 26; ++p) dot_sizes.push_back(1L << p);
    gemv_sizes = {256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    gemm_sizes = {128, 256, 512, 1024, 1536, 2048, 3072, 4096};
    if (anchors) {
        dot_sizes = {1L << 24};
        gemv_sizes = {4096};
        gemm_sizes = {2048};
    }

    const long dot_max = 1L << 26, gemv_max = 8192, gemm_max = 4096;
    real_t* dx = random_device(dot_max, 1);
    real_t* dy = random_device(dot_max, 2);
    real_t* dres = zeros_device(1);
    real_t* dref = zeros_device(1);
    real_t* gA = random_device((size_t)gemv_max * gemv_max, 3);
    real_t* gx = random_device(gemv_max, 4);
    real_t* gy = zeros_device(gemv_max);
    real_t* gyref = zeros_device(gemv_max);
    real_t* mA = random_device((size_t)gemm_max * gemm_max, 5);
    real_t* mB = random_device((size_t)gemm_max * gemm_max, 6);
    real_t* mC = zeros_device((size_t)gemm_max * gemm_max);
    real_t* mCref = zeros_device((size_t)gemm_max * gemm_max);

    using DotFn = void (*)(const real_t*, const real_t*, int, real_t*);
    using MatFn = void (*)(const real_t*, const real_t*, real_t*, int);
    struct Impl { const char* impl; const char* rev; void* fn; };
    const Impl dots[] = {
        {"atomic", "4da826e", (void*)dot_4da826e},
        {"block_reduction", "3affd9b", (void*)dot_3affd9b},
        {"sequential_addressing", "47a65d8", (void*)dot_47a65d8},
        {"two_per_thread", "7099f40", (void*)dot_7099f40},
        {"warp_shuffle", "98cc847", (void*)dot_98cc847},
    };
    const Impl gemvs[] = {
        {"eight_block_baseline", "051d7c3", (void*)gemv_051d7c3},
        {"row_column_split", "15e6bc6", (void*)gemv_15e6bc6},
    };
    const Impl gemms[] = {
        {"naive", "15e6bc6", (void*)gemm_15e6bc6},
        {"shared_tiles", "89b175b", (void*)gemm_89b175b},
        {"register_4x4", "a44068e", (void*)gemm_a44068e},
        {"register_8x8", "6d1f43a", (void*)gemm_6d1f43a},
        {"double_buffer", "c11b4fd", (void*)gemm_c11b4fd},
        {"double_buffer_worktree", "worktree", (void*)gemm_worktree},
    };

    std::printf("sweep,routine,impl,rev,size,time_us_median,time_us_min,reps,rel_err\n");
    for (int s = 0; s < sweeps; ++s) {
        CUBLAS_CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));
        for (long n : dot_sizes) {
            const int ni = (int)n;
            std::vector<Job> jobs;
            jobs.push_back({"cublas", "cublas", [=] { CUBLAS_CHECK(cublasSdot(h, ni, dx, 1, dy, 1, dref)); }, none,
                            [] { return 0.0; }});
            for (const auto& d : dots) {
                const DotFn fn = (DotFn)d.fn;
                jobs.push_back({d.impl, d.rev, [=] { fn(dx, dy, ni, dres); },
                                [=] { CUDA_ERROR_CHECK(cudaMemsetAsync(dres, 0, sizeof(real_t))); },
                                [=] {
                                    real_t got, ref;
                                    CUDA_ERROR_CHECK(cudaMemcpy(&got, dres, sizeof(real_t), cudaMemcpyDeviceToHost));
                                    CUDA_ERROR_CHECK(cudaMemcpy(&ref, dref, sizeof(real_t), cudaMemcpyDeviceToHost));
                                    return std::fabs((double)got - ref) / std::fabs((double)ref);
                                }});
            }
            run_size(s, "dot", n, jobs, budget_ms, 20, 2000);
        }
        CUBLAS_CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
        for (long n : gemv_sizes) {
            const int ni = (int)n;
            std::vector<Job> jobs;
            jobs.push_back({"cublas", "cublas",
                            [=, &one, &zero] { CUBLAS_CHECK(cublasSgemv(h, CUBLAS_OP_N, ni, ni, &one, gA, ni, gx, 1, &zero, gyref, 1)); },
                            none, [] { return 0.0; }});
            for (const auto& g : gemvs) {
                const MatFn fn = (MatFn)g.fn;
                jobs.push_back({g.impl, g.rev, [=] { fn(gA, gx, gy, ni); },
                                [=] { CUDA_ERROR_CHECK(cudaMemsetAsync(gy, 0xff, n * sizeof(real_t))); },  // NaN canary
                                [=] { return rel_err_vec(gy, gyref, n); }});
            }
            run_size(s, "gemv", n, jobs, budget_ms, 20, 2000);
        }
        for (long n : gemm_sizes) {
            const int ni = (int)n;
            std::vector<Job> jobs;
            jobs.push_back({"cublas", "cublas",
                            [=, &one, &zero] {
                                CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, ni, ni, ni, &one, mA, ni, mB, ni,
                                                         &zero, mCref, ni));
                            },
                            none, [] { return 0.0; }});
            for (const auto& g : gemms) {
                const MatFn fn = (MatFn)g.fn;
                jobs.push_back({g.impl, g.rev, [=] { fn(mA, mB, mC, ni); }, none,
                                [=] { return rel_err_vec(mC, mCref, (size_t)n * n); }});
            }
            run_size(s, "gemm", n, jobs, budget_ms, 20, 1000);
        }
    }
    cublasDestroy(h);
    return 0;
}
