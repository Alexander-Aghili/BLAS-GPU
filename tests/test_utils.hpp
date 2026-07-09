#pragma once

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

#include "blas.hpp"

constexpr int n = 4096;

extern "C" {
float sdot_(const int* n, const float* x, const int* incx, const float* y, const int* incy);
double ddot_(const int* n, const double* x, const int* incx, const double* y, const int* incy);
void scopy_(const int* n, const float* x, const int* incx, float* y, const int* incy);
void dcopy_(const int* n, const double* x, const int* incx, double* y, const int* incy);
void sswap_(const int* n, float* x, const int* incx, float* y, const int* incy);
void dswap_(const int* n, double* x, const int* incx, double* y, const int* incy);
cuda::std::complex<float> cdotu_(const int* n, const cuda::std::complex<float>* x, const int* incx,
                                 const cuda::std::complex<float>* y, const int* incy);
cuda::std::complex<double> zdotu_(const int* n, const cuda::std::complex<double>* x, const int* incx,
                                  const cuda::std::complex<double>* y, const int* incy);
cuda::std::complex<float> cdotc_(const int* n, const cuda::std::complex<float>* x, const int* incx,
                                 const cuda::std::complex<float>* y, const int* incy);
cuda::std::complex<double> zdotc_(const int* n, const cuda::std::complex<double>* x, const int* incx,
                                  const cuda::std::complex<double>* y, const int* incy);
float snrm2_(const int* n, const float* x, const int* incx);
double dnrm2_(const int* n, const double* x, const int* incx);
float scnrm2_(const int* n, const cuda::std::complex<float>* x, const int* incx);
double dznrm2_(const int* n, const cuda::std::complex<double>* x, const int* incx);
float sasum_(const int* n, const float* x, const int* incx);
double dasum_(const int* n, const double* x, const int* incx);
float scasum_(const int* n, const cuda::std::complex<float>* x, const int* incx);
double dzasum_(const int* n, const cuda::std::complex<double>* x, const int* incx);
void sgemv_(const char* trans, const int* m, const int* n, const float* alpha, const float* A, const int* lda,
            const float* x, const int* incx, const float* beta, float* y, const int* incy);
void dgemv_(const char* trans, const int* m, const int* n, const double* alpha, const double* A, const int* lda,
            const double* x, const int* incx, const double* beta, double* y, const int* incy);
}

template <typename T = real_t>
inline std::vector<T> random_vector(int size) {
    static std::mt19937 gen(12345);
    std::uniform_real_distribution<real_t> dist(0, 10);
    std::vector<T> v(size);
    for (auto& e : v) {
        if constexpr (std::is_same_v<T, complex_t>) {
            e = complex_t(dist(gen), dist(gen));
        } else {
            e = dist(gen);
        }
    }
    return v;
}

template <typename T = real_t>
inline Matrix<T> random_matrix(T* data, int rows, int cols) {
    static std::mt19937 gen(12345);
    std::uniform_real_distribution<real_t> dist(0, 10);
    Matrix<T> m{data, rows, cols, rows};
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if constexpr (std::is_same_v<T, complex_t>) {
                m.data[i + j * m.ld] = complex_t(dist(gen), dist(gen));
            } else {
                m.data[i + j * m.ld] = dist(gen);
            }
        }
    }
    return m;
}

template <typename T>
inline void verify_vector(const Vector<T>& v, const std::vector<T>& expected) {
    ASSERT_EQ(static_cast<size_t>(v.n), expected.size());
    std::vector<T> host(static_cast<size_t>(v.n) * v.inc);
    ASSERT_EQ(cudaMemcpy(host.data(), v.data, host.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < v.n; ++i) {
        EXPECT_EQ(host[static_cast<size_t>(i) * v.inc], expected[i]) << "mismatch at index " << i;
    }
}

inline void verify_vector_near(const Vector<real_t>& v, const std::vector<real_t>& expected, real_t rel_tol) {
    ASSERT_EQ(static_cast<size_t>(v.n), expected.size());
    std::vector<real_t> host(static_cast<size_t>(v.n) * v.inc);
    ASSERT_EQ(cudaMemcpy(host.data(), v.data, host.size() * sizeof(real_t), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < v.n; ++i) {
        EXPECT_NEAR(host[static_cast<size_t>(i) * v.inc], expected[i], std::abs(expected[i]) * rel_tol)
            << "mismatch at index " << i;
    }
}

inline void ref_copy(int size, const real_t* x, int incx, real_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    dcopy_(&size, x, &incx, y, &incy);
#else
    scopy_(&size, x, &incx, y, &incy);
#endif
}

inline void ref_swap(int size, real_t* x, int incx, real_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    dswap_(&size, x, &incx, y, &incy);
#else
    sswap_(&size, x, &incx, y, &incy);
#endif
}

inline real_t ref_dot(int size, const real_t* x, int incx, const real_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    return ddot_(&size, x, &incx, y, &incy);
#else
    return sdot_(&size, x, &incx, y, &incy);
#endif
}

inline complex_t ref_dotu(int size, const complex_t* x, int incx, const complex_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    return zdotu_(&size, x, &incx, y, &incy);
#else
    return cdotu_(&size, x, &incx, y, &incy);
#endif
}

inline complex_t ref_dotc(int size, const complex_t* x, int incx, const complex_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    return zdotc_(&size, x, &incx, y, &incy);
#else
    return cdotc_(&size, x, &incx, y, &incy);
#endif
}

inline real_t ref_nrm2(int size, const real_t* x, int incx) {
#ifdef DOUBLE_PRECISION
    return dnrm2_(&size, x, &incx);
#else
    return snrm2_(&size, x, &incx);
#endif
}

inline real_t ref_nrm2(int size, const complex_t* x, int incx) {
#ifdef DOUBLE_PRECISION
    return dznrm2_(&size, x, &incx);
#else
    return scnrm2_(&size, x, &incx);
#endif
}

inline real_t ref_asum(int size, const real_t* x, int incx) {
#ifdef DOUBLE_PRECISION
    return dasum_(&size, x, &incx);
#else
    return sasum_(&size, x, &incx);
#endif
}

inline real_t ref_asum(int size, const complex_t* x, int incx) {
#ifdef DOUBLE_PRECISION
    return dzasum_(&size, x, &incx);
#else
    return scasum_(&size, x, &incx);
#endif
}

inline void ref_gemv(const char* trans, int m, int n, real_t alpha, const real_t* A, int lda, const real_t* x, int incx, real_t beta, real_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    dgemv_(trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
#else
    sgemv_(trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
#endif
}
