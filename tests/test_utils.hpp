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
void chemv_(const char* uplo, const int* m, const cuda::std::complex<float>* alpha, const cuda::std::complex<float>* A,
            const int* lda, const cuda::std::complex<float>* x, const int* incx, const cuda::std::complex<float>* beta,
            cuda::std::complex<float>* y, const int* incy);
void zhemv_(const char* uplo, const int* m, const cuda::std::complex<double>* alpha, const cuda::std::complex<double>* A,
            const int* lda, const cuda::std::complex<double>* x, const int* incx, const cuda::std::complex<double>* beta,
            cuda::std::complex<double>* y, const int* incy);
void ssymv_(const char* uplo, const int* m, const float* alpha, const float* A, const int* lda,
            const float* x, const int* incx, const float* beta, float* y, const int* incy);
void dsymv_(const char* uplo, const int* m, const double* alpha, const double* A, const int* lda,
            const double* x, const int* incx, const double* beta, double* y, const int* incy);
void sgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const float* alpha,
            const float* A, const int* lda, const float* B, const int* ldb, const float* beta, float* C, const int* ldc);
void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha,
            const double* A, const int* lda, const double* B, const int* ldb, const double* beta, double* C, const int* ldc);
void cgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
            const cuda::std::complex<float>* alpha, const cuda::std::complex<float>* A, const int* lda,
            const cuda::std::complex<float>* B, const int* ldb, const cuda::std::complex<float>* beta,
            cuda::std::complex<float>* C, const int* ldc);
void zgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
            const cuda::std::complex<double>* alpha, const cuda::std::complex<double>* A, const int* lda,
            const cuda::std::complex<double>* B, const int* ldb, const cuda::std::complex<double>* beta,
            cuda::std::complex<double>* C, const int* ldc);
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

template <typename T>
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
inline Matrix<T> random_matrix_hermitian(T* data, int rows, int cols) {
    static std::mt19937 gen(12345);
    std::uniform_real_distribution<real_t> dist(0, 10);
    Matrix<T> m{data, rows, cols, rows};
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i <= j; ++i) {
            if constexpr (std::is_same_v<T, complex_t>) {
                m.data[i + j * m.ld] = (i == j) ? complex_t(dist(gen), 0) : complex_t(dist(gen), dist(gen));
                m.data[j + i * m.ld] = conj(m.data[i + j * m.ld]);
            } else {
                m.data[i + j * m.ld] = dist(gen);
                m.data[j + i * m.ld] = m.data[i + j * m.ld];
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

template <typename T>
inline void verify_vector_near(const Vector<T>& v, const std::vector<T>& expected, real_t rel_tol) {
    ASSERT_EQ(static_cast<size_t>(v.n), expected.size());
    std::vector<T> host(static_cast<size_t>(v.n) * v.inc);
    ASSERT_EQ(cudaMemcpy(host.data(), v.data, host.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < v.n; ++i) {
        const T& got = host[static_cast<size_t>(i) * v.inc];
        const T& want = expected[i];
        if constexpr (std::is_same_v<T, complex_t>) {
            const real_t tol = abs(want) * rel_tol;
            EXPECT_NEAR(got.real(), want.real(), tol) << "real mismatch at index " << i;
            EXPECT_NEAR(got.imag(), want.imag(), tol) << "imag mismatch at index " << i;
        } else {
            EXPECT_NEAR(got, want, std::abs(want) * rel_tol) << "mismatch at index " << i;
        }
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

inline void ref_hemv(const char* uplo, int m, complex_t alpha, const complex_t* A, int lda, const complex_t* x, int incx, complex_t beta, complex_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    zhemv_(uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy);
#else
    chemv_(uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy);
#endif
}

inline void ref_symv(const char* uplo, int m, real_t alpha, const real_t* A, int lda, const real_t* x, int incx, real_t beta, real_t* y, int incy) {
#ifdef DOUBLE_PRECISION
    dsymv_(uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy);
#else
    ssymv_(uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy);
#endif
}

inline void ref_gemm(const char* transa, const char* transb, int m, int n, int k, real_t alpha, const real_t* A, int lda,
                     const real_t* B, int ldb, real_t beta, real_t* C, int ldc) {
#ifdef DOUBLE_PRECISION
    dgemm_(transa, transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
#else
    sgemm_(transa, transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
#endif
}

inline void ref_gemm(const char* transa, const char* transb, int m, int n, int k, complex_t alpha, const complex_t* A, int lda,
                     const complex_t* B, int ldb, complex_t beta, complex_t* C, int ldc) {
#ifdef DOUBLE_PRECISION
    zgemm_(transa, transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
#else
    cgemm_(transa, transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
#endif
}
