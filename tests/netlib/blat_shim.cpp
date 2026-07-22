#include "blas.hpp"

#ifdef DOUBLE_PRECISION
#define SHIM_RGEMV dgemv_
#define SHIM_RSYMV dsymv_
#define SHIM_RGEMM dgemm_
#define SHIM_CGEMV zgemv_
#define SHIM_CHEMV zhemv_
#define SHIM_CGEMM zgemm_
#else
#define SHIM_RGEMV sgemv_
#define SHIM_RSYMV ssymv_
#define SHIM_RGEMM sgemm_
#define SHIM_CGEMV cgemv_
#define SHIM_CHEMV chemv_
#define SHIM_CGEMM cgemm_
#endif

template <typename T>
static Vector<T> fortran_vector(const T* base, int n, int inc) {
    T* first = (inc < 0 && n > 0) ? const_cast<T*>(base) + (long)(n - 1) * -inc : const_cast<T*>(base);
    return Vector<T>{first, n, inc};
}

template <typename T>
static Matrix<T> fortran_matrix(const T* base, int rows, int cols, int ld) {
    return Matrix<T>{const_cast<T*>(base), rows, cols, ld};
}

template <typename T>
static void shim_gemv(const char* trans, int m, int n, T alpha, const T* a, int lda,
                      const T* x, int incx, T beta, T* y, int incy) {
    const bool notrans = (*trans == 'N' || *trans == 'n');
    const Matrix<T> A = fortran_matrix(a, m, n, lda);
    const Vector<T> vx = fortran_vector(x, notrans ? n : m, incx);
    Vector<T> vy = fortran_vector(y, notrans ? m : n, incy);
    gemv(trans, alpha, A, vx, beta, vy);
}

template <typename T>
static void shim_gemm(const char* transa, const char* transb, int m, int n, int k, T alpha,
                      const T* a, int lda, const T* b, int ldb, T beta, T* c, int ldc) {
    const bool na = (*transa == 'N' || *transa == 'n');
    const bool nb = (*transb == 'N' || *transb == 'n');
    const Matrix<T> A = fortran_matrix(a, na ? m : k, na ? k : m, lda);
    const Matrix<T> B = fortran_matrix(b, nb ? k : n, nb ? n : k, ldb);
    Matrix<T> C = fortran_matrix(c, m, n, ldc);
    gemm(transa, transb, alpha, A, B, beta, C);
}

extern "C" {

void SHIM_RGEMV(const char* trans, const int* m, const int* n, const real_t* alpha,
                const real_t* a, const int* lda, const real_t* x, const int* incx,
                const real_t* beta, real_t* y, const int* incy) {
    shim_gemv(trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

void SHIM_CGEMV(const char* trans, const int* m, const int* n, const complex_t* alpha,
                const complex_t* a, const int* lda, const complex_t* x, const int* incx,
                const complex_t* beta, complex_t* y, const int* incy) {
    shim_gemv(trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

void SHIM_RSYMV(const char* uplo, const int* n, const real_t* alpha, const real_t* a,
                const int* lda, const real_t* x, const int* incx, const real_t* beta,
                real_t* y, const int* incy) {
    const Matrix<real_t> A = fortran_matrix(a, *n, *n, *lda);
    const Vector<real_t> vx = fortran_vector(x, *n, *incx);
    Vector<real_t> vy = fortran_vector(y, *n, *incy);
    symv(uplo, *alpha, A, vx, *beta, vy);
}

void SHIM_CHEMV(const char* uplo, const int* n, const complex_t* alpha, const complex_t* a,
                const int* lda, const complex_t* x, const int* incx, const complex_t* beta,
                complex_t* y, const int* incy) {
    const Matrix<complex_t> A = fortran_matrix(a, *n, *n, *lda);
    const Vector<complex_t> vx = fortran_vector(x, *n, *incx);
    Vector<complex_t> vy = fortran_vector(y, *n, *incy);
    hemv(uplo, *alpha, A, vx, *beta, vy);
}

void SHIM_RGEMM(const char* transa, const char* transb, const int* m, const int* n,
                const int* k, const real_t* alpha, const real_t* a, const int* lda,
                const real_t* b, const int* ldb, const real_t* beta, real_t* c,
                const int* ldc) {
    shim_gemm(transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

void SHIM_CGEMM(const char* transa, const char* transb, const int* m, const int* n,
                const int* k, const complex_t* alpha, const complex_t* a, const int* lda,
                const complex_t* b, const int* ldb, const complex_t* beta, complex_t* c,
                const int* ldc) {
    shim_gemm(transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

}
