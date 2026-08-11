#pragma once

//Header files for Basic Linear Algebra Subprograms (BLAS) in CUDA
//

#include <cuda_runtime.h>


#include "types.hpp"
#include "macros.hpp"

//Level 1 BLAS

//axpy: y = alpha * x + y
//T is real_t or complex_t; x and y must have the same n.
template <typename T>
void axpy(T alpha, const Vector<T>& x, Vector<T>& y);

//scal: y = alpha * y
template <typename T>
void scal(T alpha, Vector<T>& y);

//copy: y = x
template <typename T>
void copy(const Vector<T>& x, Vector<T>& y);

//swap: y = x
template <typename T>
void swap(Vector<T>& x, Vector<T>& y);

//dot: x \cdot y (real only)
real_t dot(const Vector<real_t>& x, const Vector<real_t>& y);

//dotu: sum x_i * y_i (complex, unconjugated)
complex_t dotu(const Vector<complex_t>& x, const Vector<complex_t>& y);

//dotc: sum conj(x_i) * y_i (complex)
complex_t dotc(const Vector<complex_t>& x, const Vector<complex_t>& y);

//nrm2: 2-norm (always returns a real scalar)
template <typename T>
real_t nrm2(const Vector<T>& x);

//asum: 1-norm; for complex, sum of |Re| + |Im| (always returns a real scalar)
template <typename T>
real_t asum(const Vector<T>& x);

//iamax: index of the element with the largest |x_i| (|Re| + |Im| for complex)
template <typename T>
int iamax(const Vector<T>& x);


//Level 2 BLAS
//
//gemv: y = alpha * A * x + beta * y
template <typename T>
void gemv(const char* trans, T alpha, const Matrix<T>& A, const Vector<T>& x, T beta, Vector<T>& y);

//hemv: y = alpha * A^* * x + beta * y
void hemv(const char* uplo, complex_t alpha, const Matrix<complex_t>& A, const Vector<complex_t>& x, complex_t beta, const Vector<complex_t>& y);

//symv: y = alpha * A^T * x + beta * y
void symv(const char* uplo, real_t alpha, const Matrix<real_t>& A, const Vector<real_t>& x, real_t beta, const Vector<real_t>& y);

//trmv: x=A*x
template <typename T>
void trmv(const char* uplo, const char* trans, const char* diag, const Matrix<T>& A, Vector<T>& x);


//LEVEL 3 BLAS

//gemm: C = alpha * A * B + beta * C
template <typename T>
void gemm (const char* transa, const char* transb, T alpha, const Matrix<T>& A, const Matrix<T>& B, T beta, Matrix<T>& C);

//gemmtr: C = alpha * A^T * B + beta * C
template <typename T>
void gemmtr (const char* uplo, const char* trans, T alpha, const Matrix<T>& A, const Matrix<T>& B, T beta, Matrix<T>& C);

