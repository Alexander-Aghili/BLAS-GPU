#pragma once

//Header files for Basic Linear Algebra Subprograms (BLAS) in CUDA
//

#include <cuda_runtime.h>


#include "types.hpp"
#include "macros.hpp"

//Level 1 BLAS

//axpy: y = alpha * x + y
//x and y must have the same n.
void axpy(real_t alpha, const Vector& x, Vector& y);

//scal: y = alpha * y
void scal(real_t alpha, Vector& y);

//copy: y = x
void copy(const Vector& x, Vector& y);

//swap: y = x
void swap(Vector& x, Vector& y);

//dot: x \cdot y
real_t dot(const Vector& x, const Vector& y);

//nrm2: 2-norm
real_t nrm2(const Vector& x);

//asum: 1-norm
real_t asum(const Vector& x);

//iamax: index of the element with the largest |x_i|
int iamax(const Vector& x);


//Level 2 BLAS
//
//gemv: y = alpha * A * x + beta * y
void gemv(const char* trans, real_t alpha, const Matrix& A, const Vector& x, real_t beta, Vector& y);

//symv: y = alpha * A^T * x + beta * y
void symv(const char* uplo, real_t alpha, const Matrix& A, const Vector& x, real_t beta, const Vector& y);

//trmv: x=A*x
void trmv(const char* uplo, const char* trans, const char* diag, const Matrix& A, Vector& x);


//LEVEL 3 BLAS

//gemm: C = alpha * A * B + beta * C
void gemm(const char* transa, const char* transb, real_t alpha, const Matrix& A, const Matrix& B, real_t beta, Matrix& C);

//gemmtr: C = alpha * A^T * B + beta * C
void gemmtr(const char* uplo, const char* trans, real_t alpha, const Matrix& A, const Matrix& B, real_t beta, Matrix& C);
