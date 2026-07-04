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

//rotg: generate a Givens rotation (real: srotg/drotg)
void rotg(real_t& a, real_t& b, real_t& c, real_t& s);

//rotg: generate a Givens rotation (complex: crotg/zrotg); c is always real
void rotg(complex_t& a, complex_t& b, real_t& c, complex_t& s);

//rot: apply a Givens rotation to vectors x and y; c and s are real
//even for complex vectors (srot/drot, csrot/zdrot)
template <typename T>
void rot(Vector<T>& x, Vector<T>& y, real_t c, real_t s);

//rotmg: generate a modified Givens rotation (real only in BLAS)
//d1, d2, x1 are updated in place; param[5] receives the rotation
void rotmg(real_t& d1, real_t& d2, real_t& x1, real_t y1, real_t param[5]);

//rotm: apply a modified Givens rotation to vectors x and y (real only in BLAS)
void rotm(Vector<real_t>& x, Vector<real_t>& y, const real_t param[5]);


//Level 2 BLAS
//
//gemv: y = alpha * A * x + beta * y
template <typename T> 
void gemv(const char* trans, T alpha, const Matrix<T>& A, const Vector<T>& x, T beta, const Vector<T>& y);

//hemv: y = alpha * A * x + beta * y
void hemv(const char* uplo, complex_t& alpha, const Matrix<complex_t>& A, const Vector<complex_t>& x, complex_t beta, const Vector<complex_t>& y);

//symv: y = alpha * A * x + beta * y
void symv(const char* uplo, real_t alpha, const Matrix<real_t>& A, const Vector<real_t>& x, real_t beta, const Vector<real_t>& y);

//trmv: x=A*x 
template <typename T> 
void trmv(const char* uplo, const char* trans, const char* diag, const Matrix<T>& A, Vector<T>& x);

//trsv: x=A-*x
template <typename T>
void trsv(const char* uplo, const char* trans, const char* diag, const Matrix<T>& A, Vector<T>& x);

//ger: A = A+ alpha * x * y^T
void ger(real_t alpha, const Vector<real_t>& x, const Vector<real_t>& y, Matrix<real_t>& A);

//geru : A = A + alpha * x * y^T
void geru(complex_t alpha, const Vector<complex_t>& x, const Vector<complex_t>& y, const Matrix<complex_t>& A);

//gerc : A = A + alpha * x * y^H
void gerc(complex_t alpha, const Vector<complex_t>& x, const Vector<complex_t>& y, const Matrix<complex_t>& A);

//syr : A = A + alpha * x * x^T
void syr(const char* uplo, real_t alpha, const Vector<real_t>& x, Matrix<real_t>& A);

//her : A = A + alpha * x * x^H
void her(const char* uplo, complex_t alpha, const Vector<complex_t>& x, Matrix<complex_t>& A);

//syr2 : A = A + alpha * x * y^T
void syr2(const char* uplo, real_t alpha, const Vector<real_t>& x, const Vector<real_t>& y, Matrix<real_t>& A);

//her2 : A = A + alpha * x * y^H
void her2(const char* uplo, complex_t alpha, const Vector<complex_t>& x, const Vector<complex_t>& y, Matrix<complex_t>& A);

//TODO LEVEL 2 BLAS, band storage 



//LEVEL 3 BLAS

//gemm: C = alpha * A * B + beta * C
template <typename T>
void gemm (const char* transa, const char* transb, T alpha, const Matrix<T>& A, const Matrix<T>& B, T beta, Matrix<T>& C);

//gemmtr: C = alpha * A^T * B + beta * C
template <typename T>
void gemmtr (const char* uplo, const char* trans, T alpha, const Matrix<T>& A, const Matrix<T>& B, T beta, Matrix<T>& C);

//symm: C = alpha * A * B + beta * C
template <typename T>
void symm (const char* side, const char* uplo, T alpha, const Matrix<T>& A, const Matrix<T>& B, T beta, Matrix<T>& C);

//hemm: C = alpha * A * B + beta * C
void hemm (const char* side, const char* uplo, complex_t alpha, const Matrix<complex_t>& A, const Matrix<complex_t>& B, complex_t beta, Matrix<complex_t>& C);

//trmm: C = alpha*A^*B
template <typename T>
void trmm (const char* side, const char* uplo, const char* transA, const char* diag, T alpha, const Matrix<T>& A, const Matrix<T>& B);

//trsm : C = alpha*A^-*B
template <typename T>
void trsm (const char* side, const char* uplo, const char* transA, const char* diag, T alpha, const Matrix<T>& A, const Matrix<T>& B);

//syrk : C = alpha * A * A^T + beta * C
template <typename T>
void syrk (const char* uplo, const char* trans, T alpha, const Matrix<T>& A, T beta, Matrix<T>& C);

//herk : C = alpha * A * A^H + beta * C
void herk (const char* uplo, const char* trans, complex_t alpha, const Matrix<complex_t>& A, complex_t beta, Matrix<complex_t>& C);

//syr2k : C = alpha * A * B^T + beta * C
template <typename T>
void syr2k (const char* uplo, const char* trans, T alpha, const Matrix<T>& A, const Matrix<T>& B, T beta, Matrix<T>& C);

//her2k : C = alpha * A * B^H + beta * C
void her2k (const char* uplo, const char* trans, complex_t alpha, const Matrix<complex_t>& A, const Matrix<complex_t>& B, complex_t beta, Matrix<complex_t>& C);

