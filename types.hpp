#pragma once

#include <cuda_runtime.h>

//CUDA Type Definitions for Basic Linear Algebra Subprograms (BLAS)


#ifdef DOUBLE_PRECISION
    typedef double real_t;
#else
    typedef float real_t;
#endif

//Vector: a strided view of n elements at data[0], data[inc], data[2*inc], ...
//A matrix column is {data + j*ld, rows, 1}; a row is {data + i, cols, ld}.
struct Vector {
    real_t* data;
    int n;
    int inc;  // stride between elements, in units of real_t (1 = contiguous)
};

//Matrix
struct Matrix {
    real_t* data;
    int rows, cols, ld;
};


struct NoTransAt {
    __device__ real_t operator()(const Matrix& A, long i, long j) const {
        return A.data[i + j * A.ld];
    }
};

struct TransAt {
    __device__ real_t operator()(const Matrix& A, long i, long j) const {
        return A.data[j + i * A.ld];
    }
};

struct UpperAt {
    __device__ real_t operator()(const Matrix& A, long i, long j) const {
        return (j >= i) ? A.data[i + j * A.ld] : A.data[j + i * A.ld];
    }
};

struct LowerAt {
    __device__ real_t operator()(const Matrix& A, long i, long j) const {
        return (j <= i) ? A.data[i + j * A.ld] : A.data[j + i * A.ld];
    }
};
