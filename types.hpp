#pragma once

#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

//CUDA Type Definitions for Basic Linear Algebra Subprograms (BLAS)


#ifdef DOUBLE_PRECISION
    typedef double real_t;
#else
    typedef float real_t;
#endif

using complex_t = cuda::std::complex<real_t>;

//Vector: a strided view of n elements at data[0], data[inc], data[2*inc], ...
//A matrix column is {data + j*ld, rows, 1}; a row is {data + i, cols, ld}.
template <typename T>
struct Vector {
    T* data;
    int n;
    int inc;  // stride between elements, in units of T (1 = contiguous)
};

//Matrix
template <typename T>
struct Matrix {
    T* data;
    int rows, cols, ld;
};


template <typename T>
struct NoTransAt {
  __device__ T operator()(const Matrix<T>& A, long i, long j) const {
      return A.data[i + j * A.ld];
  }
};

template <typename T>
struct TransAt {
  __device__ T operator()(const Matrix<T>& A, long i, long j) const {
      return A.data[j + i * A.ld];
  }
};

template <typename T>
struct ConjTransAt {
  __device__ T operator()(const Matrix<T>& A, long i, long j) const {
      if constexpr (cuda::std::is_same_v<T, complex_t>) {
          return conj(A.data[j + i * A.ld]);
      } else {
          return A.data[j + i * A.ld];
      }
  }
};
