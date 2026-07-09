#include "blas.hpp"

#include <cmath>

#include <cuda/std/type_traits>

//Level 1 BLAS

//axpy: y = alpha * x + y
template <typename T>
__global__ void axpy_kernel(T alpha, Vector<T> x, Vector<T> y) {
    const T* __restrict__ xp = x.data;
    T* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n;
         i += (long)gridDim.x * blockDim.x) {
        yp[i * y.inc] += alpha * xp[i * x.inc];
    }
}

template <typename T>
void axpy(T alpha, const Vector<T>& x, Vector<T>& y) {
    if (x.n <= 0 || alpha == T(0)) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, axpy_kernel<T>);

    const int grid = (x.n + block - 1) / block;
    axpy_kernel<<<grid, block>>>(alpha, x, y);
    CUDA_ERROR_CHECK(cudaGetLastError());
}

template void axpy<real_t>(real_t, const Vector<real_t>&, Vector<real_t>&);
template void axpy<complex_t>(complex_t, const Vector<complex_t>&, Vector<complex_t>&);

//scal: y = alpha * y
template <typename T>
__global__ void scal_kernel(T alpha, Vector<T> y) {
    T* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < y.n; i+= (long)gridDim.x * blockDim.x) {
	yp[i * y.inc] *= alpha;
    }
}

template <typename T>
void scal(T alpha, Vector<T>& y) {
    if (y.n <= 0) return;
    
    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, scal_kernel<T>);

    const int grid = (y.n + block - 1) / block;
    scal_kernel<<<grid, block>>>(alpha, y);
    CUDA_ERROR_CHECK(cudaGetLastError());
}

template void scal<real_t>(real_t, Vector<real_t>&);
template void scal<complex_t>(complex_t, Vector<complex_t>&);

template <typename T>
__global__ void copy_kernel(const Vector<T> x, Vector<T> y) {
    const T* __restrict__ xp = x.data;
    T* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	yp[i * y.inc] = xp[i * x.inc];
    }
}

template <typename T>
void copy(const Vector<T>& x, Vector<T>& y) {
    if (x.n <= 0 || y.n <= 0) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, copy_kernel<T>);

    const int grid = (x.n + block - 1) / block;
    copy_kernel<<<grid, block>>>(x, y);
    CUDA_ERROR_CHECK(cudaGetLastError());
}

template void copy<real_t>(const Vector<real_t>&, Vector<real_t>&);
template void copy<complex_t>(const Vector<complex_t>&, Vector<complex_t>&);

template <typename T>
__global__ void swap_kernel(Vector<T> x, Vector<T> y) {
    T* __restrict__ xp = x.data;
    T* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
        T tmp = xp[i * x.inc];
	xp[i * x.inc] = yp[i * y.inc];
	yp[i * y.inc] = tmp;
    }
}

template <typename T>
void swap(Vector<T>& x, Vector<T>& y) {
    if (x.n <= 0 || y.n <= 0) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, swap_kernel<T>);

    const int grid = (x.n + block - 1) / block;
    swap_kernel<<<grid, block>>>(x, y);
    CUDA_ERROR_CHECK(cudaGetLastError());
}

template void swap<real_t>(Vector<real_t>&, Vector<real_t>&);
template void swap<complex_t>(Vector<complex_t>&, Vector<complex_t>&);


__global__ void dot_kernel(const Vector<real_t> x, const Vector<real_t> y, real_t* result) {
    const real_t* __restrict__ xp = x.data;
    const real_t* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	atomicAdd(result, xp[i * x.inc] * yp[i * y.inc]);
    }
}

real_t dot(const Vector<real_t>& x, const Vector<real_t>& y) {
    if (x.n <= 0 || y.n <= 0) return 0;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, dot_kernel);

    const int grid = (x.n + block - 1) / block;
    real_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(real_t)));
    dot_kernel<<<grid, block>>>(x, y, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    real_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    return result;
}

__global__ void dotu_kernel(const Vector<complex_t> x, const Vector<complex_t> y, complex_t* result) {
    const complex_t* __restrict__ xp = x.data;
    const complex_t* __restrict__ yp = y.data;
    real_t* r = reinterpret_cast<real_t*>(result);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	complex_t prod = xp[i * x.inc] * yp[i * y.inc];
	atomicAdd(r, prod.real());
	atomicAdd(r + 1, prod.imag());
    }
}

complex_t dotu(const Vector<complex_t>& x, const Vector<complex_t>& y) {
    if (x.n <= 0 || y.n <= 0) return 0;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, dotu_kernel);

    const int grid = (x.n + block - 1) / block;
    complex_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(complex_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(complex_t)));
    dotu_kernel<<<grid, block>>>(x, y, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    complex_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(complex_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    return result;
}

__global__ void dotc_kernel(const Vector<complex_t> x, const Vector<complex_t> y, complex_t* result) {
    const complex_t* __restrict__ xp = x.data;
    const complex_t* __restrict__ yp = y.data;
    real_t* r = reinterpret_cast<real_t*>(result);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	complex_t prod = conj(xp[i * x.inc]) * yp[i * y.inc];
	atomicAdd(r, prod.real());
	atomicAdd(r + 1, prod.imag());
    }
}

complex_t dotc(const Vector<complex_t>& x, const Vector<complex_t>& y) {
    if (x.n <= 0 || y.n <= 0) return 0;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, dotc_kernel);

    const int grid = (x.n + block - 1) / block;
    complex_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(complex_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(complex_t)));
    dotc_kernel<<<grid, block>>>(x, y, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    complex_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(complex_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    return result;
}

template <typename T>
__global__ void nrm2_kernel(const Vector<T> x, real_t* result) {
    const T* __restrict__ xp = x.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	T v = xp[i * x.inc];
	if constexpr (cuda::std::is_same_v<T, complex_t>) {
	    atomicAdd(result, cuda::std::norm(v));
	} else {
	    atomicAdd(result, v * v);
	}
    }
}

template <typename T>
real_t nrm2(const Vector<T>& x) {
    if (x.n <= 0) return 0;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, nrm2_kernel<T>);

    const int grid = (x.n + block - 1) / block;
    real_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(real_t)));
    nrm2_kernel<<<grid, block>>>(x, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    real_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    return std::sqrt(result);
}

template real_t nrm2<real_t>(const Vector<real_t>&);
template real_t nrm2<complex_t>(const Vector<complex_t>&);


template <typename T>
__global__ void asum_kernel(const Vector<T> x, real_t* result) {
    const T* __restrict__ xp = x.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	T v = xp[i * x.inc];
	if constexpr (cuda::std::is_same_v<T, complex_t>) {
	    atomicAdd(result, cuda::std::abs(v.real()) + cuda::std::abs(v.imag()));
	} else {
	    atomicAdd(result, cuda::std::abs(v));
	}
    }
}

template <typename T>
real_t asum(const Vector<T>& x) {
    if (x.n <= 0) return 0;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, asum_kernel<T>);

    const int grid = (x.n + block - 1) / block;
    real_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(real_t)));
    asum_kernel<<<grid, block>>>(x, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    real_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    return result;
}

template real_t asum<real_t>(const Vector<real_t>&);
template real_t asum<complex_t>(const Vector<complex_t>&);


template <typename T, typename Access>
__device__ T row_dot(const Matrix<T>& A, const Vector<T>& x, long i, long len, Access at)
{
  T sum = 0;
  for (long j = 0; j < len; j++)
      sum += at(A, i, j) * x.data[j * x.inc];
  return sum;
}

template <typename T, typename Access>
__global__ void gemv_kernel(T alpha, const Matrix<T> A, const Vector<T> x, T beta, Vector<T> y, long m, long n, Access at) {
    T* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i+= (long)gridDim.x * blockDim.x) {
	T sum = row_dot(A, x, i, n, at);
	yp[i * y.inc] = alpha * sum + (beta == T(0) ? T(0) : beta * yp[i * y.inc]);
    }
}


template <typename T>
void gemv(const char* trans, T alpha, const Matrix<T>& A, const Vector<T>& x, T beta, Vector<T>& y) {
    if (A.rows <= 0 || A.cols <= 0 || (alpha == T(0) && beta == T(1))) return;

    const bool notrans = (trans[0] == 'N' || trans[0] == 'n');
    const long m = notrans ? A.rows : A.cols;
    const long n = notrans ? A.cols : A.rows;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, (gemv_kernel<T, NoTransAt<T>>));

    const int grid = (m + block - 1) / block;
    if (notrans) {
	gemv_kernel<<<grid, block>>>(alpha, A, x, beta, y, m, n, NoTransAt<T>());
    } else if (trans[0] == 'T' || trans[0] == 't') {
	gemv_kernel<<<grid, block>>>(alpha, A, x, beta, y, m, n, TransAt<T>());
    } else if (trans[0] == 'C' || trans[0] == 'c') {
	gemv_kernel<<<grid, block>>>(alpha, A, x, beta, y, m, n, ConjTransAt<T>());
    }
    CUDA_ERROR_CHECK(cudaGetLastError());
}

template void gemv<real_t>(const char*, real_t, const Matrix<real_t>&, const Vector<real_t>&, real_t, Vector<real_t>&);
template void gemv<complex_t>(const char*, complex_t, const Matrix<complex_t>&, const Vector<complex_t>&, complex_t, Vector<complex_t>&);

    

