#include "blas.cuh"

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

void copy(const Vector<T>& x, Vector<T>& y) {
    if (x.n <= 0 || y.n <= 0) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, copy_kernel<T>);

    const int grid = (x.n + block - 1) / block;
    copy_kernel<<<grid, block>>>(x, y);
    CUDA_ERROR_CHECK(cudaGetLastError());
}

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

void swap(Vector<T>& x, Vector<T>& y) {
    if (x.n <= 0 || y.n <= 0) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, swap_kernel<real_t>);

    const int grid = (x.n + block - 1) / block;
    swap_kernel<<<grid, block>>>(x, y);
    CUDA_ERROR_CHECK(cudaGetLastError());
}

template void swap<real_t>(Vector<real_t>&, Vector<real_t>&);
template void swap<complex_t>(Vector<complex_t>&, Vector<complex_t>&);


