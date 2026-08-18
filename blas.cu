#include "blas.hpp"

#include <cmath>

#define BLOCK_SIZE 256

static long span_of(int n, int inc) {
    return n > 0 ? 1 + (long)(n - 1) * (inc < 0 ? -inc : inc) : 0;
}

static long offset_of(int n, int inc) {
    return (inc < 0 && n > 0) ? (long)(n - 1) * -inc : 0;
}

static real_t* device_copy_in(const real_t* host, long count) {
    real_t* dev = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&dev, (count > 0 ? count : 1) * sizeof(real_t)));
    if (count > 0)
	CUDA_ERROR_CHECK(cudaMemcpy(dev, host, count * sizeof(real_t), cudaMemcpyHostToDevice));
    return dev;
}

static Vector stage_vector(const Vector& v, real_t*& slab) {
    const long off = offset_of(v.n, v.inc);
    slab = device_copy_in(v.data - off, span_of(v.n, v.inc));
    return Vector{slab + off, v.n, v.inc};
}

static void unstage_vector(const Vector& v, real_t* slab) {
    const long count = span_of(v.n, v.inc);
    if (count > 0)
	CUDA_ERROR_CHECK(cudaMemcpy(v.data - offset_of(v.n, v.inc), slab, count * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(slab));
}

static Matrix stage_matrix(const Matrix& A, real_t*& slab) {
    slab = device_copy_in(A.data, (long)A.ld * A.cols);
    return Matrix{slab, A.rows, A.cols, A.ld};
}

static void unstage_matrix(const Matrix& A, real_t* slab) {
    const long count = (long)A.ld * A.cols;
    if (count > 0)
	CUDA_ERROR_CHECK(cudaMemcpy(A.data, slab, count * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(slab));
}

static void release(real_t* slab) {
    CUDA_ERROR_CHECK(cudaFree(slab));
}

//Level 1 BLAS

//axpy: y = alpha * x + y
__global__ void axpy_kernel(real_t alpha, Vector x, Vector y) {
    const real_t* __restrict__ xp = x.data;
    real_t* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n;
         i += (long)gridDim.x * blockDim.x) {
        yp[i * y.inc] += alpha * xp[i * x.inc];
    }
}

void axpy(real_t alpha, const Vector& x, Vector& y) {
    if (x.n <= 0 || alpha == real_t(0)) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, axpy_kernel);

    real_t* sx = nullptr;
    real_t* sy = nullptr;
    const Vector dx = stage_vector(x, sx);
    const Vector dy = stage_vector(y, sy);

    const int grid = (x.n + block - 1) / block;
    axpy_kernel<<<grid, block>>>(alpha, dx, dy);
    CUDA_ERROR_CHECK(cudaGetLastError());
    unstage_vector(y, sy);
    release(sx);
}

//scal: y = alpha * y
__global__ void scal_kernel(real_t alpha, Vector y) {
    real_t* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < y.n; i+= (long)gridDim.x * blockDim.x) {
	yp[i * y.inc] *= alpha;
    }
}

void scal(real_t alpha, Vector& y) {
    if (y.n <= 0) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, scal_kernel);

    real_t* sy = nullptr;
    const Vector dy = stage_vector(y, sy);

    const int grid = (y.n + block - 1) / block;
    scal_kernel<<<grid, block>>>(alpha, dy);
    CUDA_ERROR_CHECK(cudaGetLastError());
    unstage_vector(y, sy);
}

__global__ void copy_kernel(const Vector x, Vector y) {
    const real_t* __restrict__ xp = x.data;
    real_t* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	yp[i * y.inc] = xp[i * x.inc];
    }
}

void copy(const Vector& x, Vector& y) {
    if (x.n <= 0 || y.n <= 0) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, copy_kernel);

    real_t* sx = nullptr;
    real_t* sy = nullptr;
    const Vector dx = stage_vector(x, sx);
    const Vector dy = stage_vector(y, sy);

    const int grid = (x.n + block - 1) / block;
    copy_kernel<<<grid, block>>>(dx, dy);
    CUDA_ERROR_CHECK(cudaGetLastError());
    unstage_vector(y, sy);
    release(sx);
}

__global__ void swap_kernel(Vector x, Vector y) {
    real_t* __restrict__ xp = x.data;
    real_t* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
        real_t tmp = xp[i * x.inc];
	xp[i * x.inc] = yp[i * y.inc];
	yp[i * y.inc] = tmp;
    }
}

void swap(Vector& x, Vector& y) {
    if (x.n <= 0 || y.n <= 0) return;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, swap_kernel);

    real_t* sx = nullptr;
    real_t* sy = nullptr;
    const Vector dx = stage_vector(x, sx);
    const Vector dy = stage_vector(y, sy);

    const int grid = (x.n + block - 1) / block;
    swap_kernel<<<grid, block>>>(dx, dy);
    CUDA_ERROR_CHECK(cudaGetLastError());
    unstage_vector(x, sx);
    unstage_vector(y, sy);
}


__global__ void dot_kernel(const Vector x, const Vector y, real_t* result) {
    const real_t* __restrict__ xp = x.data;
    const real_t* __restrict__ yp = y.data;
    __shared__ real_t sdata[BLOCK_SIZE / 32];
    unsigned int warp = threadIdx.x / 32;
    unsigned int lane = threadIdx.x % 32;
    
    real_t sum = 0;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i += (long)gridDim.x * blockDim.x) {
	sum += xp[i * x.inc] * yp[i * y.inc];
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (lane == 0)
	sdata[warp] = sum;
    __syncthreads();

    if (warp == 0) {
	sum = lane < BLOCK_SIZE / 32 ? sdata[lane] : 0;
	sum += __shfl_down_sync(0xffffffff, sum, 16);
	sum += __shfl_down_sync(0xffffffff, sum, 8);
	sum += __shfl_down_sync(0xffffffff, sum, 4);
	sum += __shfl_down_sync(0xffffffff, sum, 2);
	sum += __shfl_down_sync(0xffffffff, sum, 1);
	if (lane == 0)
	    atomicAdd(result, sum);
    }
}

real_t dot(const Vector& x, const Vector& y) {
    if (x.n <= 0 || y.n <= 0) return 0;

    real_t* sx = nullptr;
    real_t* sy = nullptr;
    const Vector dx = stage_vector(x, sx);
    const Vector dy = stage_vector(y, sy);

    const int grid = (x.n + 2L * BLOCK_SIZE - 1) / (2L * BLOCK_SIZE);
    real_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(real_t)));
    dot_kernel<<<grid, BLOCK_SIZE>>>(dx, dy, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    real_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    release(sx);
    release(sy);
    return result;
}

__global__ void nrm2_kernel(const Vector x, real_t* result) {
    const real_t* __restrict__ xp = x.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	real_t v = xp[i * x.inc];
	atomicAdd(result, v * v);
    }
}

real_t nrm2(const Vector& x) {
    if (x.n <= 0) return 0;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, nrm2_kernel);

    real_t* sx = nullptr;
    const Vector dx = stage_vector(x, sx);

    const int grid = (x.n + block - 1) / block;
    real_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(real_t)));
    nrm2_kernel<<<grid, block>>>(dx, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    real_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    release(sx);
    return std::sqrt(result);
}


__global__ void asum_kernel(const Vector x, real_t* result) {
    const real_t* __restrict__ xp = x.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < x.n; i+= (long)gridDim.x * blockDim.x) {
	real_t v = xp[i * x.inc];
	atomicAdd(result, fabs(v));
    }
}

real_t asum(const Vector& x) {
    if (x.n <= 0) return 0;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, asum_kernel);

    real_t* sx = nullptr;
    const Vector dx = stage_vector(x, sx);

    const int grid = (x.n + block - 1) / block;
    real_t* d_result = nullptr;
    CUDA_ERROR_CHECK(cudaMalloc(&d_result, sizeof(real_t)));
    CUDA_ERROR_CHECK(cudaMemset(d_result, 0, sizeof(real_t)));
    asum_kernel<<<grid, block>>>(dx, d_result);
    CUDA_ERROR_CHECK(cudaGetLastError());
    real_t result = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&result, d_result, sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    release(sx);
    return result;
}


template <typename Access>
__device__ real_t row_dot(const Matrix& A, const Vector& x, long i, long len, Access at)
{
  real_t sum = 0;
  for (long j = 0; j < len; j++)
      sum += at(A, i, j) * x.data[j * x.inc];
  return sum;
}

template <typename Access>
__global__ void gemv_kernel(real_t alpha, const Matrix A, const Vector x, real_t beta, Vector y, long m, long n, Access at) {
    real_t* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i+= (long)gridDim.x * blockDim.x) {
	real_t sum = row_dot(A, x, i, n, at);
	yp[i * y.inc] = alpha * sum + (beta == real_t(0) ? real_t(0) : beta * yp[i * y.inc]);
    }
}


void gemv(const char* trans, real_t alpha, const Matrix& A, const Vector& x, real_t beta, Vector& y) {
    if (A.rows <= 0 || A.cols <= 0 || (alpha == real_t(0) && beta == real_t(1))) return;

    const bool notrans = (trans[0] == 'N' || trans[0] == 'n');
    const long m = notrans ? A.rows : A.cols;
    const long n = notrans ? A.cols : A.rows;


    real_t* sa = nullptr;
    real_t* sx = nullptr;
    real_t* sy = nullptr;
    const Matrix dA = stage_matrix(A, sa);
    const Vector dx = stage_vector(x, sx);
    const Vector dy = stage_vector(y, sy);

    const int grid = (m + 2L * BLOCK_SIZE - 1) / (2L * BLOCK_SIZE);
    if (notrans) {
	gemv_kernel<<<grid, BLOCK_SIZE>>>(alpha, dA, dx, beta, dy, m, n, NoTransAt());
    } else {
	gemv_kernel<<<grid, BLOCK_SIZE>>>(alpha, dA, dx, beta, dy, m, n, TransAt());
    }
    CUDA_ERROR_CHECK(cudaGetLastError());
    unstage_vector(y, sy);
    release(sx);
    release(sa);
}

template <typename Access>
__global__ void symv_kernel(real_t alpha, const Matrix A, const Vector x, real_t beta, const Vector y, Access at) {
    real_t* __restrict__ yp = y.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < A.rows; i+= (long)gridDim.x * blockDim.x) {
	real_t sum = row_dot(A, x, i, A.cols, at);
	yp[i * y.inc] = alpha * sum + (beta == real_t(0) ? real_t(0) : beta * yp[i * y.inc]);
    }
}

void symv(const char* uplo, real_t alpha, const Matrix& A, const Vector& x, real_t beta, const Vector& y) {
    if (A.rows <= 0 || A.cols <= 0 || (alpha == real_t(0) && beta == real_t(1))) return;

    const bool upper = (uplo[0] == 'U' || uplo[0] == 'u');
    const long m = upper ? A.rows : A.cols;
    const long n = upper ? A.cols : A.rows;

    int min_grid = 0, block = 0;
    GET_MAX_POTENTIAL_BLOCKS_SIZE(min_grid, block, (symv_kernel<UpperAt>));

    real_t* sa = nullptr;
    real_t* sx = nullptr;
    real_t* sy = nullptr;
    const Matrix dA = stage_matrix(A, sa);
    const Vector dx = stage_vector(x, sx);
    const Vector dy = stage_vector(y, sy);

    const int grid = (m + block - 1) / block;
    if (upper) {
	symv_kernel<<<grid, block>>>(alpha, dA, dx, beta, dy, UpperAt());
    } else {
	symv_kernel<<<grid, block>>>(alpha, dA, dx, beta, dy, LowerAt());
    }

    CUDA_ERROR_CHECK(cudaGetLastError());
    unstage_vector(y, sy);
    release(sx);
    release(sa);
}

template <typename AccessA, typename AccessB>
__global__ void gemm_kernel(real_t alpha, const Matrix A, const Matrix B, real_t beta, Matrix C, long m, long n, long k, AccessA at_a, AccessB at_b) {
    real_t* __restrict__ cp = C.data;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x) {
	for (long j = blockIdx.y * (long)blockDim.y + threadIdx.y; j < n; j += (long)gridDim.y * blockDim.y) {
	    real_t sum = 0;
	    for (long l = 0; l < k; l++) {
		sum += at_a(A, i, l) * at_b(B, l, j);
	    }
	    cp[i + j * C.ld] = alpha * sum + (beta == real_t(0) ? real_t(0) : beta * cp[i + j * C.ld]);
	}
    }
}

template <typename AccessA>
static void gemm_launch(const char* transb, real_t alpha, const Matrix& A, const Matrix& B, real_t beta, Matrix& C, long m, long n, long k, dim3 grid, dim3 block, AccessA at_a) {
    if (transb[0] == 'N' || transb[0] == 'n') {
	gemm_kernel<<<grid, block>>>(alpha, A, B, beta, C, m, n, k, at_a, NoTransAt());
    } else {
	gemm_kernel<<<grid, block>>>(alpha, A, B, beta, C, m, n, k, at_a, TransAt());
    }
}

void gemm(const char* transa, const char* transb, real_t alpha, const Matrix& A, const Matrix& B, real_t beta, Matrix& C) {
    const bool na = (transa[0] == 'N' || transa[0] == 'n');
    const long m = na ? A.rows : A.cols;
    const long k = na ? A.cols : A.rows;
    const long n = (transb[0] == 'N' || transb[0] == 'n') ? B.cols : B.rows;

    if (m <= 0 || n <= 0 || (alpha == real_t(0) && beta == real_t(1))) return;

    const dim3 block(16, 16);
    long gx = (m + block.x - 1) / block.x;
    long gy = (n + block.y - 1) / block.y;
    //temporary cap on grid size
    if (gx > 65535) gx = 65535;
    if (gy > 65535) gy = 65535;
    const dim3 grid((unsigned)gx, (unsigned)gy);

    real_t* sa = nullptr;
    real_t* sb = nullptr;
    real_t* sc = nullptr;
    const Matrix dA = stage_matrix(A, sa);
    const Matrix dB = stage_matrix(B, sb);
    Matrix dC = stage_matrix(C, sc);

    if (na) {
	gemm_launch(transb, alpha, dA, dB, beta, dC, m, n, k, grid, block, NoTransAt());
    } else {
	gemm_launch(transb, alpha, dA, dB, beta, dC, m, n, k, grid, block, TransAt());
    }
    CUDA_ERROR_CHECK(cudaGetLastError());
    unstage_matrix(C, sc);
    release(sb);
    release(sa);
}
