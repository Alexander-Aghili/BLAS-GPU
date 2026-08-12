# TODO

## Level 1
- [x] axpy
- [x] scal
- [x] copy, swap
- [x] dot (atomicAdd first pass)
- [x] nrm2 (atomicAdd sum of squares, sqrt on host)
- [x] asum
- [ ] reduction skeleton (block reduction + one atomic per block) — replace per-element atomicAdd in dot/nrm2/asum
- [ ] iamax (reduction with index tracking)

## Level 2
- [x] gemv (N/T; 'C' handled as 'T', correct for real data)
- [x] symv
- [ ] trmv

## Level 3
- [x] gemm (naive first pass)
- [ ] gemmtr

## Optimizations to look into
- Tiled gemm: shared-memory blocking, register tiling
- Vectorized loads (float4/double2) fast path when inc == 1
- Warp shuffle reductions (`__shfl_down_sync`) vs shared memory
- Block size tuning per kernel (occupancy API vs profiled constants)
- Tensor cores for gemm (WMMA)
- Streams: async launches, overlapping transfers with compute

## Tests
Each test targets a specific failure mode:

- **Reference check** (vs Netlib reference BLAS, linked in tests/CMakeLists.txt) — wrong math, wrong op variant
- **Strided views** (`inc` = 2, 3; matrix row/diagonal as vector) — kernels ignoring `inc`
- **Mixed strides** (x.inc != y.inc in one call) — shared loop stride bugs
- **Submatrix views** (`ld` > rows) — indexing that assumes ld == rows
- **Edge sizes** (n = 0, 1; non-multiple of block size) — quick returns, bounds off-by-one
- **Quick returns** (alpha = 0, beta = 0/1) — spec says skip reads, not just skip math (beta=0 must not propagate NaN from C)
- **trans variants** ('N', 'T', 'C' each) — untested transpose paths
- **uplo/diag** — other triangle must be untouched; unit diag must not read A(i,i)
- **Reduction races** (large n, repeated runs bit-identical) — atomics/sync bugs
- **nrm2 overflow** (huge/tiny values) — naive sum of squares over/underflow
- **Both precisions** (float and DOUBLE_PRECISION builds) — real_t assumptions
