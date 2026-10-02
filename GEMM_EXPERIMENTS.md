# GEMM experiments

The standalone `gemm_experiment.cu` includes `blas.cu` so it can time internal kernels without the public API's allocation and transfer costs. Compile this file alone; do not also link blas_lib. CSV columns: block_width,k_depth,m,n,k,transpose,median_ms,gflops.

```sh
nvcc -O3 -arch=sm_89 -lineinfo gemm_experiment.cu -o /tmp/gemm-default
/tmp/gemm-default 2048
/tmp/gemm-default 129
nvcc -O3 -arch=sm_89 -lineinfo -DGEMM_TILE=8 -DGEMM_K_TILE=8 gemm_experiment.cu -o /tmp/gemm-small
/tmp/gemm-small 2048
nvcc -O3 -arch=sm_89 -lineinfo -DGEMM_TILE=8 -DGEMM_K_TILE=16 gemm_experiment.cu -o /tmp/gemm-small-k16
/tmp/gemm-small-k16 2048
nvcc -O3 -arch=sm_89 -DDOUBLE_PRECISION gemm_experiment.cu -o /tmp/gemm-double
/tmp/gemm-double 65
```

Target architecture 89 is this machine's RTX 4060 Laptop GPU. Inputs are deterministic signed values. All transpose combinations and beta=0/1 are checked, including NaN-filled C for beta=0 and untouched leading-dimension padding. Shapes other than 2048 are rectangular (size, size+3, size+5). CPU comparisons cover every output up to size 256, sampled outputs above that. This does not validate the public wrapper, alpha=0, empty dimensions, or races; the existing Netlib regression tests cover the public wrapper separately.

Timing: five warmups, nine batches of ten launches, median batch-average CUDA-event duration. Allocation and transfers are excluded. Inputs are reused without cache flushing; clocks are not locked. Repeated launches must run sequentially with other GPU benchmarks idle. Future work: alternating A/B order, spread reporting, clock capture, cuBLAS with specified math mode, profiler counters, sanitizer checks, and additional matrix shapes.

## 2026-09-26

Source baseline c11b4fd; CUDA 13.3.73; compiler -O3 -arch=sm_89 -lineinfo. Raw results are in experiments/2026-09-26. These padded, unprofiled timings differ from the older tightly packed Nsight runs in report_notes.txt.

NN at 2048 cubed: original 16/16 = 3.907 ms; 8/8 = 4.635 ms; 8/16 = 4.733 ms. All transpose variants regressed with either small-block candidate in this run. Do not change the FP32 default. A repeat with the generalized 16/16 code gave 3.965 ms; no speedup is claimed.

The implementation now separates GEMM_K_TILE from GEMM_TILE and derives cooperative loads per thread. FP32 retains 16/16. FP64 defaults to 16/8: the old double-buffered 16/16 layout used 64 KiB static shared memory and failed compilation (48 KiB static limit); depth 8 uses 32 KiB. Standalone padded-edge checks passed in FP32 and FP64.

Next performance experiment: sweep K depth independently of output geometry, then inspect register spills, eligible warps, barriers, and shared-memory transactions. Smaller blocks alone did not explain or close the gap. Do not infer cuBLAS implementation details solely from its kernel name.
