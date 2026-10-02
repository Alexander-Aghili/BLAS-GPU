# Kernel size sweep

Kernel-only timings of every DOT, GEMV and GEMM optimization step from the blog
article, plus cuBLAS, across input sizes. FP32, column-major, `inc = 1`,
`ld = n`, `alpha = 1`, `beta = 0`, device-resident data.

```sh
./build.sh      # extracts each revision's blas.cu with git show, builds ./sweep
./run.sh 3      # one discarded warm-up pass, then 3 passes -> raw.csv -> results.csv
./sweep 1 anchors   # quick check at the article's sizes only
```

## Environment of the recorded run

| | |
| --- | --- |
| Repository HEAD | `c11b4fdd9f0f295dd3f53b0750ca8a106476a126` (working tree `blas.cu` has the uncommitted `GEMM_K_TILE`/`GEMM_LOADS` refactor, measured separately as `worktree`) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (24 SMs, 32 MiB L2, 8 GB) |
| Driver | 595.84 (reports CUDA 13.2) |
| Toolkit | nvcc 13.3.73, cuBLAS 13.6.0.2, `-O3 -arch=native` (sm_89) |
| OS | Linux 6.17.0-1032-oem, on AC power, platform profile `performance` |
| Date | 2026-10-01, 11:50 to 12:00 EDT (`started_at.txt`) |

## What is measured

Each revision's `blas.cu` is compiled in its own namespace (`rev_tu.cu`), and
a launcher reproduces that revision's own launch geometry:

| routine | impl | rev | launch |
| --- | --- | --- | --- |
| dot | atomic | 4da826e | occupancy-API block size, one element per thread |
| dot | block_reduction | 3affd9b | 256 threads, grid = n/256 (interleaved/reindexed tree) |
| dot | sequential_addressing | 47a65d8 | 256 threads, grid = n/256 |
| dot | two_per_thread | 7099f40 | 256 threads, grid = n/512 |
| dot | warp_shuffle | 98cc847 | 256 threads, grid = n/512 (kernel identical to HEAD) |
| gemv | eight_block_baseline | 051d7c3 | 256 threads, grid = m/512 |
| gemv | row_column_split | 15e6bc6 | 32 x 16 threads, grid = m/32 (kernel identical to HEAD) |
| gemm | naive | 15e6bc6 | 16 x 16 threads, one output each |
| gemm | shared_tiles | 89b175b | 16 x 16 threads, 16 x 16 shared tiles |
| gemm | register_4x4 | a44068e | 16 x 16 threads, 64 x 64 block tile |
| gemm | register_8x8 | 6d1f43a | 16 x 16 threads, 128 x 128 block tile |
| gemm | double_buffer | c11b4fd | as above, double-buffered shared tiles (HEAD) |
| gemm | double_buffer_worktree | worktree | uncommitted refactor; FP32-equivalent to c11b4fd |

cuBLAS (`cublasSdot` with device pointer mode, `cublasSgemv`, `cublasSgemm`)
is timed the same way and is the correctness reference. The intermediate GEMV
revision `cca54d1` (shared-memory x tile) and the launch-sweep trials are not
included.

Timing: at each size every implementation is warmed up (3 launches), then all
implementations are timed round-robin, one repetition each in turn, so clock
changes hit all of them alike. Each repetition flushes L2 by reading a 96 MiB
buffer (clean lines only), resets the DOT accumulator, and times the launch with
CUDA events. Repetitions per implementation: enough for about 250 ms, minimum
20, maximum 2000 (1000 for GEMM). `results.csv` reports the median across passes
of each pass's median, the minimum over all repetitions, total repetitions,
GFLOP/s and useful GB/s from the median (article Table 1 with `beta = 0`:
DOT `8n` bytes, GEMV `4(n^2 + 2n)`, GEMM `12N^2`; FLOPs `2n`, `2n^2`, `2N^3`),
and the worst relative error against cuBLAS over the passes (`|d - d_ref|/|d_ref|`
for DOT, relative 2-norm for GEMV and GEMM). Inputs are uniform on [0, 1).

## Caveats

* Clocks are not locked (locking needs root). In the recorded run (1 s samples
  in `clocks_during.csv`) the GPU sat at its 40 W SW power cap, SM clocks ranged
  about 1.0 to 2.2 GHz (mostly 1.5 to 2.1), and the memory clock was 8001 MHz in
  59% of samples, 7001 in 23% and 5501 in 18%. Large memory-bound medians
  therefore sit up to about 15 to 30% above the minima; `time_us_min`
  approximates full memory clock. Differences of a few percent between
  implementations (e.g. `double_buffer` vs `double_buffer_worktree` at 4096)
  are within this noise.
* Nsight Compute, used for the article's captures, locks clocks to base and
  replays kernels; event timings at boost SM clocks differ from those captures,
  most visibly for compute-bound GEMM.
* CUDA event resolution is about 0.5 to 1 µs, so the smallest sizes (a few µs)
  are quantized and dominated by launch latency.
* The atomic DOT accumulates 2^k products into one FP32 value in arbitrary order,
  so its error grows with n (about 2% at 2^24, 22% at 2^26). That is a
  property of the baseline, not of the harness.
* The GEMM kernels all accumulate along k in the same order, so they agree
  bit-for-bit with each other; at N >= 2048 they also match cuBLAS exactly
  (relative error 0), which suggests cuBLAS uses the same order there.
