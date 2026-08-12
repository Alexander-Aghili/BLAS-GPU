# BLAS-GPU

A BLAS implementation in CUDA, written from scratch.

## Why

- Learn numerical linear algebra on GPUs
- Understand what cuBLAS does under the hood
- Practice GPU performance work: coalescing, occupancy, tiling

## Scope

- Levels 1-3, real only, following the BLAS spec
- Precision (float/double) set at compile time via `real_t`
- Verified against the Netlib reference BLAS

## Build

```sh
cmake --preset default
cmake --build build
./build/blas
```

Requires CUDA 13+, CMake 3.24+, an NVIDIA GPU.
