#pragma once
#include "types.hpp"

// Launch-only entry points (device pointers, inc = 1, ld = n, alpha = 1, beta = 0).
#define DECL_DOT(tag) void dot_##tag(const real_t* x, const real_t* y, int n, real_t* result);
#define DECL_GEMV(tag) void gemv_##tag(const real_t* A, const real_t* x, real_t* y, int n);
#define DECL_GEMM(tag) void gemm_##tag(const real_t* A, const real_t* B, real_t* C, int n);

DECL_DOT(4da826e)
DECL_DOT(3affd9b)
DECL_DOT(47a65d8)
DECL_DOT(7099f40)
DECL_DOT(98cc847)
DECL_GEMV(051d7c3)
DECL_GEMV(15e6bc6)
DECL_GEMM(15e6bc6)
DECL_GEMM(89b175b)
DECL_GEMM(a44068e)
DECL_GEMM(6d1f43a)
DECL_GEMM(c11b4fd)
DECL_GEMM(worktree)
