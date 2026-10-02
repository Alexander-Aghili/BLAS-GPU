#!/usr/bin/env bash
# Build the size-sweep driver. Each historical blas.cu is taken from git (or the
# working tree for "worktree") and compiled in its own namespace via rev_tu.cu.
set -euo pipefail
cd "$(dirname "$0")"
REPO=../..
mkdir -p src obj
for rev in 4da826e 3affd9b 47a65d8 7099f40 98cc847 051d7c3 15e6bc6 89b175b a44068e 6d1f43a c11b4fd; do
    git -C "$REPO" show "$rev:blas.cu" > "src/$rev.cu"
done
cp "$REPO/blas.cu" src/worktree.cu

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
FLAGS=(-O3 -std=c++17 -arch=native -lineinfo -I"$REPO" -I.)

tu() {  # tag, extra defines...
    local tag=$1; shift
    "$NVCC" "${FLAGS[@]}" -DREV_TAG="$tag" -DREV_SRC="\"src/$tag.cu\"" "$@" -c rev_tu.cu -o "obj/$tag.o" &
}
tu 4da826e -DWANT_DOT -DREV_DOT_CFG=0
tu 3affd9b -DWANT_DOT -DREV_DOT_CFG=1
tu 47a65d8 -DWANT_DOT -DREV_DOT_CFG=1
tu 7099f40 -DWANT_DOT -DREV_DOT_CFG=2
tu 98cc847 -DWANT_DOT -DREV_DOT_CFG=2
tu 051d7c3 -DWANT_GEMV -DREV_GEMV_CFG=0
tu 15e6bc6 -DWANT_GEMV -DREV_GEMV_CFG=1 -DWANT_GEMM -DREV_GEMM_CFG=0
tu 89b175b -DWANT_GEMM -DREV_GEMM_CFG=0
tu a44068e -DWANT_GEMM -DREV_GEMM_CFG=1
tu 6d1f43a -DWANT_GEMM -DREV_GEMM_CFG=1
tu c11b4fd -DWANT_GEMM -DREV_GEMM_CFG=1
tu worktree -DWANT_GEMM -DREV_GEMM_CFG=1
wait
"$NVCC" "${FLAGS[@]}" sweep.cu obj/*.o -lcublas -o sweep
echo "built $(pwd)/sweep"
