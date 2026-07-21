#include "test_utils.hpp"

namespace {

template <typename T>
void run_gemm(const char* transa, const char* transb, T alpha, T beta) {
    const int m = 384;
    const int cols = 257;
    const int k = 129;
    const bool na = (transa[0] == 'N');
    const bool nb = (transb[0] == 'N');
    const int a_rows = na ? m : k;
    const int a_cols = na ? k : m;
    const int b_rows = nb ? k : cols;
    const int b_cols = nb ? cols : k;

    std::vector<T> h_A(static_cast<size_t>(a_rows) * a_cols);
    std::vector<T> h_B(static_cast<size_t>(b_rows) * b_cols);
    random_matrix(h_A.data(), a_rows, a_cols);
    random_matrix(h_B.data(), b_rows, b_cols);
    std::vector<T> h_C = random_vector<T>(m * cols);

    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;
    ASSERT_EQ(cudaMalloc(&d_A, h_A.size() * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, h_B.size() * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, h_C.size() * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);

    Matrix<T> A{d_A, a_rows, a_cols, a_rows};
    Matrix<T> B{d_B, b_rows, b_cols, b_rows};
    Matrix<T> C{d_C, m, cols, m};

    gemm(transa, transb, alpha, A, B, beta, C);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<T> h_C_ref = h_C;
    ref_gemm(transa, transb, m, cols, k, alpha, h_A.data(), a_rows, h_B.data(), b_rows, beta, h_C_ref.data(), m);

    Vector<T> c_flat{d_C, m * cols, 1};
    verify_vector_near(c_flat, h_C_ref, 1e-4);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}

TEST(Level3, GemmNN) { run_gemm<real_t>("N", "N", real_t(2), real_t(3)); }

TEST(Level3, GemmTN) { run_gemm<real_t>("T", "N", real_t(1), real_t(0)); }

TEST(Level3, GemmNT) { run_gemm<real_t>("N", "T", real_t(2), real_t(0)); }

TEST(Level3, GemmTT) { run_gemm<real_t>("T", "T", real_t(1), real_t(2)); }

TEST(Level3, GemmComplexNN) { run_gemm<complex_t>("N", "N", complex_t(2, 1), complex_t(1, 1)); }

TEST(Level3, GemmComplexConjTransA) { run_gemm<complex_t>("C", "N", complex_t(1, 2), complex_t(0)); }

TEST(Level3, GemmComplexConjTransB) { run_gemm<complex_t>("N", "C", complex_t(1), complex_t(2, -1)); }
