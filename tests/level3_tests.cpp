#include "test_utils.hpp"

static void run_gemm(const char* transa, const char* transb, real_t alpha, real_t beta) {
    const int m = 384;
    const int cols = 257;
    const int k = 129;
    const bool na = (transa[0] == 'N');
    const bool nb = (transb[0] == 'N');
    const int a_rows = na ? m : k;
    const int a_cols = na ? k : m;
    const int b_rows = nb ? k : cols;
    const int b_cols = nb ? cols : k;

    std::vector<real_t> h_A(static_cast<size_t>(a_rows) * a_cols);
    std::vector<real_t> h_B(static_cast<size_t>(b_rows) * b_cols);
    random_matrix(h_A.data(), a_rows, a_cols);
    random_matrix(h_B.data(), b_rows, b_cols);
    std::vector<real_t> h_C = random_vector(m * cols);
    std::vector<real_t> h_C_ref = h_C;

    Matrix A{h_A.data(), a_rows, a_cols, a_rows};
    Matrix B{h_B.data(), b_rows, b_cols, b_rows};
    Matrix C{h_C.data(), m, cols, m};

    gemm(transa, transb, alpha, A, B, beta, C);

    ref_gemm(transa, transb, m, cols, k, alpha, h_A.data(), a_rows, h_B.data(), b_rows, beta, h_C_ref.data(), m);

    Vector c_flat{h_C.data(), m * cols, 1};
    verify_vector_near(c_flat, h_C_ref, 1e-4);
}

TEST(Level3, GemmNN) { run_gemm("N", "N", real_t(2), real_t(3)); }

TEST(Level3, GemmTN) { run_gemm("T", "N", real_t(1), real_t(0)); }

TEST(Level3, GemmNT) { run_gemm("N", "T", real_t(2), real_t(0)); }

TEST(Level3, GemmTT) { run_gemm("T", "T", real_t(1), real_t(2)); }
