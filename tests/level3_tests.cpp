#include "test_utils.hpp"

template <typename T>
static void run_gemm(const char* transa, const char* transb, T alpha, T beta) {
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
    std::vector<T> h_C_ref = h_C;

    Matrix<T> A{h_A.data(), a_rows, a_cols, a_rows};
    Matrix<T> B{h_B.data(), b_rows, b_cols, b_rows};
    Matrix<T> C{h_C.data(), m, cols, m};

    gemm(transa, transb, alpha, A, B, beta, C);

    ref_gemm(transa, transb, m, cols, k, alpha, h_A.data(), a_rows, h_B.data(), b_rows, beta, h_C_ref.data(), m);

    Vector<T> c_flat{h_C.data(), m * cols, 1};
    verify_vector_near(c_flat, h_C_ref, 1e-4);
}

TEST(Level3, GemmNN) { run_gemm<real_t>("N", "N", real_t(2), real_t(3)); }

TEST(Level3, GemmTN) { run_gemm<real_t>("T", "N", real_t(1), real_t(0)); }

TEST(Level3, GemmNT) { run_gemm<real_t>("N", "T", real_t(2), real_t(0)); }

TEST(Level3, GemmTT) { run_gemm<real_t>("T", "T", real_t(1), real_t(2)); }

TEST(Level3, GemmComplexNN) { run_gemm<complex_t>("N", "N", complex_t(2, 1), complex_t(1, 1)); }

TEST(Level3, GemmComplexConjTransA) { run_gemm<complex_t>("C", "N", complex_t(1, 2), complex_t(0)); }

TEST(Level3, GemmComplexConjTransB) { run_gemm<complex_t>("N", "C", complex_t(1), complex_t(2, -1)); }
