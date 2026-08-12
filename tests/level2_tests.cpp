#include "test_utils.hpp"

TEST(Level2, Gemv) {
    std::vector<real_t> h_A(static_cast<size_t>(n) * n);
    random_matrix(h_A.data(), n, n);
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);
    std::vector<real_t> h_y_ref = h_y;

    Matrix A{h_A.data(), n, n, n};
    Vector x{h_x.data(), n, 1};
    Vector y{h_y.data(), n, 1};

    gemv("N", real_t(2), A, x, real_t(3), y);

    ref_gemv("N", n, n, 2, h_A.data(), n, h_x.data(), 1, 3, h_y_ref.data(), 1);

    verify_vector_near(y, h_y_ref, 1e-4);
}

TEST(Level2, GemvTrans) {
    std::vector<real_t> h_A(static_cast<size_t>(n) * n);
    random_matrix(h_A.data(), n, n);
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);
    std::vector<real_t> h_y_ref = h_y;

    Matrix A{h_A.data(), n, n, n};
    Vector x{h_x.data(), n, 1};
    Vector y{h_y.data(), n, 1};

    gemv("T", real_t(1), A, x, real_t(0), y);

    ref_gemv("T", n, n, 1, h_A.data(), n, h_x.data(), 1, 0, h_y_ref.data(), 1);

    verify_vector_near(y, h_y_ref, 1e-4);
}

TEST(Level2, Symv) {
    std::vector<real_t> h_A(static_cast<size_t>(n) * n);
    random_matrix_symmetric(h_A.data(), n, n);
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);
    std::vector<real_t> h_y_ref = h_y;

    Matrix A{h_A.data(), n, n, n};
    Vector x{h_x.data(), n, 1};
    Vector y{h_y.data(), n, 1};

    symv("U", real_t(1), A, x, real_t(0), y);

    ref_symv("U", n, real_t(1), h_A.data(), n, h_x.data(), 1, real_t(0), h_y_ref.data(), 1);

    verify_vector_near(y, h_y_ref, 1e-4);
}
