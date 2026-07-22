#include "test_utils.hpp"

TEST(Level1, Copy) {
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y(n, 0);

    Vector<real_t> x{h_x.data(), n, 1};
    Vector<real_t> y{h_y.data(), n, 1};

    copy(x, y);

    std::vector<real_t> h_y_ref(n, 0);
    ref_copy(n, h_x.data(), 1, h_y_ref.data(), 1);

    verify_vector(y, h_y_ref);
}

TEST(Level1, Swap) {
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);

    std::vector<real_t> h_x_ref = h_x;
    std::vector<real_t> h_y_ref = h_y;

    Vector<real_t> x{h_x.data(), n, 1};
    Vector<real_t> y{h_y.data(), n, 1};

    swap(x, y);

    ref_swap(n, h_x_ref.data(), 1, h_y_ref.data(), 1);

    verify_vector(x, h_x_ref);
    verify_vector(y, h_y_ref);
}

TEST(Level1, Dot) {
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);

    Vector<real_t> x{h_x.data(), n, 1};
    Vector<real_t> y{h_y.data(), n, 1};

    real_t result = dot(x, y);

    real_t ref = ref_dot(n, h_x.data(), 1, h_y.data(), 1);
    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-4);
}

TEST(Level1, Dotu) {
    std::vector<complex_t> h_x = random_vector<complex_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    Vector<complex_t> x{h_x.data(), n, 1};
    Vector<complex_t> y{h_y.data(), n, 1};

    complex_t result = dotu(x, y);

    complex_t ref = ref_dotu(n, h_x.data(), 1, h_y.data(), 1);
    real_t tol = std::hypot(ref.real(), ref.imag()) * 1e-4;
    EXPECT_NEAR(result.real(), ref.real(), tol);
    EXPECT_NEAR(result.imag(), ref.imag(), tol);
}

TEST(Level1, Dotc) {
    std::vector<complex_t> h_x = random_vector<complex_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    Vector<complex_t> x{h_x.data(), n, 1};
    Vector<complex_t> y{h_y.data(), n, 1};

    complex_t result = dotc(x, y);

    complex_t ref = ref_dotc(n, h_x.data(), 1, h_y.data(), 1);
    real_t tol = std::hypot(ref.real(), ref.imag()) * 1e-4;
    EXPECT_NEAR(result.real(), ref.real(), tol);
    EXPECT_NEAR(result.imag(), ref.imag(), tol);
}

TEST(Level1, Nrm2) {
    std::vector<real_t> h_x = random_vector<real_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    Vector<real_t> x{h_x.data(), n, 1};
    Vector<complex_t> y{h_y.data(), n, 1};

    real_t result = nrm2(x);
    real_t resultc = nrm2(y);
    real_t ref = ref_nrm2(n, h_x.data(), 1);
    real_t refc = ref_nrm2(n, h_y.data(), 1);
    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-4);
    EXPECT_NEAR(resultc, refc, std::abs(refc) * 1e-4);
}

TEST(Level1, Asum) {
    std::vector<real_t> h_x = random_vector<real_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    Vector<real_t> x{h_x.data(), n, 1};
    Vector<complex_t> y{h_y.data(), n, 1};

    real_t result = asum(x);
    real_t resultc = asum(y);
    real_t ref = ref_asum(n, h_x.data(), 1);
    real_t refc = ref_asum(n, h_y.data(), 1);
    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-4);
    EXPECT_NEAR(resultc, refc, std::abs(refc) * 1e-4);
}
