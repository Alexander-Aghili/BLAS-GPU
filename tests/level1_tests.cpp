#include "test_utils.hpp"

constexpr int n = 4096;

TEST(Level1, Copy) {
    std::vector<real_t> h_x = random_vector(n);

    real_t* d_x = nullptr;
    real_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_y, 0, n * sizeof(real_t)), cudaSuccess);

    Vector<real_t> x{d_x, n, 1};
    Vector<real_t> y{d_y, n, 1};

    copy(x, y);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<real_t> h_y_ref(n, 0);
    ref_copy(n, h_x.data(), 1, h_y_ref.data(), 1);

    verify_vector(y, h_y_ref);

    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level1, Swap) {
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);

    real_t* d_x = nullptr;
    real_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);

    Vector<real_t> x{d_x, n, 1};
    Vector<real_t> y{d_y, n, 1};

    swap(x, y);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<real_t> h_x_ref = h_x;
    std::vector<real_t> h_y_ref = h_y;
    ref_swap(n, h_x_ref.data(), 1, h_y_ref.data(), 1);

    verify_vector(x, h_x_ref);
    verify_vector(y, h_y_ref);

    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level1, Dot) {
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);

    real_t* d_x = nullptr;
    real_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);

    Vector<real_t> x{d_x, n, 1};
    Vector<real_t> y{d_y, n, 1};

    real_t result = dot(x, y);

    real_t ref = ref_dot(n, h_x.data(), 1, h_y.data(), 1);
    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-4);

    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level1, Dotu) {
    std::vector<complex_t> h_x = random_vector<complex_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    complex_t* d_x = nullptr;
    complex_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);

    Vector<complex_t> x{d_x, n, 1};
    Vector<complex_t> y{d_y, n, 1};

    complex_t result = dotu(x, y);

    complex_t ref = ref_dotu(n, h_x.data(), 1, h_y.data(), 1);
    real_t tol = std::hypot(ref.real(), ref.imag()) * 1e-4;
    EXPECT_NEAR(result.real(), ref.real(), tol);
    EXPECT_NEAR(result.imag(), ref.imag(), tol);

    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level1, Dotc) {
    std::vector<complex_t> h_x = random_vector<complex_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    complex_t* d_x = nullptr;
    complex_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);

    Vector<complex_t> x{d_x, n, 1};
    Vector<complex_t> y{d_y, n, 1};

    complex_t result = dotc(x, y);

    complex_t ref = ref_dotc(n, h_x.data(), 1, h_y.data(), 1);
    real_t tol = std::hypot(ref.real(), ref.imag()) * 1e-4;
    EXPECT_NEAR(result.real(), ref.real(), tol);
    EXPECT_NEAR(result.imag(), ref.imag(), tol);

    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level1, Nrm2) {
    std::vector<real_t> h_x = random_vector<real_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    real_t* d_x = nullptr;
    complex_t* d_y = nullptr;

    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);

    Vector<real_t> x{d_x, n, 1};
    Vector<complex_t> y{d_y, n, 1};

    real_t result = nrm2(x);
    real_t resultc = nrm2(y);
    real_t ref = ref_nrm2(n, h_x.data(), 1);
    real_t refc = ref_nrm2(n, h_y.data(), 1);
    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-4);
    EXPECT_NEAR(resultc, refc, std::abs(refc) * 1e-4);

    cudaFree(d_x);
    cudaFree(d_y);
}
