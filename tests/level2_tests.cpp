#include "test_utils.hpp"

TEST(Level2, Gemv) {
    std::vector<real_t> h_A(static_cast<size_t>(n) * n);
    random_matrix(h_A.data(), n, n);
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);

    real_t* d_A = nullptr;
    real_t* d_x = nullptr;
    real_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_A, h_A.size() * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);

    Matrix<real_t> A{d_A, n, n, n};
    Vector<real_t> x{d_x, n, 1};
    Vector<real_t> y{d_y, n, 1};

    gemv("N", real_t(2), A, x, real_t(3), y);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<real_t> h_y_ref = h_y;
    ref_gemv("N", n, n, 2, h_A.data(), n, h_x.data(), 1, 3, h_y_ref.data(), 1);

    verify_vector_near(y, h_y_ref, 1e-4);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level2, GemvTrans) {
    std::vector<real_t> h_A(static_cast<size_t>(n) * n);
    random_matrix(h_A.data(), n, n);
    std::vector<real_t> h_x = random_vector(n);
    std::vector<real_t> h_y = random_vector(n);

    real_t* d_A = nullptr;
    real_t* d_x = nullptr;
    real_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_A, h_A.size() * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);

    Matrix<real_t> A{d_A, n, n, n};
    Vector<real_t> x{d_x, n, 1};
    Vector<real_t> y{d_y, n, 1};

    gemv("T", real_t(1), A, x, real_t(0), y);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<real_t> h_y_ref = h_y;
    ref_gemv("T", n, n, 1, h_A.data(), n, h_x.data(), 1, 0, h_y_ref.data(), 1);

    verify_vector_near(y, h_y_ref, 1e-4);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level2, Hemv) {
    std::vector<complex_t> h_A(static_cast<size_t>(n) * n);
    random_matrix_hermitian<complex_t>(h_A.data(), n, n);
    std::vector<complex_t> h_x = random_vector<complex_t>(n);
    std::vector<complex_t> h_y = random_vector<complex_t>(n);

    complex_t* d_A = nullptr;
    complex_t* d_x = nullptr;
    complex_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_A, h_A.size() * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(complex_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(complex_t), cudaMemcpyHostToDevice), cudaSuccess);

    Matrix<complex_t> A{d_A, n, n, n};
    Vector<complex_t> x{d_x, n, 1};
    Vector<complex_t> y{d_y, n, 1};

    hemv("U", complex_t(1), A, x, complex_t(0), y);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<complex_t> h_y_ref = h_y;
    ref_hemv("U", n, complex_t(1), h_A.data(), n, h_x.data(), 1, complex_t(0), h_y_ref.data(), 1);

    verify_vector_near(y, h_y_ref, 1e-4);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

TEST(Level2, Symv) {
    std::vector<real_t> h_A(static_cast<size_t>(n) * n);
    random_matrix_hermitian<real_t>(h_A.data(), n, n);
    std::vector<real_t> h_x = random_vector<real_t>(n);
    std::vector<real_t> h_y = random_vector<real_t>(n);

    real_t* d_A = nullptr;
    real_t* d_x = nullptr;
    real_t* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_A, h_A.size() * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, n * sizeof(real_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, h_y.data(), n * sizeof(real_t), cudaMemcpyHostToDevice), cudaSuccess);

    Matrix<real_t> A{d_A, n, n, n};
    Vector<real_t> x{d_x, n, 1};
    Vector<real_t> y{d_y, n, 1};

    symv("U", real_t(1), A, x, real_t(0), y);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<real_t> h_y_ref = h_y;
    ref_symv("U", n, real_t(1), h_A.data(), n, h_x.data(), 1, real_t(0), h_y_ref.data(), 1);

    verify_vector_near(y, h_y_ref, 1e-4);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}
