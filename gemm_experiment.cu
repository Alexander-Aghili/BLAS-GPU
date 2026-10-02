#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <limits>
#include "blas.cu"

// Standalone kernel experiment: allocation/transfers are outside event timing.
int main(int argc, char** argv) {
    int size = argc > 1 ? std::atoi(argv[1]) : 2048;
    if (size < 1) return 2;
    for (int ta = 0; ta < 2; ++ta) for (int tb = 0; tb < 2; ++tb) {
        int m = size, n = size == 2048 ? size : size + 3, k = size == 2048 ? size : size + 5;
        int ar = ta ? k : m, ac = ta ? m : k;
        int br = tb ? n : k, bc = tb ? k : n;
        int lda = ar + 3, ldb = br + 5, ldc = m + 7;
        std::vector<real_t> a(lda*ac), b(ldb*bc), c(ldc*n, real_t(-123));
        for (int j=0;j<ac;++j) for(int i=0;i<ar;++i) a[i+j*lda]=real_t(((i*3+j*7)%17)-8)/16;
        for (int j=0;j<bc;++j) for(int i=0;i<br;++i) b[i+j*ldb]=real_t(((i*5+j*11)%19)-9)/16;
        real_t *da, *db, *dc;
        CUDA_ERROR_CHECK(cudaMalloc(&da,a.size()*sizeof(real_t)));
        CUDA_ERROR_CHECK(cudaMalloc(&db,b.size()*sizeof(real_t)));
        CUDA_ERROR_CHECK(cudaMalloc(&dc,c.size()*sizeof(real_t)));
        CUDA_ERROR_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(real_t),cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(real_t),cudaMemcpyHostToDevice));
        Matrix A{da,ar,ac,lda}, B{db,br,bc,ldb}, C{dc,m,n,ldc};
        dim3 block(GEMM_TILE,GEMM_TILE), grid((m+GEMM_BLOCK_TILE-1)/GEMM_BLOCK_TILE,(n+GEMM_BLOCK_TILE-1)/GEMM_BLOCK_TILE);
        auto launch = [&](real_t beta) {
            if (ta) gemm_launch(tb ? "T" : "N",real_t(1),A,B,beta,C,m,n,k,grid,block,TransAt());
            else gemm_launch(tb ? "T" : "N",real_t(1),A,B,beta,C,m,n,k,grid,block,NoTransAt());
            CUDA_ERROR_CHECK(cudaGetLastError());
        };
        // Validate both beta paths, transpose combinations, padding and ragged tiles.
        for (int betaCase=0; betaCase<2; ++betaCase) {
            for(int j=0;j<n;++j) for(int i=0;i<m;++i) c[i+j*ldc]=betaCase ? real_t(.25) : std::numeric_limits<real_t>::quiet_NaN();
            CUDA_ERROR_CHECK(cudaMemcpy(dc,c.data(),c.size()*sizeof(real_t),cudaMemcpyHostToDevice));
            launch(real_t(betaCase));
            CUDA_ERROR_CHECK(cudaMemcpy(c.data(),dc,c.size()*sizeof(real_t),cudaMemcpyDeviceToHost));
            // Full CPU comparison for edge shapes; deterministic sampling for large shapes.
            int step = size > 256 ? 61 : 1;
            for(int j=0;j<n;j+=step) for(int i=0;i<m;i+=step) {
                double expected=betaCase*.25;
                for(int l=0;l<k;++l) expected += double(a[ta ? l+i*lda : i+l*lda])*double(b[tb ? j+l*ldb : l+j*ldb]);
                if (!std::isfinite(double(c[i+j*ldc])) || std::abs(double(c[i+j*ldc])-expected)>1e-4*(1+std::abs(expected))) {
                    std::fprintf(stderr,"Mismatch size=%d ta=%d tb=%d at %d,%d\n",size,ta,tb,i,j); return 1;
                }
            }
            for(int j=0;j<n;++j) for(int i=m;i<ldc;++i) if(c[i+j*ldc]!=real_t(-123)) return 1;
        }
        for(int w=0;w<5;++w) launch(0);
        cudaEvent_t start,stop;
        CUDA_ERROR_CHECK(cudaEventCreate(&start)); CUDA_ERROR_CHECK(cudaEventCreate(&stop));
        std::vector<float> times;
        for(int r=0;r<9;++r) {
            CUDA_ERROR_CHECK(cudaEventRecord(start));
            for(int i=0;i<10;++i) launch(0);
            CUDA_ERROR_CHECK(cudaEventRecord(stop)); CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
            float ms; CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms,start,stop)); times.push_back(ms/10);
        }
        std::sort(times.begin(),times.end());
        std::printf("%d,%d,%d,%d,%d,%c%c,%.6f,%.3f\n",GEMM_TILE,GEMM_K_TILE,m,n,k,ta?'T':'N',tb?'T':'N',times[4],2.*m*n*k/(times[4]*1e6));
        CUDA_ERROR_CHECK(cudaEventDestroy(start)); CUDA_ERROR_CHECK(cudaEventDestroy(stop));
        CUDA_ERROR_CHECK(cudaFree(da)); CUDA_ERROR_CHECK(cudaFree(db)); CUDA_ERROR_CHECK(cudaFree(dc));
    }
}
