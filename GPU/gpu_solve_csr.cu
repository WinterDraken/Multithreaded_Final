#include "gpu_solve_csr.h"

#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>

// ------------------- Error-checking macros -------------------

#define CHECK_CUSPARSE(call)                                                 \
    do {                                                                     \
        cusparseStatus_t s__ = (call);                                       \
        if (s__ != CUSPARSE_STATUS_SUCCESS) {                                \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << static_cast<int>(s__) << std::endl;     \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t s__ = (call);                                         \
        if (s__ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << static_cast<int>(s__) << std::endl;     \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t e__ = (call);                                            \
        if (e__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(e__) << std::endl;      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


// ------------------- Simple Jacobi preconditioner -------------------

// Extract diagonal of CSR matrix A and build diagonal inverse Minv = 1 / diag(A).
__global__
void extractDiagonalKernel(int n,
                           const int* __restrict__ rowPtr,
                           const int* __restrict__ colInd,
                           const double* __restrict__ values,
                           double* __restrict__ Minv)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = rowPtr[row];
    int end   = rowPtr[row + 1];

    double diag = 0.0;
    for (int idx = start; idx < end; ++idx) {
        if (colInd[idx] == row) {
            diag = values[idx];
            break;
        }
    }

    if (std::fabs(diag) > 0.0) {
        Minv[row] = 1.0 / diag;
    } else {
        // Fallback if diagonal is missing or zero: no scaling.
        Minv[row] = 1.0;
    }
}

// Apply Jacobi preconditioner: z = Minv .* r
__global__
void applyJacobiKernel(int n,
                       const double* __restrict__ Minv,
                       const double* __restrict__ r,
                       double* __restrict__ z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = Minv[i] * r[i];
}


// ------------------- Preconditioned CG solver -------------------

void solveCG_CSR_GPU(
    int n, int nnz,
    const int* d_rowPtr,
    const int* d_colInd,
    const double* d_values,
    const double* d_b,
    double* d_x,
    int maxIters,
    double tol,
    int &cg_iters)
{
    cg_iters = 0;

    // Handles
    cusparseHandle_t cusparseH = nullptr;
    cublasHandle_t   cublasH   = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // Create sparse matrix descriptor A in CSR
    cusparseSpMatDescr_t matA = nullptr;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        /*rows=*/n, /*cols=*/n, /*nnz=*/nnz,
        (void*)d_rowPtr,
        (void*)d_colInd,
        (void*)d_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    // Allocate vectors: r, p, Ap, z (preconditioned residual), Minv (diag inverse)
    double *d_r    = nullptr;
    double *d_p    = nullptr;
    double *d_Ap   = nullptr;
    double *d_z    = nullptr;
    double *d_Minv = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_r,    n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_p,    n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Ap,   n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_z,    n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Minv, n * sizeof(double)));

    // Build Jacobi preconditioner: Minv = 1 / diag(A)
    {
        int block = 256;
        int grid  = (n + block - 1) / block;
        extractDiagonalKernel<<<grid, block>>>(n, d_rowPtr, d_colInd, d_values, d_Minv);
        CHECK_CUDA(cudaGetLastError());
    }

    // Dense vector descriptors for SpMV: we only need p and Ap
    cusparseDnVecDescr_t vecP  = nullptr;
    cusparseDnVecDescr_t vecAp = nullptr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP,  n, d_p,  CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_64F));

    // SpMV buffer
    double alpha = 1.0;
    double beta  = 0.0;
    size_t bufferSize = 0;
    void*  dBuffer    = nullptr;

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        cusparseH,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecP,
        &beta,
        vecAp,
        CUDA_R_64F,
        CUSPARSE_MV_ALG_DEFAULT,
        &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Initial guess x is assumed to be zero (main does cudaMemset).
    // r0 = b - A*x0 = b.
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // z0 = M^{-1} r0
    {
        int block = 256;
        int grid  = (n + block - 1) / block;
        applyJacobiKernel<<<grid, block>>>(n, d_Minv, d_r, d_z);
        CHECK_CUDA(cudaGetLastError());
    }

    // p0 = z0
    CHECK_CUBLAS(cublasDcopy(cublasH, n, d_z, 1, d_p, 1));

    // rsold = r^T z
    double rsold = 0.0;
    CHECK_CUBLAS(cublasDdot(cublasH, n, d_r, 1, d_z, 1, &rsold));

    // Norm of b for relative tolerance (optional)
    double bnorm = 0.0;
    CHECK_CUBLAS(cublasDdot(cublasH, n, d_b, 1, d_b, 1, &bnorm));
    bnorm = std::sqrt(bnorm);
    if (bnorm == 0.0) bnorm = 1.0;
    const double absTol = tol * bnorm;

    // CG iteration
    for (int k = 0; k < maxIters; ++k) {
        // Ap = A * p
        alpha = 1.0;
        beta  = 0.0;
        CHECK_CUSPARSE(cusparseSpMV(
            cusparseH,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecP,
            &beta,
            vecAp,
            CUDA_R_64F,
            CUSPARSE_MV_ALG_DEFAULT,
            dBuffer));

        // pAp = p^T Ap
        double pAp = 0.0;
        CHECK_CUBLAS(cublasDdot(cublasH, n, d_p, 1, d_Ap, 1, &pAp));

        if (pAp == 0.0) {
            std::cerr << "CG breakdown: p^T A p == 0\n";
            break;
        }

        double alpha_k = rsold / pAp;

        // x_{k+1} = x_k + alpha * p
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &alpha_k, d_p, 1, d_x, 1));

        // r_{k+1} = r_k - alpha * Ap
        double neg_alpha = -alpha_k;
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &neg_alpha, d_Ap, 1, d_r, 1));

        // Check residual norm ||r||
        double rsq = 0.0;
        CHECK_CUBLAS(cublasDdot(cublasH, n, d_r, 1, d_r, 1, &rsq));
        double rnorm = std::sqrt(rsq);
        if (rnorm < absTol) {
            cg_iters = k + 1;
            break;
        }

        // z_{k+1} = M^{-1} r_{k+1}
        {
            int block = 256;
            int grid  = (n + block - 1) / block;
            applyJacobiKernel<<<grid, block>>>(n, d_Minv, d_r, d_z);
            CHECK_CUDA(cudaGetLastError());
        }

        // rsnew = r_{k+1}^T z_{k+1}
        double rsnew = 0.0;
        CHECK_CUBLAS(cublasDdot(cublasH, n, d_r, 1, d_z, 1, &rsnew));

        if (rsnew == 0.0) {
            cg_iters = k + 1;
            break;
        }

        double beta_k = rsnew / rsold;

        // p_{k+1} = z_{k+1} + beta * p_k
        CHECK_CUBLAS(cublasDscal(cublasH, n, &beta_k, d_p, 1));
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &one, d_z, 1, d_p, 1));

        rsold = rsnew;
        cg_iters = k + 1;
    }

    // Cleanup
    if (dBuffer)   CHECK_CUDA(cudaFree(dBuffer));
    if (d_r)       CHECK_CUDA(cudaFree(d_r));
    if (d_p)       CHECK_CUDA(cudaFree(d_p));
    if (d_Ap)      CHECK_CUDA(cudaFree(d_Ap));
    if (d_z)       CHECK_CUDA(cudaFree(d_z));
    if (d_Minv)    CHECK_CUDA(cudaFree(d_Minv));

    if (vecP)      CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    if (vecAp)     CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    if (matA)      CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    if (cusparseH) CHECK_CUSPARSE(cusparseDestroy(cusparseH));
    if (cublasH)   CHECK_CUBLAS(cublasDestroy(cublasH));
}
