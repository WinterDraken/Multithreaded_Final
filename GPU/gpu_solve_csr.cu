#include "gpu_solve_csr.h"
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK_CUSPARSE(call)                                                 \
    do {                                                                     \
        cusparseStatus_t s = (call);                                         \
        if (s != CUSPARSE_STATUS_SUCCESS) {                                  \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << s << std::endl;                         \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t s = (call);                                           \
        if (s != CUBLAS_STATUS_SUCCESS) {                                    \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << s << std::endl;                         \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Implementation using cuSPARSE SpMV for A*p and cuBLAS for vector ops
void solveCG_CSR_GPU(
    int n, int nnz,
    const int* d_rowPtr,
    const int* d_colInd,
    const double* d_values,
    const double* d_b,
    double* d_x,
    int maxIters,
    double tol)
{
    cusparseHandle_t cusparseH = nullptr;
    cublasHandle_t cublasH = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // Create sparse matrix descriptor (CSR)
    cusparseSpMatDescr_t matA = nullptr;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, nnz,
                                     (void*)d_rowPtr, (void*)d_colInd, (void*)d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Dense vector descriptors
    double *d_r = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    cudaMalloc((void**)&d_r, n * sizeof(double));
    cudaMalloc((void**)&d_p, n * sizeof(double));
    cudaMalloc((void**)&d_Ap, n * sizeof(double));

    cusparseDnVecDescr_t vecX = nullptr, vecR = nullptr, vecP = nullptr, vecAP = nullptr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAP, n, d_Ap, CUDA_R_64F));

    // Buffer for SpMV
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    const double alpha_neg = -1.0;
    const double beta_one = 1.0;
    // compute buffer size for r = b - A*x
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseH,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_neg, matA, vecX, &beta_one, vecR,
                                           CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) cudaMalloc(&dBuffer, bufferSize);

    // r = b - A*x
    cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);
    CHECK_CUSPARSE(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_neg, matA, vecX, &beta_one, vecR,
                                CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer));

    // p = r
    cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice);

    double rsold = 0.0;
    CHECK_CUBLAS(cublasDdot(cublasH, n, d_r, 1, d_r, 1, &rsold));

    if (rsold == 0.0) {
        // Already zero residual
        if (dBuffer) cudaFree(dBuffer);
        cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecR);
        cusparseDestroyDnVec(vecP); cusparseDestroyDnVec(vecAP);
        cusparseDestroy(cusparseH);
        cublasDestroy(cublasH);
        return;
    }

    for (int iter = 0; iter < maxIters; ++iter) {
        // Ap = A * p
        const double alpha_one = 1.0;
        const double beta_zero = 0.0;
        CHECK_CUSPARSE(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_one, matA, vecP, &beta_zero, vecAP,
                                    CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer));

        // p^T * Ap
        double pAp = 0.0;
        CHECK_CUBLAS(cublasDdot(cublasH, n, d_p, 1, d_Ap, 1, &pAp));
        if (pAp == 0.0) {
            std::cerr << "Break: pAp==0 at iter " << iter << std::endl;
            break;
        }

        double alpha_cg = rsold / pAp;

        // x = x + alpha*p
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &alpha_cg, d_p, 1, d_x, 1));

        // r = r - alpha*Ap
        double neg_alpha = -alpha_cg;
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &neg_alpha, d_Ap, 1, d_r, 1));

        double rsnew = 0.0;
        CHECK_CUBLAS(cublasDdot(cublasH, n, d_r, 1, d_r, 1, &rsnew));

        if (std::sqrt(rsnew) < tol) {
            std::cout << "CG converged in " << (iter + 1) << " iterations, residual = " << std::sqrt(rsnew) << std::endl;
            break;
        }

        double beta = rsnew / rsold;

        // p = r + beta * p  => p = beta*p; p = p + r
        CHECK_CUBLAS(cublasDscal(cublasH, n, &beta, d_p, 1));
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &one, d_r, 1, d_p, 1));

        rsold = rsnew;
    }

    // cleanup
    if (dBuffer) cudaFree(dBuffer);
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecR);
    cusparseDestroyDnVec(vecP); cusparseDestroyDnVec(vecAP);
    cusparseDestroy(cusparseH);
    cublasDestroy(cublasH);
}
