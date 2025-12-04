// GPU/gpu_cholesky_solver.cu
//
// Direct sparse Cholesky solve on the GPU using cuSOLVER.
// Solves A x = b where A is SPD and stored in CSR on the device.
//
// This version deliberately sets `reorder = 0` so that cuSOLVER
// does not perform its own internal reordering. Any change in
// performance when you reorder the CSR externally (RCM/AMD/COLAMD)
// should come from your permutation, not from cuSOLVER.
//
// Signature (matches header & main):
//   void gpu_direct_cholesky(
//       int n, int nnz,
//       const int* d_rowPtr,
//       const int* d_colInd,
//       const double* d_values,
//       const double* d_b,
//       double* d_x,
//       int &singularity,
//       double &factor_ms);

#include "gpu_cholesky_solver.h"

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <chrono>
#include <iostream>
#include <cstdlib>

// ----------------- Error-checking macros -----------------

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << static_cast<int>(err__) << " ("         \
                      << cudaGetErrorString(err__) << ")\n";                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_CUSPARSE(call)                                                 \
    do {                                                                     \
        cusparseStatus_t st__ = (call);                                      \
        if (st__ != CUSPARSE_STATUS_SUCCESS) {                               \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << static_cast<int>(st__) << "\n";         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_CUSOLVER(call)                                                 \
    do {                                                                     \
        cusolverStatus_t st__ = (call);                                      \
        if (st__ != CUSOLVER_STATUS_SUCCESS) {                               \
            std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << static_cast<int>(st__) << "\n";         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


// ----------------- Direct Cholesky solve -----------------

void gpu_direct_cholesky(
    int n, int nnz,
    const int*    d_rowPtr,
    const int*    d_colInd,
    const double* d_values,
    const double* d_b,
    double*       d_x,
    int &singularity,
    double &factor_ms)
{
    singularity = -1;
    factor_ms   = 0.0;

    // Handles
    cusolverSpHandle_t solverH = nullptr;
    cusparseMatDescr_t descrA  = nullptr;

    CHECK_CUSOLVER(cusolverSpCreate(&solverH));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

    // IMPORTANT:
    // Use GENERAL here — cuSOLVER's csrlsvchol expects this,
    // even though mathematically A is SPD. The SPD assumption
    // is handled internally by csrlsvchol.
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    // Leave fill mode / diag type at defaults.

    const double tol     = 1e-12;  // relative tolerance for zero pivot
    const int    reorder = 0;      // *** do not internally reorder ***

    // Time just the Cholesky solve
    auto t0 = std::chrono::high_resolution_clock::now();

    CHECK_CUSOLVER(
        cusolverSpDcsrlsvchol(
            solverH,
            n,
            nnz,
            descrA,
            d_values,  // A values (CSR) on device
            d_rowPtr,  // row offsets (CSR) on device
            d_colInd,  // column indices (CSR) on device
            d_b,       // RHS b on device
            tol,
            reorder,   // 0 => use CSR ordering as given
            d_x,       // solution x on device (output)
            &singularity));

    CHECK_CUDA(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    factor_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Clean up
    if (descrA)  { CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA)); }
    if (solverH) { CHECK_CUSOLVER(cusolverSpDestroy(solverH)); }
}
