// GPU/gpu_cholesky_solver.cu
//
// Direct sparse Cholesky solve on the GPU using cuSOLVER.
// Solves A x = b where A is (numerically) SPD and stored in CSR on the device.
//
// This version deliberately sets `reorder = 0` so that cuSOLVER
// does not perform its own internal reordering. Any change in
// performance when you reorder the CSR externally (RCM/AMD/etc.)
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

// NOTE: we do *not* use this macro for the actual csrlsvchol call anymore,
// because we want to handle ALLOC_FAILED gracefully there.
#define CHECK_CUSOLVER_FATAL(call)                                           \
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
    // singularity:
    //   -1  : no zero pivot (successful factorization)
    //   >=0 : index of zero pivot (matrix singular)
    //   -2  : cuSOLVER failed (e.g., ALLOC_FAILED)
    singularity = -1;
    factor_ms   = 0.0;

    // Handles
    cusolverSpHandle_t solverH = nullptr;
    cusparseMatDescr_t descrA  = nullptr;

    CHECK_CUSOLVER_FATAL(cusolverSpCreate(&solverH));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

    // Treat A as a general sparse matrix in CSR;
    // csrlsvchol assumes SPD internally.
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    const double tol     = 1e-12;  // tolerance for numerical zero pivot
    const int    reorder = 0;      // 0 => use CSR ordering as given (no internal reordering)

    // Time just the Cholesky solve
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- call cuSOLVER but handle errors explicitly (no macro) ----
    cusolverStatus_t st = cusolverSpDcsrlsvchol(
        solverH,
        n,
        nnz,
        descrA,
        d_values,  // CSR values
        d_rowPtr,  // CSR row offsets
        d_colInd,  // CSR col indices
        d_b,       // RHS
        tol,
        reorder,
        d_x,       // solution (output)
        &singularity);

    if (st != CUSOLVER_STATUS_SUCCESS) {
        // We get here for ALLOC_FAILED (code 2) and other failures.
        std::cerr << "cuSOLVER csrlsvchol failed: status = "
                  << static_cast<int>(st) << "\n";

        if (st == CUSOLVER_STATUS_ALLOC_FAILED) {
            std::cerr << "  -> CUSOLVER_STATUS_ALLOC_FAILED: resource allocation failed\n"
                      << "     This typically means the GPU (or host) is out of memory for\n"
                      << "     the Cholesky factorization workspace of this matrix.\n"
                      << "     Suggestions:\n"
                      << "       * Use the CG solver for this large mesh (--solver=cg),\n"
                      << "       * Or free unneeded GPU buffers before calling gpu_direct_cholesky,\n"
                      << "       * Or run on a GPU with more memory / use a smaller mesh.\n";
        }

        // Mark failure with a special singularity value and bail out.
        singularity = -2;
        factor_ms   = 0.0;

        // Clean up and return without aborting the whole program.
        if (descrA)  { cusparseDestroyMatDescr(descrA); }
        if (solverH) { cusolverSpDestroy(solverH); }
        return;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    factor_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Clean up
    if (descrA)  { CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA)); }
    if (solverH) { CHECK_CUSOLVER_FATAL(cusolverSpDestroy(solverH)); }
}
