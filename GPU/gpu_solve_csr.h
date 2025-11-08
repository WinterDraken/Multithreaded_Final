#ifndef GPU_SOLVE_CSR_H
#define GPU_SOLVE_CSR_H

// Solve A x = b with conjugate gradient on GPU using CSR (cuSPARSE + cuBLAS).
// Inputs:
//   n        - matrix dimension
//   nnz      - number of nonzeros in CSR
//   d_rowPtr - device pointer to rowPtr (length n+1) (int*)
//   d_colInd - device pointer to col indices (length nnz) (int*)
//   d_values - device pointer to CSR values (length nnz) (double*)
//   d_b      - device pointer to rhs b (length n) (double*)
//   d_x      - device pointer to initial guess / output x (length n) (double*)
// Optional:
//   maxIters, tol - CG stopping criteria
//
// Notes: Requires linking with -lcusparse -lcublas.
void solveCG_CSR_GPU(
    int n, int nnz,
    const int* d_rowPtr,
    const int* d_colInd,
    const double* d_values,
    const double* d_b,
    double* d_x,
    int maxIters = 1000,
    double tol = 1e-10);

#endif // GPU_SOLVE_CSR_H
