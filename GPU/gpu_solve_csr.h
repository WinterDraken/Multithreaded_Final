#ifndef GPU_SOLVE_CSR_H
#define GPU_SOLVE_CSR_H

// Solve A x = b using (preconditioned) Conjugate Gradient on the GPU.
// A is real symmetric positive definite in CSR format on the device.
// This version uses a simple Jacobi (diagonal) preconditioner built
// from the CSR diagonal of A.
void solveCG_CSR_GPU(
    int n, int nnz,
    const int* d_rowPtr,
    const int* d_colInd,
    const double* d_values,
    const double* d_b,
    double* d_x,
    int maxIters,
    double tol,
    int &cg_iters);

#endif // GPU_SOLVE_CSR_H
