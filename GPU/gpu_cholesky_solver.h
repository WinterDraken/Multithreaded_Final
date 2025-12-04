void gpu_direct_cholesky(
    int n, int nnz,
    const int* d_rowPtr,
    const int* d_colInd,
    const double* d_values,
    const double* d_b,
    double* d_x,
    int& singularity,
    double& factor_ms);
