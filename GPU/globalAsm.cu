// globalAsm.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

// ---------------------------------------------
// Small helpers
// ---------------------------------------------

// Binary search col 'c' in CSR row r: rowPtr[r]..rowPtr[r+1]-1
__device__ __forceinline__
int csr_find_pos(const int* __restrict__ rowPtr,
                 const int* __restrict__ colIdx,
                 int r, int c)
{
    int lo = rowPtr[r];
    int hi = rowPtr[r+1] - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int cc  = colIdx[mid];
        if (cc < c)      lo = mid + 1;
        else if (cc > c) hi = mid - 1;
        else             return mid;  // found
    }
    return -1; // not found (shouldn't happen if CSR pattern was built correctly)
}

// Map (local dof index) -> (global dof index) for 3D, ndof_per_node = 3
__device__ __forceinline__
int global_dof(int global_node, int d, int ndof_per_node)
{
    return global_node * ndof_per_node + d;
}

// ---------------------------------------------
// Kernel: assemble element Ke into global CSR
// Each thread handles ONE element.
// Ke is (nen*ndof) x (nen*ndof)
// ---------------------------------------------
__global__
void assembleCSR_atomic_kernel(
    const double* __restrict__ elemKe,   // [nElem * (nen*ndof)^2]
    const int*    __restrict__ elem_conn,// [nElem * nen] (global node IDs)
    const int*    __restrict__ rowPtr,   // CSR row offsets for GLOBAL DOFs
    const int*    __restrict__ colIdx,   // CSR col indices for GLOBAL DOFs
    double*       __restrict__ values,   // CSR values (initialized to 0)
    int nElem,
    int nen,              // nodes per element (Tet4 -> 4)
    int ndof_per_node,    // 3 for 3D elasticity
    bool symmetric_upper) // if CSR stores only upper triangle
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    const int ndofe = nen * ndof_per_node;     // element DOFs (Tet4,3D) = 12
    const int KeBase = e * (ndofe * ndofe);    // row-major block for element e

    // Load element connectivity (global node IDs)
    // Keep in registers for reuse
    int gnode[8]; // enough for many small elements; nen<=8 typical; here nen=4
    #pragma unroll
    for (int a = 0; a < nen; ++a) {
        gnode[a] = elem_conn[e * nen + a];
    }

    // Double loop over element dofs (I,J)
    // map to global row/col DOFs, then atomicAdd into CSR.values
    for (int a = 0; a < nen; ++a) {
        for (int da = 0; da < ndof_per_node; ++da) {
            int I = a * ndof_per_node + da;
            int r_dof = global_dof(gnode[a], da, ndof_per_node);

            for (int b = 0; b < nen; ++b) {
                for (int db = 0; db < ndof_per_node; ++db) {
                    int J = b * ndof_per_node + db;
                    int c_dof = global_dof(gnode[b], db, ndof_per_node);

                    int row = r_dof;
                    int col = c_dof;
                    double val = elemKe[KeBase + I * ndofe + J];

                    // If CSR stores only upper triangle, redirect (row>col) to (col,row)
                    if (symmetric_upper && row > col) {
                        // Since Ke is symmetric for linear elasticity, we can just add to the mirrored slot
                        int tmp = row; row = col; col = tmp;
                        // val remains the same (Ke[I,J] == Ke[J,I] ideally)
                    }

                    // Find position in CSR
                    int pos = csr_find_pos(rowPtr, colIdx, row, col);
                    if (pos >= 0) {
                        atomicAdd(&values[pos], val);
                    } else {
                        // Pattern mismatch (shouldn't happen if CSR built from adjacency+DOFs)
                        // You may count/report errors if needed.
                    }
                }
            }
        }
    }
}

// ---------------------------------------------
// Host launcher
// ---------------------------------------------
void launch_assembleCSR_atomic(
    const double* d_elemKe,     // device
    const int*    d_elem_conn,  // device
    const int*    d_rowPtr,     // device
    const int*    d_colIdx,     // device
    double*       d_values,     // device (zeroed before call)
    int nElem,
    int nen,                    // e.g., 4
    int ndof_per_node,          // e.g., 3
    bool symmetric_upper,
    cudaStream_t stream = 0)
{
    int block = 128;
    int grid  = (nElem + block - 1) / block;
    assembleCSR_atomic_kernel<<<grid, block, 0, stream>>>(
        d_elemKe, d_elem_conn,
        d_rowPtr, d_colIdx, d_values,
        nElem, nen, ndof_per_node, symmetric_upper);
}
