#ifndef GLOBAL_ASM_H
#define GLOBAL_ASM_H

#include <cuda_runtime.h>

// Assemble element stiffness matrices into global CSR matrix
// Inputs:
//   d_elemKe: device array [nElem * (nen*ndof)^2] of element stiffness matrices
//   d_elem_conn: device array [nElem * nen] of element connectivity (global node IDs)
//   d_rowPtr, d_colIdx: device arrays for CSR structure (global DOF level)
//   d_values: device array for CSR values (must be zeroed before call)
//   nElem: number of elements
//   nen: nodes per element (e.g., 4 for Tet4)
//   ndof_per_node: DOFs per node (e.g., 3 for 3D elasticity)
//   symmetric_upper: if true, CSR stores only upper triangle
//   stream: CUDA stream (optional, default 0)
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
    cudaStream_t stream = 0);

#endif // GLOBAL_ASM_H

