#ifndef GLOBAL_ASM_H
#define GLOBAL_ASM_H

#include <cuda_runtime.h>

// Assemble element stiffness matrices into global CSR matrix
// Inputs:
//   d_elemKe:      [nElem * (nen*ndof)^2] element stiffness matrices
//   d_elem_conn:   [nElem * nen] connectivity (global node IDs)
//   d_rowPtr, d_colIdx: CSR structure (global DOF, *permuted*)
//   d_values:      CSR values (must be zeroed before call)
//   d_dofOld2New:  map old DOF -> new DOF (permutation)
//   nElem:         number of elements
//   nen:           nodes per element (e.g., 4 for Tet4)
//   ndof_per_node: DOFs per node (e.g., 3 for 3D)
//   symmetric_upper: if true, CSR stores only upper triangle
//   stream:        CUDA stream (optional, default 0)
void launch_assembleCSR_atomic(
    const double* d_elemKe,      // device
    const int*    d_elem_conn,   // device
    const int*    d_rowPtr,      // device (permuted)
    const int*    d_colIdx,      // device (permuted)
    double*       d_values,      // device (zeroed before call)
    const int*    d_dofOld2New,  // device map old_dof -> new_dof
    int           nElem,
    int           nen,
    int           ndof_per_node,
    bool          symmetric_upper,
    cudaStream_t  stream = 0);

#endif // GLOBAL_ASM_H
