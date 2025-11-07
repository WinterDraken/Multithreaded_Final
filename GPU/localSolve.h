#ifndef LOCAL_SOLVE_H
#define LOCAL_SOLVE_H

#include <cuda_runtime.h>

// Compute local element stiffness matrices for Tet4 elements in 3D
// Inputs:
//   d_nodes_x, d_nodes_y, d_nodes_z: device arrays of node coordinates
//   d_elem_conn: device array of element connectivity [nElem * 4] (node IDs)
//   E_uniform, nu_uniform: material properties (Young's modulus, Poisson's ratio)
//   d_elemKe: output device array [nElem * 144] (12x12 matrix per element, row-major)
//   nElem: number of elements
//   stream: CUDA stream (optional, default 0)
void launchLocalKe_Tet4_3D(
    const double* d_nodes_x,
    const double* d_nodes_y,
    const double* d_nodes_z,
    const int*    d_elem_conn,
    double E_uniform, double nu_uniform,
    double* d_elemKe,
    int nElem, 
    cudaStream_t stream = 0);

#endif // LOCAL_SOLVE_H

