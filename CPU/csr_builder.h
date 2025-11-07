#ifndef CSR_BUILDER_H
#define CSR_BUILDER_H

#include <vector>
#include "mesh_parser.h"

// Build CSR structure (rowPtr, colIdx) from element connectivity
// For 3D elasticity: each node has 3 DOFs (x, y, z)
// 
// Inputs:
//   elements: vector of elements with connectivity
//   nodes: vector of nodes (to determine max node ID)
//   ndof_per_node: DOFs per node (typically 3 for 3D)
//   symmetric_upper: if true, only store upper triangle
//
// Outputs:
//   rowPtr: CSR row pointer array (size = nDOF + 1)
//   colIdx: CSR column index array
//   nDOF: total number of DOFs (nodes.size() * ndof_per_node)
//
// Returns: number of non-zeros (nnz)
int buildCSRPattern(
    const std::vector<Element>& elements,
    const std::vector<Node>& nodes,
    int ndof_per_node,
    bool symmetric_upper,
    std::vector<int>& rowPtr,
    std::vector<int>& colIdx,
    int& nDOF);

#endif // CSR_BUILDER_H

