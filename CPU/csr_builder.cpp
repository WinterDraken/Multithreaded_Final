#include "csr_builder.h"
#include <set>
#include <algorithm>
#include <iostream>

int buildCSRPattern(
    const std::vector<Element>& elements,
    const std::vector<Node>& nodes,
    int ndof_per_node,
    bool symmetric_upper,
    std::vector<int>& rowPtr,
    std::vector<int>& colIdx,
    int& nDOF)
{
    // Find max node ID to determine DOF range
    int maxNodeId = 0;
    for (const auto& node : nodes) {
        if (node.id > maxNodeId) maxNodeId = node.id;
    }
    for (const auto& elem : elements) {
        for (int nid : elem.node_ids) {
            if (nid > maxNodeId) maxNodeId = nid;
        }
    }
    
    nDOF = (maxNodeId + 1) * ndof_per_node;
    
    // Build adjacency: for each DOF, collect connected DOFs
    std::vector<std::set<int>> adj(nDOF);
    
    for (const auto& elem : elements) {
        // For each element, connect all DOFs within the element
        for (size_t a = 0; a < elem.node_ids.size(); ++a) {
            int nodeA = elem.node_ids[a];
            for (int da = 0; da < ndof_per_node; ++da) {
                int dofA = nodeA * ndof_per_node + da;
                
                for (size_t b = 0; b < elem.node_ids.size(); ++b) {
                    int nodeB = elem.node_ids[b];
                    for (int db = 0; db < ndof_per_node; ++db) {
                        int dofB = nodeB * ndof_per_node + db;
                        
                        if (symmetric_upper) {
                            // Only store upper triangle (row <= col)
                            if (dofA <= dofB) {
                                adj[dofA].insert(dofB);
                            }
                        } else {
                            // Store full matrix
                            adj[dofA].insert(dofB);
                        }
                    }
                }
            }
        }
    }
    
    // Build CSR structure
    rowPtr.clear();
    colIdx.clear();
    rowPtr.reserve(nDOF + 1);
    rowPtr.push_back(0);
    
    int nnz = 0;
    for (int i = 0; i < nDOF; ++i) {
        // Convert set to sorted vector
        std::vector<int> cols(adj[i].begin(), adj[i].end());
        std::sort(cols.begin(), cols.end());
        
        colIdx.insert(colIdx.end(), cols.begin(), cols.end());
        nnz += cols.size();
        rowPtr.push_back(nnz);
    }
    
    return nnz;
}

