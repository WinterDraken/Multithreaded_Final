#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include "CPU/mesh_parser.h"
#include "CPU/csr_builder.h"
#include "GPU/localSolve.h"
#include "GPU/globalAsm.h"

// Helper: Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string meshFile = "CPU/bracket_3d.msh";
    if (argc > 1) {
        meshFile = argv[1];
    }
    
    double E = 200000.0;  // Young's modulus (MPa)
    double nu = 0.3;      // Poisson's ratio
    bool symmetric_upper = true;  // Store only upper triangle
    
    std::cout << "=== FEM Assembly Pipeline ===" << std::endl;
    std::cout << "Mesh file: " << meshFile << std::endl;
    std::cout << "Material: E=" << E << " MPa, nu=" << nu << std::endl;
    
    // ============================================
    // Step 1: Parse mesh file
    // ============================================
    std::vector<Node> nodes;
    std::vector<Element> elements;
    
    if (!parseMeshFile(meshFile, nodes, elements)) {
        std::cerr << "Failed to parse mesh file" << std::endl;
        return 1;
    }
    
    std::cout << "\nParsed " << nodes.size() << " nodes and " 
              << elements.size() << " elements" << std::endl;
    
    // ============================================
    // Step 2: Filter Tet4 elements (type 4) and remap node IDs
    // ============================================
    std::vector<Element> tet4Elements;
    for (const auto& elem : elements) {
        if (elem.type == 4 && elem.node_ids.size() == 4) {  // Tet4
            tet4Elements.push_back(elem);
        }
    }
    
    if (tet4Elements.empty()) {
        std::cerr << "No Tet4 elements found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << tet4Elements.size() << " Tet4 elements" << std::endl;
    
    // Build node ID to index mapping (Gmsh uses 1-based, we need 0-based)
    std::map<int, int> nodeIdToIndex;
    for (size_t i = 0; i < nodes.size(); ++i) {
        nodeIdToIndex[nodes[i].id] = i;
    }
    
    // Find max node index used in elements
    int maxNodeIndex = -1;
    for (const auto& elem : tet4Elements) {
        for (int nid : elem.node_ids) {
            if (nodeIdToIndex.find(nid) != nodeIdToIndex.end()) {
                int idx = nodeIdToIndex[nid];
                if (idx > maxNodeIndex) maxNodeIndex = idx;
            }
        }
    }
    int nNodes = maxNodeIndex + 1;
    
    // Build node coordinate arrays (0-based indexing)
    std::vector<double> nodeX(nNodes), nodeY(nNodes), nodeZ(nNodes);
    for (const auto& node : nodes) {
        if (nodeIdToIndex.find(node.id) != nodeIdToIndex.end()) {
            int idx = nodeIdToIndex[node.id];
            if (idx < nNodes) {
                nodeX[idx] = node.x;
                nodeY[idx] = node.y;
                nodeZ[idx] = node.z;
            }
        }
    }
    
    // Build element connectivity array (0-based node indices)
    int nElem = tet4Elements.size();
    std::vector<int> elemConn(nElem * 4);
    for (int e = 0; e < nElem; ++e) {
        for (int i = 0; i < 4; ++i) {
            int gmshNodeId = tet4Elements[e].node_ids[i];
            int localIdx = nodeIdToIndex[gmshNodeId];
            elemConn[e * 4 + i] = localIdx;
        }
    }
    
    std::cout << "Using " << nNodes << " nodes (0-based indexing)" << std::endl;
    
    // ============================================
    // Step 3: Build CSR pattern
    // ============================================
    std::cout << "\nBuilding CSR pattern..." << std::endl;
    std::vector<int> rowPtr, colIdx;
    int nDOF;
    int ndof_per_node = 3;  // 3D elasticity
    
    // Create elements with remapped node IDs for CSR builder
    std::vector<Element> remappedElements = tet4Elements;
    for (auto& elem : remappedElements) {
        for (size_t i = 0; i < elem.node_ids.size(); ++i) {
            int gmshId = elem.node_ids[i];
            elem.node_ids[i] = nodeIdToIndex[gmshId];
        }
    }
    
    // Create remapped nodes (0-based IDs)
    std::vector<Node> remappedNodes(nNodes);
    for (int i = 0; i < nNodes; ++i) {
        remappedNodes[i].id = i;
        remappedNodes[i].x = nodeX[i];
        remappedNodes[i].y = nodeY[i];
        remappedNodes[i].z = nodeZ[i];
    }
    
    int nnz = buildCSRPattern(remappedElements, remappedNodes, 
                              ndof_per_node, symmetric_upper,
                              rowPtr, colIdx, nDOF);
    
    std::cout << "CSR: " << nDOF << " DOFs, " << nnz << " non-zeros" << std::endl;
    
    // ============================================
    // Step 4: Allocate GPU memory
    // ============================================
    std::cout << "\nAllocating GPU memory..." << std::endl;
    
    double *d_nodes_x, *d_nodes_y, *d_nodes_z;
    int *d_elem_conn;
    double *d_elemKe;
    int *d_rowPtr, *d_colIdx;
    double *d_values;
    
    CUDA_CHECK(cudaMalloc(&d_nodes_x, nNodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nodes_y, nNodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nodes_z, nNodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_elem_conn, nElem * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_elemKe, nElem * 144 * sizeof(double)));  // 12x12 = 144 per element
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (nDOF + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_nodes_x, nodeX.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_y, nodeY.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_z, nodeZ.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elem_conn, elemConn.data(), nElem * 4 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), (nDOF + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    
    // Zero out CSR values
    CUDA_CHECK(cudaMemset(d_values, 0, nnz * sizeof(double)));
    
    // ============================================
    // Step 5: Compute local element stiffness matrices
    // ============================================
    std::cout << "\nComputing local element stiffness matrices..." << std::endl;
    launchLocalKe_Tet4_3D(d_nodes_x, d_nodes_y, d_nodes_z,
                          d_elem_conn,
                          E, nu,
                          d_elemKe,
                          nElem);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Local Ke computation complete" << std::endl;
    
    // ============================================
    // Step 6: Assemble global matrix
    // ============================================
    std::cout << "\nAssembling global CSR matrix..." << std::endl;
    launch_assembleCSR_atomic(d_elemKe, d_elem_conn,
                              d_rowPtr, d_colIdx, d_values,
                              nElem,
                              4,  // nen (nodes per element for Tet4)
                              ndof_per_node,
                              symmetric_upper);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Global assembly complete" << std::endl;
    
    // ============================================
    // Step 7: Optional: Copy results back and verify
    // ============================================
    std::vector<double> values(nnz);
    CUDA_CHECK(cudaMemcpy(values.data(), d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Compute some statistics
    double sum = 0.0, maxVal = 0.0, minVal = 0.0;
    bool first = true;
    for (int i = 0; i < nnz; ++i) {
        sum += values[i];
        if (first || values[i] > maxVal) maxVal = values[i];
        if (first || values[i] < minVal) minVal = values[i];
        first = false;
    }
    
    std::cout << "\n=== Assembly Statistics ===" << std::endl;
    std::cout << "Matrix size: " << nDOF << " x " << nDOF << std::endl;
    std::cout << "Non-zeros: " << nnz << std::endl;
    std::cout << "Value range: [" << minVal << ", " << maxVal << "]" << std::endl;
    std::cout << "Sum of all values: " << sum << std::endl;
    
    // ============================================
    // Step 8: Save matrix to file
    // ============================================
    std::string matrixFile = "global_matrix.txt";
    std::cout << "\nSaving matrix to " << matrixFile << "..." << std::endl;
    
    std::ofstream outFile(matrixFile);
    if (!outFile.is_open()) {
        std::cerr << "Warning: Could not open " << matrixFile << " for writing" << std::endl;
    } else {
        // Write header
        outFile << "# Global Stiffness Matrix (CSR format)" << std::endl;
        outFile << "# Matrix size: " << nDOF << " x " << nDOF << std::endl;
        outFile << "# Number of non-zeros: " << nnz << std::endl;
        outFile << "# Material properties: E = " << E << " MPa, nu = " << nu << std::endl;
        outFile << "# Symmetric upper triangle: " << (symmetric_upper ? "yes" : "no") << std::endl;
        outFile << "# Format: Row Column Value (0-based indices)" << std::endl;
        outFile << "#" << std::endl;
        
        // Write matrix entries in coordinate format (row, col, value)
        outFile << std::scientific << std::setprecision(15);
        for (int i = 0; i < nDOF; ++i) {
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                int col = colIdx[j];
                double val = values[j];
                outFile << i << " " << col << " " << val << std::endl;
            }
        }
        
        outFile.close();
        std::cout << "Matrix saved successfully!" << std::endl;
        
        // Also save CSR arrays for reference
        std::string csrFile = "csr_arrays.txt";
        std::ofstream csrOut(csrFile);
        if (csrOut.is_open()) {
            csrOut << "# CSR Arrays" << std::endl;
            csrOut << "# Matrix size: " << nDOF << " x " << nDOF << std::endl;
            csrOut << "# Non-zeros: " << nnz << std::endl;
            csrOut << "#" << std::endl;
            csrOut << "# RowPtr array (length " << (nDOF + 1) << "):" << std::endl;
            for (size_t i = 0; i < rowPtr.size(); ++i) {
                csrOut << rowPtr[i];
                if (i < rowPtr.size() - 1) csrOut << " ";
            }
            csrOut << std::endl << std::endl;
            
            csrOut << "# ColIdx array (length " << nnz << "):" << std::endl;
            for (int i = 0; i < nnz; ++i) {
                csrOut << colIdx[i];
                if (i < nnz - 1) csrOut << " ";
            }
            csrOut << std::endl << std::endl;
            
            csrOut << "# Values array (length " << nnz << "):" << std::endl;
            csrOut << std::scientific << std::setprecision(15);
            for (int i = 0; i < nnz; ++i) {
                csrOut << values[i];
                if (i < nnz - 1) csrOut << " ";
            }
            csrOut << std::endl;
            csrOut.close();
            std::cout << "CSR arrays also saved to " << csrFile << std::endl;
        }
    }
    
    // ============================================
    // Cleanup
    // ============================================
    cudaFree(d_nodes_x);
    cudaFree(d_nodes_y);
    cudaFree(d_nodes_z);
    cudaFree(d_elem_conn);
    cudaFree(d_elemKe);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

