#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

#include "CPU/mesh_parser.h"
#include "CPU/csr_builder.h"
#include "CPU/reorder.h"           // <-- Added
#include "GPU/localSolve.h"
#include "GPU/globalAsm.h"
#include "GPU/gpu_solve_csr.h"     // <-- Added

// ------------------------------------------------------------
// CUDA Error Check
// ------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ------------------------------------------------------------
// Main FEM GPU Pipeline
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string meshFile = "CPU/bracket_3d.msh";
    if (argc > 1) meshFile = argv[1];

    double E = 200000.0;  // Young's modulus (MPa)
    double nu = 0.3;      // Poisson's ratio
    bool symmetric_upper = true;
    int ndof_per_node = 3;

    std::cout << "=== FEM Assembly Pipeline ===" << std::endl;
    std::cout << "Mesh file: " << meshFile << std::endl;
    std::cout << "Material: E=" << E << " MPa, nu=" << nu << std::endl;

    // ============================================
    // Step 1: Parse mesh
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
    // Step 2: Filter Tet4 and remap IDs
    // ============================================
    std::vector<Element> tet4Elements;
    for (const auto& elem : elements) {
        if (elem.type == 4 && elem.node_ids.size() == 4)
            tet4Elements.push_back(elem);
    }
    if (tet4Elements.empty()) {
        std::cerr << "No Tet4 elements found!" << std::endl;
        return 1;
    }
    std::cout << "Found " << tet4Elements.size() << " Tet4 elements" << std::endl;

    std::map<int, int> nodeIdToIndex;
    for (size_t i = 0; i < nodes.size(); ++i)
        nodeIdToIndex[nodes[i].id] = i;

    int nNodes = nodes.size();
    std::vector<double> nodeX(nNodes), nodeY(nNodes), nodeZ(nNodes);
    for (const auto& node : nodes) {
        int idx = nodeIdToIndex[node.id];
        nodeX[idx] = node.x;
        nodeY[idx] = node.y;
        nodeZ[idx] = node.z;
    }

    int nElem = tet4Elements.size();
    std::vector<int> elemConn(nElem * 4);
    for (int e = 0; e < nElem; ++e)
        for (int i = 0; i < 4; ++i)
            elemConn[e * 4 + i] = nodeIdToIndex[tet4Elements[e].node_ids[i]];

    std::cout << "Using " << nNodes << " nodes (0-based indexing)" << std::endl;

    // ============================================
    // Step 3: Build CSR pattern
    // ============================================
    std::vector<int> rowPtr, colIdx;
    int nDOF;
    int nnz = buildCSRPattern(tet4Elements, nodes,
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
    CUDA_CHECK(cudaMalloc(&d_elemKe, nElem * 144 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (nDOF + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_nodes_x, nodeX.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_y, nodeY.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_z, nodeZ.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elem_conn, elemConn.data(), nElem * 4 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), (nDOF + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_values, 0, nnz * sizeof(double)));

    // ============================================
    // Step 5: Local and global assembly
    // ============================================
    std::cout << "\nComputing local element stiffness matrices..." << std::endl;
    launchLocalKe_Tet4_3D(d_nodes_x, d_nodes_y, d_nodes_z,
                          d_elem_conn, E, nu, d_elemKe, nElem);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Assembling global CSR matrix..." << std::endl;
    launch_assembleCSR_atomic(d_elemKe, d_elem_conn, d_rowPtr, d_colIdx,
                              d_values, nElem, 4, ndof_per_node, symmetric_upper);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Global assembly complete" << std::endl;

    // ============================================
    // Step 6: Copy results back and save original
    // ============================================
    std::vector<double> values(nnz);
    CUDA_CHECK(cudaMemcpy(values.data(), d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost));

    std::ofstream outFile("global_matrix.txt");
    outFile << "# Global stiffness matrix (original order)\n";
    for (int i = 0; i < nDOF; ++i)
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
            outFile << i << " " << colIdx[j] << " " << values[j] << "\n";
    outFile.close();
    std::cout << "Original matrix saved to global_matrix.txt\n";

    // ============================================
    // Step 7: Apply Reverse Cuthill–McKee reordering
    // ============================================
    std::cout << "\nApplying Reverse Cuthill–McKee reordering..." << std::endl;
    std::vector<int> order = reverseCuthillMcKee(nDOF, rowPtr, colIdx);

    std::vector<int> rowPtr_r, colIdx_r;
    std::vector<double> values_r;
    reorderCSR(nDOF, rowPtr, colIdx, values, order, rowPtr_r, colIdx_r, values_r);
    std::cout << "Reordering complete.\n";

    // Save reordered matrix
    std::ofstream outRe("global_matrix_reordered.txt");
    outRe << "# Global stiffness matrix (reordered)\n";
    for (int i = 0; i < nDOF; ++i)
        for (int j = rowPtr_r[i]; j < rowPtr_r[i + 1]; ++j)
            outRe << i << " " << colIdx_r[j] << " " << values_r[j] << "\n";
    outRe.close();
    std::cout << "Reordered matrix saved to global_matrix_reordered.txt\n";

    // ============================================
    // Step 8: Solve reordered system on GPU
    // ============================================
    std::cout << "\nSolving reordered global system Kx = f ..." << std::endl;

    // Dummy RHS (unit load)
    std::vector<double> b(nDOF, 1.0);
    std::vector<double> x_reordered(nDOF, 0.0);

    double *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_b, nDOF * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, nDOF * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), nDOF * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x_reordered.data(), nDOF * sizeof(double), cudaMemcpyHostToDevice));

    // Solve in reordered space
    solveCG_CSR_GPU(nDOF, nnz, d_rowPtr, d_colIdx, d_values, d_b, d_x, 5000, 1e-10);

    CUDA_CHECK(cudaMemcpy(x_reordered.data(), d_x, nDOF * sizeof(double), cudaMemcpyDeviceToHost));

    // --------------------------------------------
    // Step 9: Map solution back to original order
    // --------------------------------------------
    std::vector<int> inv_perm(nDOF);
    for (int i = 0; i < nDOF; ++i)
        inv_perm[order[i]] = i;

    std::vector<double> x_original(nDOF);
    for (int i = 0; i < nDOF; ++i)
        x_original[i] = x_reordered[inv_perm[i]];

    // Save reordered-space solution (for debugging)
    std::ofstream outSolR("solution_vector_reordered.txt");
    outSolR << "# Solution vector (reordered)" << std::endl;
    for (double val : x_reordered)
        outSolR << val << "\n";
    outSolR.close();

    // Save mapped-back solution (original order)
    std::ofstream outSol("solution_vector.txt");
    outSol << "# Solution vector (original node order)" << std::endl;
    for (double val : x_original)
        outSol << val << "\n";
    outSol.close();

    std::cout << "Solution saved: solution_vector.txt (original order)\n";
    std::cout << "Reordered-space solution also saved for reference.\n";

    // Cleanup
    cudaFree(d_nodes_x); cudaFree(d_nodes_y); cudaFree(d_nodes_z);
    cudaFree(d_elem_conn); cudaFree(d_elemKe);
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
    cudaFree(d_b); cudaFree(d_x);

    std::cout << "\nDone!" << std::endl;
    return 0;

}
