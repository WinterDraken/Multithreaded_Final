#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <numeric>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>

#include "CPU/mesh_parser.h"
#include "CPU/csr_builder.h"
#include "CPU/reorder.h"
#include "GPU/localSolve.h"
#include "GPU/globalAsm.h"
#include "GPU/gpu_solve_csr.h"

// CUDA error check macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

using Clock = std::chrono::high_resolution_clock;

// Create directory if it doesn't exist
static void makeDir(const std::string& p) {
    struct stat st = {0};
    if (stat(p.c_str(), &st) == -1) {
        mkdir(p.c_str(), 0755);
    }
}

// ------------------------------------------------------------
// MAIN FEM GPU PIPELINE
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    // -------------------- Args: mesh & method & run_id ----------------------
    std::string meshFile = "CPU/bracket_3d.msh";
    if (argc > 1) meshFile = argv[1];

    std::string methodStr = "rcm";
    if (argc > 2) methodStr = argv[2];
    
    std::string runId = "";  // Optional run identifier for unique filenames
    if (argc > 3) runId = argv[3];

    ReorderMethod method = parseReorderMethod(methodStr);
    std::string methodTag = methodToString(method);

    // Extract mesh base name (no directory, no extension)
    std::string meshBase;
    {
        size_t slash = meshFile.find_last_of("/\\");
        std::string tmp = (slash == std::string::npos) ?
            meshFile : meshFile.substr(slash + 1);
        size_t dot = tmp.find_last_of(".");
        meshBase = (dot == std::string::npos) ? tmp : tmp.substr(0, dot);
    }

    // Output directories
    std::string rootFolder   = "results";
    std::string methodFolder = rootFolder + "/" + methodTag;
    makeDir(rootFolder);
    makeDir(methodFolder);

    double E = 200000.0;
    double nu = 0.3;
    bool symmetric_upper = true;
    int ndof_per_node = 3;

    std::cout << "=== FEM GPU Pipeline ===\n";
    std::cout << "Mesh file: " << meshFile << "\n";
    std::cout << "Mesh base: " << meshBase << "\n";
    std::cout << "Reordering method: " << methodTag << "\n";
    std::cout << "Material: E=" << E << " MPa, nu=" << nu << "\n\n";

    // -------------------- Timing accumulators ----------------------
    double cpuAssemblyMs = 0.0;
    double reorderMs     = 0.0;
    double h2dMs         = 0.0;
    double solveMs       = 0.0;

    // Timed host-to-device copy helper
    auto h2d_copy = [&](auto* dst, const auto* src, size_t bytes) {
        auto t0 = Clock::now();
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        auto t1 = Clock::now();
        h2dMs += std::chrono::duration<double,std::milli>(t1 - t0).count();
    };

    // -------------------- Step 1: Parse mesh -----------------------
    std::vector<Node> nodes;
    std::vector<Element> elements;
    if (!parseMeshFile(meshFile, nodes, elements)) {
        std::cerr << "ERROR: Failed to parse mesh file.\n";
        return 1;
    }

    std::cout << "Parsed " << nodes.size() << " nodes, "
              << elements.size() << " elements.\n";

    // -------------------- Step 2: Filter Tet4 ----------------------
    std::vector<Element> tet4Elements;
    tet4Elements.reserve(elements.size());
    for (const auto& elem : elements) {
        if (elem.type == 4 && elem.node_ids.size() == 4)
            tet4Elements.push_back(elem);
    }

    if (tet4Elements.empty()) {
        std::cerr << "ERROR: No Tet4 elements found.\n";
        return 1;
    }

    int nNodes = static_cast<int>(nodes.size());
    int nElem  = static_cast<int>(tet4Elements.size());
    std::cout << "Tet4 elements: " << nElem << "\n";

    // Map node IDs to contiguous indices
    std::map<int,int> nodeIdToIndex;
    for (int i = 0; i < nNodes; ++i)
        nodeIdToIndex[nodes[i].id] = i;

    std::vector<double> nodeX(nNodes), nodeY(nNodes), nodeZ(nNodes);
    for (const auto& n : nodes) {
        int idx = nodeIdToIndex[n.id];
        nodeX[idx] = n.x;
        nodeY[idx] = n.y;
        nodeZ[idx] = n.z;
    }

    std::vector<int> elemConn(nElem * 4);
    for (int e = 0; e < nElem; ++e)
        for (int i = 0; i < 4; ++i)
            elemConn[e*4 + i] = nodeIdToIndex[tet4Elements[e].node_ids[i]];

    // -------------------- Step 3: Build CSR Pattern (CPU) ----------
    std::vector<int> rowPtr, colIdx;
    int nDOF;
    auto tA0 = Clock::now();
    int nnz = buildCSRPattern(tet4Elements, nodes,
                              ndof_per_node, symmetric_upper,
                              rowPtr, colIdx, nDOF);
    auto tA1 = Clock::now();
    cpuAssemblyMs = std::chrono::duration<double,std::milli>(tA1 - tA0).count();

    std::cout << "CSR pattern: nDOF=" << nDOF
              << ", nnz=" << nnz << "\n";

    // -------------------- Step 4: Allocate GPU memory --------------
    std::cout << "Allocating GPU memory...\n";

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

    // H2D copies (timed)
    h2d_copy(d_nodes_x, nodeX.data(), nNodes * sizeof(double));
    h2d_copy(d_nodes_y, nodeY.data(), nNodes * sizeof(double));
    h2d_copy(d_nodes_z, nodeZ.data(), nNodes * sizeof(double));
    h2d_copy(d_elem_conn, elemConn.data(), nElem * 4 * sizeof(int));
    h2d_copy(d_rowPtr, rowPtr.data(), (nDOF + 1) * sizeof(int));
    h2d_copy(d_colIdx, colIdx.data(), nnz * sizeof(int));
    CUDA_CHECK(cudaMemset(d_values, 0, nnz * sizeof(double)));

    // -------------------- Step 5: Local & Global Assembly (GPU) ----
    std::cout << "Computing local element stiffness (GPU)...\n";
    launchLocalKe_Tet4_3D(d_nodes_x, d_nodes_y, d_nodes_z,
                          d_elem_conn, E, nu, d_elemKe, nElem);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Assembling global CSR matrix (GPU)...\n";
    launch_assembleCSR_atomic(d_elemKe, d_elem_conn,
                              d_rowPtr, d_colIdx,
                              d_values,
                              nElem, 4,
                              ndof_per_node, symmetric_upper);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Assembly complete.\n";

    // -------------------- Step 6: Copy global matrix back ----------
    std::vector<double> values(nnz);
    CUDA_CHECK(cudaMemcpy(values.data(), d_values,
                          nnz * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Commented out: Save original matrix
    // {
    //     std::ofstream out(methodFolder + "/" +
    //                       meshBase + "__global_matrix.txt");
    //     out << "# Global stiffness matrix (original order)\n";
    //     for (int i = 0; i < nDOF; ++i)
    //         for (int j = rowPtr[i]; j < rowPtr[i+1]; ++j)
    //             out << i << " " << colIdx[j] << " " << values[j] << "\n";
    //     std::cout << "Saved original matrix.\n";
    // }

    // -------------------- Step 7: Reordering (AFTER assembly) -----
    std::vector<int> perm;
    std::vector<int> rowPtr_r, colIdx_r;
    std::vector<double> values_r;

    auto tR0 = Clock::now();
    perm = computePermutation(nDOF, rowPtr, colIdx, method);

    if (method == ReorderMethod::NONE) {
        rowPtr_r = rowPtr;
        colIdx_r = colIdx;
        values_r = values;
    } else {
        reorderCSR(nDOF, rowPtr, colIdx, values,
                   perm, rowPtr_r, colIdx_r, values_r);
    }
    auto tR1 = Clock::now();
    reorderMs = std::chrono::duration<double,std::milli>(tR1 - tR0).count();

    // Commented out: Save reordered matrix
    // {
    //     std::ofstream out(methodFolder + "/" +
    //                       meshBase + "__global_matrix_reordered.txt");
    //     out << "# Global stiffness matrix (reordered: " << methodTag << ")\n";
    //     for (int i = 0; i < nDOF; ++i)
    //         for (int j = rowPtr_r[i]; j < rowPtr_r[i+1]; ++j)
    //             out << i << " " << colIdx_r[j] << " " << values_r[j] << "\n";
    //     std::cout << "Saved reordered matrix.\n";
    // }

    // -------------------- Step 8: Solve Kx=b in reordered space ----
    std::cout << "Solving Kx = f in reordered space...\n";

    // RHS in original ordering
    std::vector<double> b_original(nDOF, 1.0);

    // Permute RHS: b_perm[new] = b_original[old]
    std::vector<double> b_perm(nDOF);
    for (int newIdx = 0; newIdx < nDOF; ++newIdx) {
        int oldIdx = perm[newIdx];
        b_perm[newIdx] = b_original[oldIdx];
    }

    std::vector<double> x_reordered(nDOF, 0.0);

    double *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_b, nDOF * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, nDOF * sizeof(double)));

    // Overwrite CSR & values on device with reordered versions
    h2d_copy(d_rowPtr, rowPtr_r.data(), (nDOF + 1) * sizeof(int));
    h2d_copy(d_colIdx, colIdx_r.data(), nnz * sizeof(int));
    h2d_copy(d_values, values_r.data(), nnz * sizeof(double));
    h2d_copy(d_b,      b_perm.data(),  nDOF * sizeof(double));
    h2d_copy(d_x,      x_reordered.data(), nDOF * sizeof(double));

    auto tS0 = Clock::now();
    solveCG_CSR_GPU(nDOF, nnz,
                    d_rowPtr, d_colIdx, d_values,
                    d_b, d_x,
                    5000, 1e-10);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto tS1 = Clock::now();
    solveMs = std::chrono::duration<double,std::milli>(tS1 - tS0).count();

    CUDA_CHECK(cudaMemcpy(x_reordered.data(), d_x,
                          nDOF * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // -------------------- Step 9: Map back to original order -------
    std::vector<int> inv_perm(nDOF);
    for (int newIdx = 0; newIdx < nDOF; ++newIdx) {
        int oldIdx = perm[newIdx];
        inv_perm[oldIdx] = newIdx;
    }

    std::vector<double> x_original(nDOF);
    for (int oldIdx = 0; oldIdx < nDOF; ++oldIdx)
        x_original[oldIdx] = x_reordered[inv_perm[oldIdx]];

    // Commented out: Save solution vectors
    // {
    //     std::ofstream out(methodFolder + "/" +
    //                       meshBase + "__solution_vector_reordered.txt");
    //     out << "# Solution vector (reordered space, method=" << methodTag << ")\n";
    //     for (double v : x_reordered) out << v << "\n";
    //     std::cout << "Saved reordered solution.\n";
    // }

    // {
    //     std::ofstream out(methodFolder + "/" +
    //                       meshBase + "__solution_vector_original.txt");
    //     out << "# Solution vector (original ordering, method=" << methodTag << ")\n";
    //     for (double v : x_original) out << v << "\n";
    //     std::cout << "Saved original-order solution.\n";
    // }

    // -------------------- Step 10: Save timing results -------------
    {
        std::string timingFile;
        if (runId.empty()) {
            // Default: single results file (backward compatible)
            timingFile = methodFolder + "/" + meshBase + "__" + methodTag + "_results.txt";
        } else {
            // With run ID: unique filename for each run
            timingFile = methodFolder + "/" + meshBase + "__" + methodTag + "_run" + runId + "_results.txt";
        }
        std::ofstream tf(timingFile);
        tf << "# Timing results for mesh=" << meshBase
           << ", method=" << methodTag;
        if (!runId.empty()) {
            tf << ", run=" << runId;
        }
        tf << "\n";
        tf << "CPU_Assembly_ms " << cpuAssemblyMs << "\n";
        tf << "Reordering_ms "   << reorderMs     << "\n";
        tf << "HostToDevice_ms " << h2dMs         << "\n";
        tf << "GPU_Solve_ms "    << solveMs       << "\n";
        tf.close();
        std::cout << "Timing results saved to " << timingFile << "\n";
    }

    // -------------------- Cleanup ----------------------------------
    cudaFree(d_nodes_x); cudaFree(d_nodes_y); cudaFree(d_nodes_z);
    cudaFree(d_elem_conn); cudaFree(d_elemKe);
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
    cudaFree(d_b); cudaFree(d_x);

    std::cout << "Done.\n";
    return 0;
}
