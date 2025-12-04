#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>
#include <numeric>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <cuda_runtime.h>
#include <cmath>    // for std::abs

#include "CPU/mesh_parser.h"
#include "CPU/csr_builder.h"
#include "CPU/reorder.h"
#include "GPU/localSolve.h"
#include "GPU/globalAsm.h"
#include "GPU/gpu_solve_csr.h"
#include "GPU/gpu_cholesky_solver.h"

// CUDA error check
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);} \
    } while(0)

using Clock = std::chrono::high_resolution_clock;

// Create directory if not exists
static void makeDir(const std::string &p) {
    struct stat st = {0};
    if (stat(p.c_str(), &st) == -1) mkdir(p.c_str(), 0755);
}

// Simple bandwidth metric: max |col - row|
static int compute_bandwidth(int n,
                             const std::vector<int>& rowPtr,
                             const std::vector<int>& colIdx)
{
    int bw = 0;
    for (int i = 0; i < n; ++i) {
        for (int k = rowPtr[i]; k < rowPtr[i+1]; ++k) {
            int j = colIdx[k];
            int d = std::abs(j - i);
            if (d > bw) bw = d;
        }
    }
    return bw;
}

// Convenience wrapper for your CSR reordering
static void apply_permutation_to_csr(
    int n,
    const std::vector<int>& rowPtr,
    const std::vector<int>& colIdx,
    const std::vector<double>& vals,
    const std::vector<int>& perm,      // perm[new] = old
    std::vector<int>& outRow,
    std::vector<int>& outCol,
    std::vector<double>& outVal)
{
    reorderCSR(n, rowPtr, colIdx, vals, perm, outRow, outCol, outVal);
}

int main(int argc, char* argv[]) 
{
    // -------------------- Parse Args --------------------
    std::string meshFile = "CPU/bracket_3d.msh";
    std::string reorderMethodStr = "none";
    std::string solverStr = "cg";   // cg or chol
    bool verbose = false;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose") verbose = true;
        else if (a.rfind("--solver=",0) == 0) solverStr = a.substr(9);
        else if (positional == 0) { meshFile = a; positional++; }
        else if (positional == 1) { reorderMethodStr = a; positional++; }
    }

    ReorderMethod reorderMethod = parseReorderMethod(reorderMethodStr);
    std::string methodTag = methodToString(reorderMethod);

    // Extract mesh base name
    std::string meshBase;
    {
        size_t slash = meshFile.find_last_of("/\\");
        std::string tmp = (slash == std::string::npos) ? meshFile : meshFile.substr(slash + 1);
        size_t dot = tmp.find_last_of(".");
        meshBase = (dot == std::string::npos) ? tmp : tmp.substr(0, dot);
    }

    // Output directories
    makeDir("results");
    makeDir("results/" + methodTag);

    // Timestamp for the result file only
    auto tnow = std::chrono::system_clock::now();
    auto tnow_time = std::chrono::system_clock::to_time_t(tnow);
    std::tm ti;
#ifdef _WIN32
    localtime_s(&ti, &tnow_time);
#else
    localtime_r(&tnow_time, &ti);
#endif
    std::stringstream ts;
    ts << std::put_time(&ti, "%Y%m%d_%H%M%S");

    std::string resultFile =
        "results/" + methodTag + "/" +
        meshBase + "__" + methodTag + "__" + solverStr + "__" + ts.str() + ".txt";

    // ---------------- Mesh Parsing ----------------
    std::vector<Node> nodes;
    std::vector<Element> elements;
    if (!parseMeshFile(meshFile, nodes, elements)) {
        std::cerr << "Mesh parse failed.\n";
        return 1;
    }

    // Filter Tet4
    std::vector<Element> tet4;
    for (auto &e : elements)
        if (e.type == 4 && e.node_ids.size() == 4)
            tet4.push_back(e);

    int nNodes = nodes.size();
    int nElem  = tet4.size();

    // Build DOF map
    std::map<int,int> idToIdx;
    for (int i = 0; i < nNodes; i++)
        idToIdx[nodes[i].id] = i;

    std::vector<double> X(nNodes), Y(nNodes), Z(nNodes);
    for (auto &n : nodes) {
        int i = idToIdx[n.id];
        X[i] = n.x; Y[i] = n.y; Z[i] = n.z;
    }

    std::vector<int> conn(nElem * 4);
    for (int e = 0; e < nElem; e++)
        for (int i = 0; i < 4; i++)
            conn[e*4+i] = idToIdx[tet4[e].node_ids[i]];

    // ---------------- CSR Pattern ----------------
    int ndof_per_node = 3;
    std::vector<int> rowPtr, colIdx;
    int nDOF = 0;
    int nnz = buildCSRPattern(tet4, nodes, ndof_per_node, true, rowPtr, colIdx, nDOF);

    // ---------------- GPU Memory ----------------
    double *dX, *dY, *dZ;
    int *dConn;
    double *d_elemKe;
    int *d_rowPtr, *d_colIdx;
    double *d_vals;

    CUDA_CHECK(cudaMalloc(&dX, nNodes*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dY, nNodes*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dZ, nNodes*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dConn, nElem*4*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_elemKe, nElem*144*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (nDOF+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz*sizeof(double)));

    cudaMemcpy(dX, X.data(), nNodes*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y.data(), nNodes*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dZ, Z.data(), nNodes*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dConn, conn.data(), nElem*4*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, rowPtr.data(), (nDOF+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_vals, 0, nnz*sizeof(double));

    // ---------------- GPU Local + Global Assembly ----------------
    launchLocalKe_Tet4_3D(dX, dY, dZ, dConn, 200000, 0.3, d_elemKe, nElem);
    cudaDeviceSynchronize();

    launch_assembleCSR_atomic(d_elemKe, dConn, d_rowPtr, d_colIdx, d_vals,
                              nElem, 4, ndof_per_node, true);
    cudaDeviceSynchronize();

    // Copy matrix values back for CPU-side work
    std::vector<double> vals(nnz);
    cudaMemcpy(vals.data(), d_vals, nnz*sizeof(double), cudaMemcpyDeviceToHost);

    // ---------------- Bandwidth BEFORE reorder -------------------
    int bw_orig = compute_bandwidth(nDOF, rowPtr, colIdx);

    // ---------------- Reordering (timed) -------------------------
    double reorder_ms = 0.0;

    // perm_improve[new] = old
    std::vector<int> perm_improve(nDOF);
    std::iota(perm_improve.begin(), perm_improve.end(), 0);

    // final CSR to send to GPU
    std::vector<int> rRow = rowPtr;
    std::vector<int> rCol = colIdx;
    std::vector<double> rVal = vals;

    if (reorderMethod != ReorderMethod::NONE) {
        auto t0 = Clock::now();

        if (reorderMethod == ReorderMethod::AMD) {
            // *** INTERPRET "amd" AS GEOMETRIC ORDERING ***
            // Sort nodes by (z, y, x), then map DOFs accordingly.
            struct NodeKey {
                int    oldNode;
                double x, y, z;
            };
            std::vector<NodeKey> keys(nNodes);
            for (int i = 0; i < nNodes; ++i) {
                keys[i] = { i, X[i], Y[i], Z[i] };
            }
            std::sort(keys.begin(), keys.end(),
                      [](const NodeKey& a, const NodeKey& b){
                          if (a.z != b.z) return a.z < b.z;
                          if (a.y != b.y) return a.y < b.y;
                          return a.x < b.x;
                      });

            perm_improve.assign(nDOF, -1);
            int newNode = 0;
            for (const auto& k : keys) {
                int oldNode = k.oldNode;
                for (int d = 0; d < ndof_per_node; ++d) {
                    int newDof = newNode * ndof_per_node + d;
                    int oldDof = oldNode * ndof_per_node + d;
                    perm_improve[newDof] = oldDof;
                }
                ++newNode;
            }
        } else {
            // RCM / COLAMD etc use the graph-based permutation
            perm_improve = computePermutation(nDOF, rowPtr, colIdx, reorderMethod);
        }

        apply_permutation_to_csr(nDOF, rowPtr, colIdx, vals,
                                 perm_improve, rRow, rCol, rVal);

        auto t1 = Clock::now();
        reorder_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    }

    // ---------------- Diagonal regularization (make SPD) --------
    // Add a tiny lambda*I so Cholesky always factors fully and
    // we can meaningfully compare factor times across permutations.
    const double lambda = 1e-6;
    for (int i = 0; i < nDOF; ++i) {
        for (int k = rRow[i]; k < rRow[i+1]; ++k) {
            if (rCol[k] == i) {
                rVal[k] += lambda;
                break;
            }
        }
    }

    // Bandwidth AFTER reorder (and regularization)
    int bw_reord = compute_bandwidth(nDOF, rRow, rCol);

    // Upload (possibly reordered) CSR to GPU
    cudaMemcpy(d_rowPtr, rRow.data(), (nDOF+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, rCol.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals,  rVal.data(),  nnz*sizeof(double), cudaMemcpyHostToDevice);

    // ---------------- RHS ----------------
    std::vector<double> b(nDOF, 1.0);

    // overall permutation from original numbering -> final CSR numbering
    std::vector<int> perm_total(nDOF);
    perm_total = perm_improve; // no extra permutations elsewhere

    // b_perm[new] = b[old]
    std::vector<double> b_perm(nDOF);
    for (int i = 0; i < nDOF; i++)
        b_perm[i] = b[ perm_total[i] ];

    double *d_b, *d_x;
    cudaMalloc(&d_b, nDOF*sizeof(double));
    cudaMalloc(&d_x, nDOF*sizeof(double));
    cudaMemcpy(d_b, b_perm.data(), nDOF*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, nDOF*sizeof(double));

    // ---------------- SOLVER SELECT ----------------
    int cg_iters = 0;
    int singularity = -1;
    double chol_ms = 0.0;
    double solve_ms = 0.0;

    auto tSolve0 = Clock::now();

    if (solverStr == "chol") {
        gpu_direct_cholesky(
            nDOF, nnz,
            d_rowPtr, d_colIdx, d_vals,
            d_b, d_x,
            singularity, chol_ms
        );
        cudaDeviceSynchronize();
    }
    else {
        solveCG_CSR_GPU(
            nDOF, nnz,
            d_rowPtr, d_colIdx, d_vals,
            d_b, d_x,
            5000, 1e-10,
            cg_iters
        );
        cudaDeviceSynchronize();
    }

    auto tSolve1 = Clock::now();
    solve_ms = std::chrono::duration<double,std::milli>(tSolve1 - tSolve0).count();

    // ---------------- Save Solution Vector (Displacements) ----------------
    std::vector<double> x_solution(nDOF);
    cudaMemcpy(x_solution.data(), d_x, nDOF*sizeof(double), cudaMemcpyDeviceToHost);
    
    // Apply inverse permutation to get solution in original ordering
    std::vector<int> inv_perm(nDOF);
    for (int i = 0; i < nDOF; i++) {
        inv_perm[perm_total[i]] = i;
    }
    std::vector<double> x_original(nDOF);
    for (int i = 0; i < nDOF; i++) {
        x_original[i] = x_solution[inv_perm[i]];
    }
    
    // Save solution vector
    std::string solutionFile = "results/" + methodTag + "/" + meshBase + "__" + methodTag + "_solution.txt";
    std::ofstream solf(solutionFile);
    solf << "# Solution vector (displacements) for mesh: " << meshBase << ", method: " << methodTag << "\n";
    solf << "# Format: DOF_index displacement_value\n";
    solf << "# nDOF: " << nDOF << "\n";
    solf << "# nNodes: " << nNodes << "\n";
    solf << "# ndof_per_node: " << ndof_per_node << "\n";
    for (int i = 0; i < nDOF; i++) {
        solf << i << " " << std::scientific << std::setprecision(15) << x_original[i] << "\n";
    }
    solf.close();
    
    // ---------------- Save Sparse Matrix Patterns ----------------
    // Save original matrix pattern (before reordering)
    std::string matrixOrigFile = "results/" + methodTag + "/" + meshBase + "__" + methodTag + "_matrix_original.txt";
    std::ofstream matorig(matrixOrigFile);
    matorig << "# Sparse matrix pattern (original, before reordering)\n";
    matorig << "# Format: rowPtr array, then colIdx array\n";
    matorig << "# nDOF: " << nDOF << "\n";
    matorig << "# nnz: " << nnz << "\n";
    matorig << "# rowPtr (length " << (nDOF+1) << "):\n";
    for (size_t i = 0; i < rowPtr.size(); i++) {
        matorig << rowPtr[i];
        if (i < rowPtr.size() - 1) matorig << " ";
    }
    matorig << "\n# colIdx (length " << nnz << "):\n";
    for (int i = 0; i < nnz; i++) {
        matorig << colIdx[i];
        if (i < nnz - 1) matorig << " ";
    }
    matorig << "\n";
    matorig.close();
    
    // Save reordered matrix pattern
    std::string matrixReordFile = "results/" + methodTag + "/" + meshBase + "__" + methodTag + "_matrix_reordered.txt";
    std::ofstream matreord(matrixReordFile);
    matreord << "# Sparse matrix pattern (after reordering)\n";
    matreord << "# Format: rowPtr array, then colIdx array\n";
    matreord << "# nDOF: " << nDOF << "\n";
    matreord << "# nnz: " << nnz << "\n";
    matreord << "# rowPtr (length " << (nDOF+1) << "):\n";
    for (size_t i = 0; i < rRow.size(); i++) {
        matreord << rRow[i];
        if (i < rRow.size() - 1) matreord << " ";
    }
    matreord << "\n# colIdx (length " << nnz << "):\n";
    for (int i = 0; i < nnz; i++) {
        matreord << rCol[i];
        if (i < nnz - 1) matreord << " ";
    }
    matreord << "\n";
    matreord.close();
    
    // ---------------- Save Node Coordinates for Visualization ----------------
    std::string nodesFile = "results/" + methodTag + "/" + meshBase + "__" + methodTag + "_nodes.txt";
    std::ofstream nodesf(nodesFile);
    nodesf << "# Node coordinates for mesh: " << meshBase << "\n";
    nodesf << "# Format: node_index x y z\n";
    nodesf << "# nNodes: " << nNodes << "\n";
    for (int i = 0; i < nNodes; i++) {
        nodesf << i << " " << std::scientific << std::setprecision(15) 
               << X[i] << " " << Y[i] << " " << Z[i] << "\n";
    }
    nodesf.close();
    
    // ---------------- Save Element Connectivity for Visualization ----------------
    std::string elementsFile = "results/" + methodTag + "/" + meshBase + "__" + methodTag + "_elements.txt";
    std::ofstream elemf(elementsFile);
    elemf << "# Element connectivity (tetrahedral elements)\n";
    elemf << "# Format: element_index node0 node1 node2 node3\n";
    elemf << "# nElem: " << nElem << "\n";
    for (int e = 0; e < nElem; e++) {
        elemf << e;
        for (int i = 0; i < 4; i++) {
            elemf << " " << conn[e*4 + i];
        }
        elemf << "\n";
    }
    elemf.close();

    // ---------------- Save Timing ----------------
    std::ofstream tf(resultFile);
    tf << "# FEM solve results\n";
    tf << "mesh " << meshFile << "\n";
    tf << "reorder " << methodTag << "\n";
    tf << "solver  " << solverStr << "\n";

    tf << "Bandwidth_orig "  << bw_orig  << "\n";
    tf << "Bandwidth_reord " << bw_reord << "\n";
    tf << "Reorder_ms "      << reorder_ms << "\n";

    if (solverStr == "cg") {
        tf << "CG_iters " << cg_iters << "\n";
        tf << "Solve_ms " << solve_ms << "\n";
    } else {
        tf << "Cholesky_singularity " << singularity << "\n";
        tf << "Cholesky_factor_ms "  << chol_ms << "\n";
        tf << "Solve_ms " << solve_ms << "\n";
    }

    tf.close();

    std::cout << "Wrote: " << resultFile << "\n";
    std::cout << "Wrote: " << solutionFile << "\n";
    std::cout << "Wrote: " << matrixOrigFile << "\n";
    std::cout << "Wrote: " << matrixReordFile << "\n";
    std::cout << "Wrote: " << nodesFile << "\n";
    std::cout << "Wrote: " << elementsFile << "\n";

    // Cleanup GPU memory
    cudaFree(dX); cudaFree(dY); cudaFree(dZ);
    cudaFree(dConn); cudaFree(d_elemKe);
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_vals);
    cudaFree(d_b); cudaFree(d_x);

    return 0;
}
