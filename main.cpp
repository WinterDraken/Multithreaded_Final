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
#include <cmath>

#include "CPU/mesh_parser.h"
#include "CPU/csr_builder.h"
#include "CPU/reorder.h"
#include "GPU/localSolve.h"
#include "GPU/globalAsm.h"
#include "GPU/gpu_solve_csr.h"
#include "GPU/gpu_cholesky_solver.h"

// CUDA error check
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err__) << std::endl; \
            std::exit(1); \
        } \
    } while (0)

using Clock = std::chrono::high_resolution_clock;

// Create directory if not exists
static void makeDir(const std::string &p) {
    struct stat st = {0};
    if (stat(p.c_str(), &st) == -1) {
        mkdir(p.c_str(), 0755);
    }
}

// Simple bandwidth metric: max |col - row|
static int compute_bandwidth(int n,
                             const std::vector<int>& rowPtr,
                             const std::vector<int>& colIdx)
{
    int bw = 0;
    for (int i = 0; i < n; ++i) {
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
            int j = colIdx[k];
            int d = std::abs(j - i);
            if (d > bw) bw = d;
        }
    }
    return bw;
}

int main(int argc, char* argv[])
{
    // -------------------- Parse Args --------------------
    std::string meshFile = "CPU/bracket_3d.msh";
    std::string reorderMethodStr = "none";
    std::string solverStr = "cg";
    bool verbose = false;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else if (a.rfind("--solver=", 0) == 0) {
            solverStr = a.substr(9);
        } else if (positional == 0) {
            meshFile = a;
            positional++;
        } else if (positional == 1) {
            reorderMethodStr = a;
            positional++;
        }
    }

    ReorderMethod reorderMethod = parseReorderMethod(reorderMethodStr);
    std::string methodTag = methodToString(reorderMethod);

    std::string solverNorm = solverStr;
    std::transform(solverNorm.begin(), solverNorm.end(), solverNorm.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    bool use_direct = (solverNorm == "chol" ||
                       solverNorm == "lu"   ||
                       solverNorm == "direct");

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

    // Timestamp for result filename
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
        meshBase + "__" + methodTag + "__" + solverNorm + "__" + ts.str() + ".txt";
    
    // Solution vector file (without timestamp, for visualization)
    std::string solutionFile =
        "results/" + methodTag + "/" +
        meshBase + "__" + methodTag + "_solution.txt";

    // ---------------- Mesh Parsing ----------------
    std::vector<Node> nodes;
    std::vector<Element> elements;
    if (!parseMeshFile(meshFile, nodes, elements)) {
        std::cerr << "Mesh parse failed.\n";
        return 1;
    }

    // Filter Tet4
    std::vector<Element> tet4;
    tet4.reserve(elements.size());
    for (auto &e : elements) {
        if (e.type == 4 && e.node_ids.size() == 4)
            tet4.push_back(e);
    }

    int nNodes = static_cast<int>(nodes.size());
    int nElem  = static_cast<int>(tet4.size());
    int ndof_per_node = 3;

    if (verbose) {
        std::cout << "Mesh: " << meshFile << "\n";
        std::cout << "  Nodes: " << nNodes << "\n";
        std::cout << "  Tet4 elements: " << nElem << "\n";
    }

    // Map GMSH node IDs to contiguous indices
    std::map<int,int> idToIdx;
    for (int i = 0; i < nNodes; i++) {
        idToIdx[nodes[i].id] = i;
    }

    // Coordinates
    std::vector<double> X(nNodes), Y(nNodes), Z(nNodes);
    for (auto &n : nodes) {
        int idx = idToIdx[n.id];
        X[idx] = n.x;
        Y[idx] = n.y;
        Z[idx] = n.z;
    }

    // Connectivity (node indices)
    std::vector<int> conn(nElem * 4);
    for (int e = 0; e < nElem; e++) {
        for (int i = 0; i < 4; i++) {
            conn[e * 4 + i] = idToIdx[tet4[e].node_ids[i]];
        }
    }

    // ---------------- Build CSR Pattern (original ordering) ----------------
    std::vector<int> rowPtr, colIdx;
    int nDOF = 0;

    int nnz = buildCSRPattern(
        tet4, nodes, ndof_per_node,
        /*symmetric_upper=*/false,
        rowPtr, colIdx, nDOF);

    int bw_orig = compute_bandwidth(nDOF, rowPtr, colIdx);

    // ---------------- Compute permutation ----------------
    std::vector<int> perm(nDOF);
    double reorder_ms = 0.0;
    auto tRe0 = Clock::now();

    perm = computePermutation(nDOF, rowPtr, colIdx, reorderMethod);

    auto tRe1 = Clock::now();
    reorder_ms = std::chrono::duration<double, std::milli>(tRe1 - tRe0).count();

    // ---------------- Build permuted CSR pattern ----------------
    std::vector<int> rRow, rCol;
    std::vector<double> dummyVals(nnz, 0.0), dummyValsOut;

    reorderCSR(nDOF, rowPtr, colIdx, dummyVals,
               perm, rRow, rCol, dummyValsOut);

    // Inverse permutation: old -> new
    std::vector<int> dofOld2New(nDOF);
    for (int newi = 0; newi < nDOF; ++newi) {
        int oldi = perm[newi];
        dofOld2New[oldi] = newi;
    }

    int bw_reord = compute_bandwidth(nDOF, rRow, rCol);
    int nnz_perm = static_cast<int>(rCol.size());

    if (verbose) {
        std::cout << "Bandwidth orig:  " << bw_orig  << "\n";
        std::cout << "Bandwidth reord: " << bw_reord << "\n";
        std::cout << "nnz (orig):  " << nnz << "\n";
        std::cout << "nnz (perm):  " << nnz_perm << "\n";
    }

    // ---------------- Allocate GPU Buffers ----------------
    double *dX = nullptr, *dY = nullptr, *dZ = nullptr;
    int *dConn = nullptr;
    double *d_elemKe = nullptr;
    int *d_rowPtr = nullptr, *d_colIdx_perm = nullptr;
    double *d_vals = nullptr;
    int *d_dofOld2New = nullptr;

    CUDA_CHECK(cudaMalloc(&dX, nNodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dY, nNodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dZ, nNodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dConn, nElem * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_elemKe, nElem * 144 * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_rowPtr, (nDOF + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx_perm, nnz_perm * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz_perm * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dofOld2New, nDOF * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dX, X.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, Y.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dZ, Z.data(), nNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dConn, conn.data(), nElem * 4 * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_rowPtr, rRow.data(), (nDOF + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx_perm, rCol.data(), nnz_perm * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vals, 0, nnz_perm * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_dofOld2New, dofOld2New.data(),
                          nDOF * sizeof(int), cudaMemcpyHostToDevice));

    // ---------------- GPU Local Ke ----------------
    launchLocalKe_Tet4_3D(dX, dY, dZ, dConn,
                          /*E*/200000.0, /*nu*/0.3,
                          d_elemKe, nElem);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- Global Assembly into permuted CSR ----------------
    launch_assembleCSR_atomic(
        d_elemKe, dConn,
        d_rowPtr, d_colIdx_perm, d_vals,
        d_dofOld2New,
        nElem, 4, ndof_per_node,
        /*symmetric_upper=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- Regularization (host-side) ----------------
    std::vector<double> rVal(nnz_perm);
    CUDA_CHECK(cudaMemcpy(rVal.data(), d_vals,
                          nnz_perm * sizeof(double),
                          cudaMemcpyDeviceToHost));

    const double lambda = 1e-6;
    for (int i = 0; i < nDOF; ++i) {
        for (int k = rRow[i]; k < rRow[i + 1]; ++k) {
            if (rCol[k] == i) {
                rVal[k] += lambda;
                break;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_vals, rVal.data(),
                          nnz_perm * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---------------- RHS ----------------
    std::vector<double> b(nDOF, 1.0);
    std::vector<double> b_perm(nDOF);
    for (int newi = 0; newi < nDOF; ++newi) {
        b_perm[newi] = b[perm[newi]];
    }

    double *d_b = nullptr, *d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b, nDOF * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, nDOF * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_b, b_perm.data(),
                          nDOF * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, nDOF * sizeof(double)));

    // ---------------- Free assembly-only buffers before direct solve ----------------
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dZ));
    CUDA_CHECK(cudaFree(dConn));
    CUDA_CHECK(cudaFree(d_elemKe));
    CUDA_CHECK(cudaFree(d_dofOld2New));
    dX = dY = dZ = nullptr;
    dConn = nullptr;
    d_elemKe = nullptr;
    d_dofOld2New = nullptr;

    // ---------------- SOLVE ----------------
    int cg_iters    = 0;   // for CG
    int singularity = -1;  // for Cholesky
    double chol_ms  = 0.0;
    double solve_ms = 0.0;

    auto tSolve0 = Clock::now();

    if (use_direct) {
        // Direct Cholesky (or LU) on GPU, using cuSOLVER.
        gpu_direct_cholesky(
            nDOF, nnz_perm,
            d_rowPtr, d_colIdx_perm, d_vals,
            d_b, d_x,
            singularity, chol_ms
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Conjugate Gradient on GPU with cuSPARSE/cuBLAS
        solveCG_CSR_GPU(
            nDOF, nnz_perm,
            d_rowPtr, d_colIdx_perm, d_vals,
            d_b, d_x,
            /*maxIters=*/5000,
            /*tol=*/1e-10,
            cg_iters
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto tSolve1 = Clock::now();
    solve_ms = std::chrono::duration<double, std::milli>(tSolve1 - tSolve0).count();
    

    // ---------------- Un-permute solution ----------------
    std::vector<double> x_perm(nDOF);
    CUDA_CHECK(cudaMemcpy(x_perm.data(), d_x,
                          nDOF * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::vector<double> x_orig(nDOF);
    for (int newi = 0; newi < nDOF; ++newi) {
        int oldi = perm[newi];
        x_orig[oldi] = x_perm[newi];
    }

    // ---------------- Save results ----------------
    std::ofstream tf(resultFile);
    tf << "# FEM solve results\n";
    tf << "mesh " << meshFile << "\n";
    tf << "reorder " << methodTag << "\n";
    tf << "solver  " << solverNorm << "\n";
    tf << "Bandwidth_orig "  << bw_orig  << "\n";
    tf << "Bandwidth_reord " << bw_reord << "\n";
    tf << "Reorder_ms "      << reorder_ms << "\n";

    if (use_direct) {
        tf << "Cholesky_singularity " << singularity << "\n";
        tf << "Cholesky_factor_ms "   << chol_ms << "\n";
        tf << "Solve_ms "             << solve_ms << "\n";
    } else {
        tf << "CG_iters " << cg_iters << "\n";  // currently -1 (not tracked)
        tf << "Solve_ms " << solve_ms << "\n";
    }

    tf.close();

    std::cout << "Wrote: " << resultFile << "\n";

    // ---------------- Save solution vector ----------------
    std::ofstream sf(solutionFile);
    sf << "# Solution vector (displacements) for mesh: " << meshBase << ", method: " << methodTag << "\n";
    sf << "# Format: DOF_index displacement_value\n";
    sf << "# nDOF: " << nDOF << "\n";
    sf << "# nNodes: " << nNodes << "\n";
    sf << "# ndof_per_node: " << ndof_per_node << "\n";
    for (int i = 0; i < nDOF; ++i) {
        sf << i << " " << std::scientific << std::setprecision(15) << x_orig[i] << "\n";
    }
    sf.close();

    std::cout << "Wrote: " << solutionFile << "\n";

    // Free remaining device buffers
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx_perm));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));

    return 0;
}
