#include "reorder.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <tuple>
#include <queue>
#include <climits>
#include <cctype>

// --------------------- Helper: lowercase -----------------------
static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

// --------------------- Method parsing ---------------------------
ReorderMethod parseReorderMethod(const std::string& sIn) {
    std::string s = toLower(sIn);
    if (s == "none") return ReorderMethod::NONE;
    if (s == "rcm")  return ReorderMethod::RCM;
    if (s == "amd")  return ReorderMethod::AMD;

    std::cerr << "Warning: unknown reordering method '" << sIn
              << "', defaulting to RCM.\n";
    return ReorderMethod::RCM;
}

std::string methodToString(ReorderMethod m) {
    switch (m) {
        case ReorderMethod::NONE: return "none";
        case ReorderMethod::RCM:  return "rcm";
        case ReorderMethod::AMD:  return "amd";
        default:                  return "unknown";
    }
}

// --------------------- RCM (your original) ----------------------
std::vector<int> reverseCuthillMcKee(
    int n, const std::vector<int>& row_ptr, const std::vector<int>& col_ind)
{
    std::vector<int> degree(n);
    for (int i = 0; i < n; ++i)
        degree[i] = row_ptr[i + 1] - row_ptr[i];

    std::vector<int> visited(n, 0);
    std::vector<int> order; order.reserve(n);

    auto bfs_component = [&](int start) {
        std::queue<int> q;
        q.push(start);
        visited[start] = 1;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            order.push_back(v);

            // collect unvisited neighbors
            std::vector<int> nbrs;
            for (int k = row_ptr[v]; k < row_ptr[v + 1]; ++k) {
                int u = col_ind[k];
                if (!visited[u]) {
                    visited[u] = 1;
                    nbrs.push_back(u);
                }
            }
            std::sort(nbrs.begin(), nbrs.end(),
                 [&](int a, int b) { return degree[a] < degree[b]; });
            for (int u : nbrs) q.push(u);
        }
    };

    // process each component
    while ((int)order.size() < n) {
        int start = -1, min_deg = INT_MAX;
        for (int i = 0; i < n; ++i)
            if (!visited[i] && degree[i] < min_deg)
                min_deg = degree[i], start = i;
        if (start == -1) break;
        bfs_component(start);
    }

    std::reverse(order.begin(), order.end()); // RCM
    // order[new] = old
    return order;
}

// --------------------- Approximate AMD --------------------------
//
// Very simple AMD-style heuristic:
//   - Build symmetric neighbor list
//   - degrees[i] = number of neighbors
//   - at each step, pick active vertex with smallest current degree
//   - approximate fill-in by increasing neighbor degrees when a vertex
//     is eliminated (without explicitly modifying the graph)
//
// This is not full AMD but captures the idea that eliminating high-degree
// nodes increases fill.
//
// Returns perm[new] = old
// ----------------------------------------------------------------
std::vector<int> approximateAMD(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind)
{
    // Build symmetric adjacency (neighbors[i] is list of neighbors of i)
    std::vector<std::vector<int>> neighbors(n);
    neighbors.assign(n, {});
    neighbors.reserve(n);

    for (int i = 0; i < n; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            int j = col_ind[k];
            if (j == i) continue; // ignore diagonal
            neighbors[i].push_back(j);
            neighbors[j].push_back(i);
        }
    }

    // Sort & unique neighbor lists
    for (int i = 0; i < n; ++i) {
        auto &nb = neighbors[i];
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }

    std::vector<int> degree(n);
    for (int i = 0; i < n; ++i)
        degree[i] = (int)neighbors[i].size();

    std::vector<char> active(n, 1);
    std::vector<int> perm; perm.reserve(n); // perm[new] = old

    for (int newIdx = 0; newIdx < n; ++newIdx) {
        int best = -1;
        int bestDeg = INT_MAX;

        // pick active vertex with smallest current degree
        for (int i = 0; i < n; ++i) {
            if (!active[i]) continue;
            if (degree[i] < bestDeg) {
                bestDeg = degree[i];
                best = i;
            }
        }

        if (best == -1) {
            // fallback: if something goes wrong, just fill remaining
            for (int i = 0; i < n; ++i) {
                if (active[i]) {
                    perm.push_back(i);
                    active[i] = 0;
                }
            }
            break;
        }

        int v = best;
        perm.push_back(v);
        active[v] = 0;

        // Approximate fill: neighbors of v will see more connections,
        // so increase their degrees.
        int activeNbrCount = 0;
        for (int u : neighbors[v]) {
            if (active[u]) ++activeNbrCount;
        }
        for (int u : neighbors[v]) {
            if (active[u]) degree[u] += std::max(0, activeNbrCount - 1);
        }
    }

    return perm; // perm[new] = old
}

// --------------------- Dispatch -------------------------------
std::vector<int> computePermutation(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    ReorderMethod method)
{
    switch (method) {
        case ReorderMethod::NONE: {
            std::vector<int> id(n);
            for (int i = 0; i < n; ++i) id[i] = i;
            return id;
        }
        case ReorderMethod::RCM:
            return reverseCuthillMcKee(n, row_ptr, col_ind);
        case ReorderMethod::AMD:
            return approximateAMD(n, row_ptr, col_ind);
        default: {
            std::vector<int> id(n);
            for (int i = 0; i < n; ++i) id[i] = i;
            return id;
        }
    }
}

// --------------------- CSR Reordering (your original) ----------
void reorderCSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<int>& perm,  // perm[new] = old
    std::vector<int>& new_row_ptr,
    std::vector<int>& new_col_ind,
    std::vector<double>& new_values)
{
    // Build inverse permutation: inv_perm[old] = new
    std::vector<int> inv_perm(n);
    for (int newIdx = 0; newIdx < n; ++newIdx) {
        int oldIdx = perm[newIdx];
        inv_perm[oldIdx] = newIdx;
    }

    // Convert CSR to triplets with perm applied
    std::vector<std::tuple<int,int,double>> triplets;
    triplets.reserve(col_ind.size());

    for (int oldRow = 0; oldRow < n; ++oldRow) {
        for (int k = row_ptr[oldRow]; k < row_ptr[oldRow + 1]; ++k) {
            int oldCol = col_ind[k];
            double v   = values[k];
            int newRow = inv_perm[oldRow];
            int newCol = inv_perm[oldCol];
            triplets.emplace_back(newRow, newCol, v);
        }
    }

    // Sort by (row, col)
    std::sort(triplets.begin(), triplets.end(),
              [](const std::tuple<int,int,double>& a,
                 const std::tuple<int,int,double>& b) {
                  if (std::get<0>(a) != std::get<0>(b))
                      return std::get<0>(a) < std::get<0>(b);
                  return std::get<1>(a) < std::get<1>(b);
              });

    // Rebuild CSR
    new_row_ptr.assign(n + 1, 0);
    new_col_ind.resize(triplets.size());
    new_values.resize(triplets.size());

    int current_row = 0;
    new_row_ptr[0] = 0;
    for (size_t k = 0; k < triplets.size(); ++k) {
        int r = std::get<0>(triplets[k]);
        int c = std::get<1>(triplets[k]);
        double v = std::get<2>(triplets[k]);

        new_col_ind[k] = c;
        new_values[k]  = v;

        while (current_row < r) {
            ++current_row;
            new_row_ptr[current_row] = static_cast<int>(k);
        }
    }
    for (int i = current_row + 1; i <= n; ++i)
        new_row_ptr[i] = static_cast<int>(triplets.size());
}
