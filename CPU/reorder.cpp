#include "reorder.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <tuple>

std::vector<int> reverseCuthillMcKee(
    int n, const vector<int>& row_ptr, const vector<int>& col_ind)
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
            sort(nbrs.begin(), nbrs.end(),
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

    reverse(order.begin(), order.end());
    return order;
}

// Apply RCM permutation to both rows and columns of CSR matrix
// Inputs:
//  - n: number of rows/cols
//  - row_ptr, col_ind, values: CSR representation
//  - perm: permutation vector (order[i] = old index now becomes new index)
// Outputs:
//  - new_row_ptr, new_col_ind, new_values (fully reordered CSR)
void reorderCSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<int>& perm,
    std::vector<int>& new_row_ptr,
    std::vector<int>& new_col_ind,
    std::vector<double>& new_values)
{
    // --- Step 1: Build inverse permutation ---
    std::vector<int> inv_perm(n);
    for (int i = 0; i < n; ++i)
        inv_perm[perm[i]] = i;

    // --- Step 2: Convert CSR to triplets (row, col, val) ---
    std::vector<std::tuple<int,int,double>> triplets;
    triplets.reserve(col_ind.size());
    for (int i = 0; i < n; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
            int j = col_ind[k];
            double v = values[k];
            // Apply permutation: (i,j) → (perm[i], perm[j])
            triplets.emplace_back(inv_perm[i], inv_perm[j], v);
        }
    }

    // --- Step 3: Sort triplets by (row,col) ---
    std::sort(triplets.begin(), triplets.end(),
              [](auto& a, auto& b) {
                  if (std::get<0>(a) != std::get<0>(b))
                      return std::get<0>(a) < std::get<0>(b);
                  return std::get<1>(a) < std::get<1>(b);
              });

    // --- Step 4: Rebuild CSR ---
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
        new_values[k] = v;
        while (current_row < r) {
            ++current_row;
            new_row_ptr[current_row] = k;
        }
    }
    // Close last row
    for (int i = current_row + 1; i <= n; ++i)
        new_row_ptr[i] = triplets.size();
}
