#include "reorder.h"

#include <algorithm>
#include <iostream>

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