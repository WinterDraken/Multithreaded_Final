// CPU/reorder.h
#pragma once
#include <vector>
#include <string>

enum class ReorderMethod {
    NONE,
    RCM,
    AMD,
    METIS
};

// Parse method string ("rcm", "amd", "metis", "none")
ReorderMethod parseReorderMethod(const std::string& s);

// Convert method to lowercase tag used in filenames
std::string methodToString(ReorderMethod m);

// RCM
std::vector<int> reverseCuthillMcKee(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind);

// Compute permutation (perm[new_index] = old_index)
std::vector<int> computePermutation(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    ReorderMethod method);

// Apply permutation simultaneously to rows and columns of CSR
void reorderCSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<int>& perm,       // perm[new] = old
    std::vector<int>& new_row_ptr,
    std::vector<int>& new_col_ind,
    std::vector<double>& new_values);
