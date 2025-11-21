#pragma once
#include <vector>
#include <string>

// Which reordering method is used
enum class ReorderMethod {
    NONE,
    RCM,
    AMD
};

// Parse method string ("none", "rcm", "amd"), default = RCM on unknown
ReorderMethod parseReorderMethod(const std::string& s);

// Convert method enum to tag string ("none", "rcm", "amd")
std::string methodToString(ReorderMethod m);

// Reverse Cuthill–McKee permutation.
// Returns perm[new_index] = old_index
std::vector<int> reverseCuthillMcKee(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind);

// Approximate AMD permutation (simple minimum-degree heuristic with
// crude fill-in degree updates, no external libraries).
// Returns perm[new_index] = old_index
std::vector<int> approximateAMD(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind);

// Dispatch to the correct permutation generator given a method.
// Returns perm[new_index] = old_index
std::vector<int> computePermutation(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    ReorderMethod method);

// Apply a DOF permutation to CSR matrix.
// Inputs:
//   n, row_ptr, col_ind, values: original CSR
//   perm: perm[new] = old
// Outputs:
//   new_row_ptr, new_col_ind, new_values: reordered CSR
void reorderCSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<int>& perm,
    std::vector<int>& new_row_ptr,
    std::vector<int>& new_col_ind,
    std::vector<double>& new_values);
