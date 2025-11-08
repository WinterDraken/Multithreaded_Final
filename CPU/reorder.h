#ifndef REORDER_H
#define REORDER_H

#include <vector>

std::vector<int> reverseCuthillMcKee(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind);


void reorderCSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<int>& perm,
    std::vector<int>& new_row_ptr,
    std::vector<int>& new_col_ind,
    std::vector<double>& new_values);

#endif // REORDER_H
