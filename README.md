# Demo 1 (11/8)
## Overview of the Project
GPU-accelerated FEM solver for 3D linear elasticity problems. The solver assembles a global stiffness matrix from a finite element mesh using CUDA-enabled GPUs for parallel computation.

### Key Concepts

#### 1. Local Stiffness Matrix (Ke)

A **local stiffness matrix** describes the relationship between forces and displacements for a single finite element. For each tetrahedral element (Tet4) in the mesh:

- Each element has 4 nodes, and each node has 3 degrees of freedom (x, y, z displacements)
- This results in a 12×12 matrix (4 nodes × 3 DOFs = 12 DOFs per element)
- The matrix captures how forces at one node affect displacements at all nodes within that element
- It's computed using the element's geometry, material properties (Young's modulus E, Poisson's ratio ν), and the strain-displacement relationship

As of now, the local stiffness matrices are computed in parallel on the GPU, with each thread handling one element.

#### 2. Global Stiffness Matrix (K)

The **global stiffness matrix** represents the overall structural behavior of the entire mesh. It's assembled by:

- Combining interactions from all local element stiffness matrices
    - Mapping element-local degrees of freedom to global degrees of freedom
- Summing contributions when multiple elements share the same nodes

The global matrix is typically very large (thousands to millions of DOFs) and sparse (most entries are zero) because nodes only interact with their neighbors through shared elements.

#### 3. CSR (Compressed Sparse Row) Format

**CSR (Compressed Sparse Row)** is an efficient storage format for sparse matrices. Instead of storing all zeros, CSR stores only non-zero values using three arrays:

- **`rowPtr`**: Points to where each row's data starts in the `colIdx` and `values` arrays
- **`colIdx`**: Stores the column indices of non-zero entries
- **`values`**: Stores the actual non-zero matrix values

For example, a matrix with mostly zeros might have millions of rows but only store a few non-zero entries per row, dramatically reducing memory requirements and enabling efficient sparse matrix operations.

## Progress
- (CPU) Parse .msh to generate data structures needed for GPU local solve
- (GPU) Compute local stiffness matrix
- (GPU) Populate global (unordered, sprase) matrix
- (CPU) Reorder -- using CSR -- the global matrix

## Next Steps
- (GPU) Solve the reordered matrix
This will be the end of the first design cycle. Next steps will optimize performance exploiting GPU (and potentially CPU, trying to think of the ways to make it work while GPU does computations):
- Generate baseline data (without reordering perf)
- Studies on different reordering strategies:
    - Different configuration of threads/blocks can benefit different methods
- Studies on using different GPU architecture?
    - rtx (frontera)
    - a100 (ls6)
    - h100 (ls6)