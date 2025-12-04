# Visualization Guide for Demo

This guide explains how to generate and visualize the results for your FEM solver demo.

## Overview

The visualization system creates:
1. **Sparsity pattern plots** - Shows the structure of sparse matrices before and after reordering
2. **Displacement magnitude visualization** - Shows how the mesh deforms under load (correlates with stress)

## Step 1: Run the Solver

First, run your solver for each reordering method to generate the data files:

```bash
# Run for each method
./fem_assembly CPU/bracket_3d.msh none
./fem_assembly CPU/bracket_3d.msh rcm
./fem_assembly CPU/bracket_3d.msh amd
```

This will create files in `results/{method}/`:
- `{mesh}__{method}_matrix_original.txt` - Original sparse matrix pattern
- `{mesh}__{method}_matrix_reordered.txt` - Reordered sparse matrix pattern
- `{mesh}__{method}_solution.txt` - Displacement solution vector
- `{mesh}__{method}_nodes.txt` - Node coordinates
- `{mesh}__{method}_elements.txt` - Element connectivity

## Step 2: Generate Visualizations

Run the visualization script:

```bash
python3 visualize_demo.py --mesh bracket_3d --methods none rcm amd
```

### Options:
- `--mesh`: Mesh base name (default: `bracket_3d`)
- `--methods`: List of methods to visualize (default: `none rcm amd`)
- `--results-dir`: Directory containing results (default: `results`)
- `--output-dir`: Directory for output visualizations (default: `visualizations`)

### Example:
```bash
# Visualize all methods for bracket_3d
python3 visualize_demo.py --mesh bracket_3d

# Visualize specific methods for a different mesh
python3 visualize_demo.py --mesh bracket_3d_large --methods none rcm
```

## Step 3: View Results

The script generates visualizations in the `visualizations/` directory:

1. **Sparsity Patterns**:
   - `{mesh}__{method}_sparsity_original.png` - Original matrix pattern
   - `{mesh}__{method}_sparsity_reordered.png` - Reordered matrix pattern
   - `{mesh}_sparsity_comparison.png` - Side-by-side comparison of all methods

2. **Displacement Visualization**:
   - `{mesh}__{method}_displacement.png` - 3D mesh colored by displacement magnitude

## Dependencies

Required:
- `numpy`
- `matplotlib`

Optional (for better 3D visualization):
- `pyvista` - Provides high-quality 3D mesh rendering
  - Install with: `pip install pyvista`

If `pyvista` is not available, the script will fall back to matplotlib's 3D plotting.

## Understanding the Visualizations

### Sparsity Patterns
- **Black dots** represent non-zero entries in the matrix
- **Dense regions** show where nodes are strongly connected
- **Bandwidth** is visible as how far from the diagonal the non-zeros extend
- **Better reordering** should show a tighter band near the diagonal

### Displacement Magnitude
- **Color scale** shows displacement magnitude (how much each point moves)
- **Red/hot colors** = high displacement (high stress regions)
- **Blue/cold colors** = low displacement (low stress regions)
- This correlates with stress: areas with high displacement typically have high stress

## Demo Workflow

For your demo presentation:

1. **Show original matrix** (none method):
   ```bash
   python3 visualize_demo.py --mesh bracket_3d --methods none
   ```
   - Explain the sparse structure
   - Point out the bandwidth

2. **Show reordered matrices**:
   ```bash
   python3 visualize_demo.py --mesh bracket_3d --methods rcm amd
   ```
   - Compare sparsity patterns
   - Show how reordering improves bandwidth

3. **Show comparison**:
   ```bash
   python3 visualize_demo.py --mesh bracket_3d
   ```
   - Display the comparison plot showing all methods side-by-side

4. **Show displacement/stress**:
   - Display the displacement magnitude plots
   - Explain how this correlates with stress
   - Point out high-stress regions

## Troubleshooting

**Error: "Directory does not exist"**
- Make sure you've run the solver first to generate the data files

**Error: "Could not parse matrix dimensions"**
- Check that the matrix files were generated correctly
- Verify the file format matches what the script expects

**3D visualization looks poor**
- Install `pyvista` for better quality: `pip install pyvista`
- The matplotlib fallback works but has lower quality

**Large matrices take too long to visualize**
- The script automatically samples large matrices (>5000 DOFs)
- For very large meshes, consider using a smaller test mesh for visualization

