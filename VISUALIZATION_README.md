# FEM Solution Visualization Guide

This guide explains how to visualize displacement solution vectors from the FEM solver in Gmsh.

## Overview

The solver generates solution vectors containing displacement data for each degree of freedom (DOF). This data can be converted to Gmsh format for visualization with color maps showing displacement magnitude and vector fields.

## Quick Start

### Automated Approach (Recommended)

Use the provided shell script to run the solver and generate visualization files:

```bash
# Run solver and generate Gmsh files for all methods
./run_and_visualize.sh --mesh bracket_3d

# For a different mesh
./run_and_visualize.sh --mesh bracket_3d_large

# Only visualize existing results (skip solver)
./run_and_visualize.sh --mesh bracket_3d --skip-run
```

### Manual Approach

#### Step 1: Run the Solver

Run the solver for each reordering method you want to visualize:

```bash
# Run with no reordering
./fem_assembly CPU/bracket_3d.msh none

# Run with RCM reordering
./fem_assembly CPU/bracket_3d.msh rcm

# Run with AMD reordering
./fem_assembly CPU/bracket_3d.msh amd
```

This generates solution files in:
- `results/none/bracket_3d__none_solution.txt`
- `results/rcm/bracket_3d__rcm_solution.txt`
- `results/amd/bracket_3d__amd_solution.txt`

#### Step 2: Convert to Gmsh Format

Use the Python script to convert solution vectors to Gmsh post-processing format:

```bash
# Convert solution for each method
python3 gmsh_visualize.py --mesh bracket_3d --method none
python3 gmsh_visualize.py --mesh bracket_3d --method rcm
python3 gmsh_visualize.py --mesh bracket_3d --method amd
```

This creates Gmsh files in `gmsh_results/`:
- `bracket_3d__none_with_displacement.msh`
- `bracket_3d__rcm_with_displacement.msh`
- `bracket_3d__amd_with_displacement.msh`

#### Step 3: Visualize in Gmsh

Open the generated files in Gmsh:

```bash
# Open in Gmsh GUI
gmsh gmsh_results/bracket_3d__none_with_displacement.msh
```

## Gmsh Visualization Steps

1. **Open the file** in Gmsh (GUI or command line)

2. **View the displacement field**:
   - Go to **View → Options → Visibility**
   - Select **"Displacement_Magnitude"** for scalar color map
   - Or select **"Displacement_Vector"** for vector arrows

3. **Adjust visualization**:
   - **Colormap**: View → Options → Colormap
   - **Scale**: View → Options → Scale
   - **Range**: View → Options → Range (min/max values)

4. **Compare methods**:
   - Open multiple files in different Gmsh windows
   - Or use View → Options → General → Multi-view to see side-by-side

## Solution File Format

The solution files have the following format:

```
# Solution vector (displacements) for mesh: bracket_3d, method: none
# Format: DOF_index displacement_value
# nDOF: 165
# nNodes: 55
# ndof_per_node: 3
0 0.00440466
1 0.00440466
2 0.00440466
3 -1.50421e-05
...
```

For 3D problems:
- Each node has 3 DOFs: x-displacement, y-displacement, z-displacement
- DOFs are ordered as: `[node0_x, node0_y, node0_z, node1_x, node1_y, node1_z, ...]`
- The script automatically maps DOFs back to nodes

## Generated Gmsh Fields

Each generated `.msh` file contains two post-processing fields:

1. **Displacement_Magnitude** (scalar):
   - Magnitude = √(ux² + uy² + uz²)
   - Best for color maps showing overall displacement

2. **Displacement_Vector** (vector):
   - Components: (ux, uy, uz)
   - Best for vector arrows showing direction

## Command Line Options

### run_and_visualize.sh

```bash
./run_and_visualize.sh [OPTIONS]

Options:
  --mesh MESH          Mesh base name (default: bracket_3d)
  --mesh-dir DIR       Directory with mesh files (default: CPU)
  --results-dir DIR    Directory for results (default: results)
  --output-dir DIR     Output directory for Gmsh files (default: gmsh_results)
  --methods LIST       Comma-separated list (default: none,rcm,amd)
  --skip-run           Skip solver, only visualize
  --skip-viz           Skip visualization, only run solver
  -h, --help           Show help
```

### gmsh_visualize.py

```bash
python3 gmsh_visualize.py [OPTIONS]

Options:
  --mesh MESH          Mesh base name (required)
  --method METHOD      Reordering method: none, rcm, or amd (required)
  --mesh-dir DIR       Directory with mesh files (default: CPU)
  --results-dir DIR    Directory with solution files (default: results)
  --output-dir DIR     Output directory (default: gmsh_results)
  --field-name NAME    Scalar field name (default: Displacement_Magnitude)
```

## Examples

### Example 1: Visualize all methods for bracket_3d

```bash
./run_and_visualize.sh --mesh bracket_3d
```

### Example 2: Only visualize RCM and AMD (skip none)

```bash
./run_and_visualize.sh --mesh bracket_3d --methods rcm,amd
```

### Example 3: Only convert existing results (don't run solver)

```bash
./run_and_visualize.sh --mesh bracket_3d --skip-run
```

### Example 4: Custom directories

```bash
./run_and_visualize.sh \
    --mesh bracket_3d_large \
    --mesh-dir CPU \
    --results-dir results \
    --output-dir my_visualizations
```

### Example 5: Manual conversion for specific method

```bash
python3 gmsh_visualize.py \
    --mesh bracket_3d_xlarge \
    --method rcm \
    --output-dir custom_output
```

## Troubleshooting

### Solution file not found

**Error**: `ERROR: Solution file not found: results/none/bracket_3d__none_solution.txt`

**Solution**: Run the solver first:
```bash
./fem_assembly CPU/bracket_3d.msh none
```

### Mesh file not found

**Error**: `ERROR: Mesh file not found: CPU/bracket_3d.msh`

**Solution**: Check that the mesh file exists and use `--mesh-dir` if it's in a different location:
```bash
python3 gmsh_visualize.py --mesh bracket_3d --method none --mesh-dir /path/to/meshes
```

### Gmsh file not opening

**Solution**: 
- Make sure Gmsh is installed: `gmsh --version`
- Check file permissions: `chmod 644 gmsh_results/*.msh`
- Try opening from command line: `gmsh file.msh`

### Displacement values look wrong

**Solution**:
- Check that the solution file format matches expected format
- Verify nDOF matches nNodes × 3 (for 3D)
- Check that the mesh file and solution file are from the same run

## File Structure

After running the scripts, you'll have:

```
project/
├── CPU/
│   └── bracket_3d.msh              # Original mesh
├── results/
│   ├── none/
│   │   └── bracket_3d__none_solution.txt
│   ├── rcm/
│   │   └── bracket_3d__rcm_solution.txt
│   └── amd/
│       └── bracket_3d__amd_solution.txt
└── gmsh_results/
    ├── bracket_3d__none_with_displacement.msh
    ├── bracket_3d__rcm_with_displacement.msh
    └── bracket_3d__amd_with_displacement.msh
```

## Advanced: Comparing Methods

To compare displacement fields from different reordering methods:

1. Generate Gmsh files for all methods:
   ```bash
   ./run_and_visualize.sh --mesh bracket_3d
   ```

2. Open multiple files in Gmsh:
   ```bash
   gmsh gmsh_results/bracket_3d__none_with_displacement.msh &
   gmsh gmsh_results/bracket_3d__rcm_with_displacement.msh &
   gmsh gmsh_results/bracket_3d__amd_with_displacement.msh &
   ```

3. Use the same colormap scale for fair comparison:
   - View → Options → Range
   - Set the same min/max for all windows

## Notes

- The solution vectors contain displacements in the original mesh ordering (after inverse permutation)
- Displacement magnitude is computed as: `√(ux² + uy² + uz²)`
- For 3D problems, each node has 3 DOFs (x, y, z displacements)
- The Gmsh files preserve the original mesh structure and add post-processing data

