# Gmsh Visualization Guide

This guide explains how to visualize displacement/stress fields from your FEM solver results using Gmsh.

## Solution Vector Location

The solution vectors (displacements) are saved in:
```
results/{method}/{mesh}__{method}_solution.txt
```

For example:
- `results/none/bracket_3d__none_solution.txt`
- `results/rcm/bracket_3d__rcm_solution.txt`
- `results/amd/bracket_3d__amd_solution.txt`

## Quick Start

### Step 1: Run the Solver

First, generate the solution vectors by running the solver:

```bash
./fem_assembly CPU/bracket_3d.msh none
./fem_assembly CPU/bracket_3d.msh rcm
./fem_assembly CPU/bracket_3d.msh amd
```

### Step 2: Convert to Gmsh Format

Use the conversion script to create Gmsh post-processing files:

```bash
# Convert solution for a specific mesh and method
python3 gmsh_visualize.py --mesh bracket_3d --method none

# Convert all methods
python3 gmsh_visualize.py --mesh bracket_3d --method none
python3 gmsh_visualize.py --mesh bracket_3d --method rcm
python3 gmsh_visualize.py --mesh bracket_3d --method amd
```

This creates files in `gmsh_results/`:
- `bracket_3d__none_with_displacement.msh`
- `bracket_3d__rcm_with_displacement.msh`
- `bracket_3d__amd_with_displacement.msh`

### Step 3: Visualize in Gmsh

Open the file in Gmsh:

```bash
gmsh gmsh_results/bracket_3d__none_with_displacement.msh
```

**In Gmsh GUI:**
1. Go to **View → Options** (or press `Alt+O`)
2. In the **Visibility** tab, you'll see:
   - `Displacement_Magnitude` - Scalar field (displacement magnitude)
   - `Displacement_Vector` - Vector field (x, y, z components)
3. Select the field you want to visualize
4. Adjust colormap, scale, and other visualization options
5. The mesh will be colored by displacement magnitude (which correlates with stress)

## Script Options

```bash
python3 gmsh_visualize.py --help
```

**Common options:**
- `--mesh`: Mesh base name (required)
- `--method`: Reordering method: none, rcm, or amd (required)
- `--mesh-dir`: Directory with mesh files (default: `CPU`)
- `--results-dir`: Directory with solution files (default: `results`)
- `--output-dir`: Output directory (default: `gmsh_results`)
- `--field-name`: Name for the scalar field (default: `Displacement_Magnitude`)

**Example with custom paths:**
```bash
python3 gmsh_visualize.py \
    --mesh bracket_3d_large \
    --method rcm \
    --mesh-dir CPU \
    --results-dir results \
    --output-dir gmsh_visualizations
```

## Understanding the Output

The script creates a `.msh` file with two post-processing fields:

1. **Displacement_Magnitude** (scalar):
   - Shows `√(ux² + uy² + uz²)` at each node
   - **Red/hot colors** = high displacement (high stress regions)
   - **Blue/cold colors** = low displacement (low stress regions)
   - This is what you'll typically use for stress visualization

2. **Displacement_Vector** (vector):
   - Shows the full displacement vector (ux, uy, uz) at each node
   - Can be visualized as arrows or vector field
   - Useful for understanding deformation direction

## Gmsh Visualization Tips

### Viewing Scalar Fields (Displacement Magnitude)
1. Open the `.msh` file in Gmsh
2. Go to **View → Options** → **Visibility**
3. Select `Displacement_Magnitude`
4. In **General** tab:
   - Adjust **Range** to see the full scale
   - Change **Colormap** (e.g., "Viridis", "Hot", "Cool")
   - Enable **Smooth** for better visualization
5. In **Scale** tab:
   - Adjust **Min** and **Max** to focus on specific ranges

### Viewing Vector Fields
1. Select `Displacement_Vector` in Visibility
2. In **Vector** tab:
   - Enable **Draw vectors**
   - Adjust **Scale factor** to make arrows visible
   - Change **Arrow size** and **Color**

### Exporting Images
1. Go to **File → Export**
2. Choose format (PNG, PDF, etc.)
3. Adjust resolution and options
4. Save

## Automated Workflow

You can combine everything in one script. Add this to your workflow:

```bash
#!/bin/bash
# Run solver and convert to Gmsh format

MESH="bracket_3d"
METHODS="none rcm amd"

# Run solver
for method in $METHODS; do
    ./fem_assembly CPU/${MESH}.msh $method
done

# Convert to Gmsh format
for method in $METHODS; do
    python3 gmsh_visualize.py --mesh $MESH --method $method
done

echo "Gmsh files ready in gmsh_results/"
```

## Troubleshooting

**Error: "Solution file not found"**
- Make sure you've run the solver first
- Check that the file path matches: `results/{method}/{mesh}__{method}_solution.txt`

**Error: "Mesh file not found"**
- Check that the mesh file exists in the `CPU/` directory
- Verify the mesh name is correct

**Gmsh shows no data**
- Make sure you selected the field in View → Options → Visibility
- Check that the field name appears in the list
- Try reloading the file: File → Reload

**Displacement values seem wrong**
- Check the solution file format
- Verify that nDOF, nNodes, and ndof_per_node match expectations
- The script assumes DOF indexing: `dof = node_id * 3 + component`

## File Format Details

The solution vector file format:
```
# Solution vector (displacements) for mesh: bracket_3d, method: none
# Format: DOF_index displacement_value
# nDOF: 4095
# nNodes: 1365
# ndof_per_node: 3
0 <value>
1 <value>
2 <value>
...
```

DOF indexing:
- DOF 0, 1, 2 → Node 0 (x, y, z)
- DOF 3, 4, 5 → Node 1 (x, y, z)
- DOF 6, 7, 8 → Node 2 (x, y, z)
- etc.

The Gmsh post-processing format adds `$NodeData` sections to the mesh file with the computed displacement values.

