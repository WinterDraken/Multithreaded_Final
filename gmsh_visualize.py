#!/usr/bin/env python3
"""
Convert FEM solver solution vector to Gmsh post-processing format
Creates a .msh file with displacement data that can be visualized in Gmsh
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

def parse_gmsh_nodes(msh_file):
    """Parse node IDs and coordinates from Gmsh mesh file"""
    nodes = {}  # node_id -> (x, y, z)
    version = None
    
    with open(msh_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect version
        if line == "$MeshFormat":
            i += 1
            if i < len(lines):
                version_str = lines[i].strip()
                if "4." in version_str:
                    version = 4
                else:
                    version = 2
            i += 1
            continue
        
        # Parse nodes (v2 format)
        if line == "$Nodes" and version == 2:
            i += 1
            num_nodes = int(lines[i].strip())
            i += 1
            for _ in range(num_nodes):
                parts = lines[i].strip().split()
                node_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes[node_id] = (x, y, z)
                i += 1
            continue
        
        # Parse nodes (v4 format)
        if line == "$Nodes" and version == 4:
            i += 1
            num_entity_blocks, num_nodes, min_node_tag, max_node_tag = map(int, lines[i].strip().split())
            i += 1
            node_ids_list = []
            for _ in range(num_entity_blocks):
                entity_dim, entity_tag, parametric, num_nodes_in_block = map(int, lines[i].strip().split())
                i += 1
                # Read node IDs for this block
                block_node_ids = []
                for _ in range(num_nodes_in_block):
                    node_id = int(lines[i].strip())
                    block_node_ids.append(node_id)
                    i += 1
                # Read coordinates for this block (in same order as node IDs)
                for j in range(num_nodes_in_block):
                    parts = lines[i].strip().split()
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    node_id = block_node_ids[j]
                    nodes[node_id] = (x, y, z)
                    i += 1
            continue
        
        i += 1
    
    return nodes, version

def load_solution_vector(solution_file):
    """Load solution vector from file"""
    displacements = []
    nDOF = None
    nNodes = None
    ndof_per_node = 3  # Default for 3D
    
    with open(solution_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Parse metadata
                if 'nDOF' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'nDOF:':
                            nDOF = int(parts[i+1])
                if 'nNodes' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'nNodes:':
                            nNodes = int(parts[i+1])
                if 'ndof_per_node' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'ndof_per_node:':
                            ndof_per_node = int(parts[i+1])
                continue
            
            # Parse displacement values
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    dof_idx = int(parts[0])
                    disp_val = float(parts[1])
                    displacements.append(disp_val)
                except ValueError:
                    continue
    
    if nDOF is None:
        nDOF = len(displacements)
    if nNodes is None:
        nNodes = nDOF // ndof_per_node
    
    return displacements, nDOF, nNodes, ndof_per_node

def create_gmsh_postprocessing(msh_file, solution_file, output_file, field_name='Displacement_Magnitude'):
    """Create Gmsh .msh file with displacement post-processing data"""
    
    # Parse original mesh to get node mapping
    print(f"Parsing mesh file: {msh_file}")
    nodes, version = parse_gmsh_nodes(msh_file)
    print(f"  Found {len(nodes)} nodes")
    
    # Load solution vector
    print(f"Loading solution vector: {solution_file}")
    displacements, nDOF, nNodes, ndof_per_node = load_solution_vector(solution_file)
    print(f"  nDOF: {nDOF}, nNodes: {nNodes}, ndof_per_node: {ndof_per_node}")
    
    # Map DOF displacements to nodes
    # For 3D: DOF order is [node0_x, node0_y, node0_z, node1_x, node1_y, node1_z, ...]
    node_displacements = {}  # node_id -> (ux, uy, uz)
    node_magnitudes = {}     # node_id -> magnitude
    
    # Get sorted node IDs (by their index in the solution vector, not by Gmsh tag)
    # The solution vector uses contiguous indices 0..nNodes-1
    # We need to map these to actual Gmsh node tags
    sorted_node_ids = sorted(nodes.keys())
    
    # Check if we need to map by index or by tag
    # If nNodes matches len(nodes), assume 1:1 mapping by sorted order
    if nNodes == len(nodes):
        for node_idx, node_id in enumerate(sorted_node_ids):
            dof_base = node_idx * ndof_per_node
            if dof_base + 2 < len(displacements):
                ux = displacements[dof_base]
                uy = displacements[dof_base + 1]
                uz = displacements[dof_base + 2]
                node_displacements[node_id] = (ux, uy, uz)
                magnitude = np.sqrt(ux*ux + uy*uy + uz*uz)
                node_magnitudes[node_id] = magnitude
            else:
                node_displacements[node_id] = (0.0, 0.0, 0.0)
                node_magnitudes[node_id] = 0.0
    else:
        print(f"  WARNING: Node count mismatch: solution has {nNodes}, mesh has {len(nodes)}")
        # Try to map by assuming node IDs start from 1 and are contiguous
        for node_idx in range(min(nNodes, len(nodes))):
            node_id = sorted_node_ids[node_idx]
            dof_base = node_idx * ndof_per_node
            if dof_base + 2 < len(displacements):
                ux = displacements[dof_base]
                uy = displacements[dof_base + 1]
                uz = displacements[dof_base + 2]
                node_displacements[node_id] = (ux, uy, uz)
                magnitude = np.sqrt(ux*ux + uy*uy + uz*uz)
                node_magnitudes[node_id] = magnitude
    
    # Copy entire original mesh file and append post-processing data
    print(f"Writing Gmsh file with displacements: {output_file}")
    with open(msh_file, 'r') as orig:
        with open(output_file, 'w') as f:
            # Copy everything up to and including $EndElements
            copy_mode = True
            for line in orig:
                if copy_mode:
                    f.write(line)
                    # Stop copying after $EndElements
                    if line.strip() == "$EndElements":
                        copy_mode = False
                        break
            
            # Write post-processing view data
            f.write("$NodeData\n")
            f.write("1\n")  # Number of string tags
            f.write(f'"{field_name}"\n')  # Field name
            f.write("1\n")  # Number of real tags
            f.write("0.0\n")  # Time value
            f.write("3\n")  # Number of integer tags
            f.write("0\n")  # Time step index
            f.write("1\n")  # Number of components (scalar)
            f.write(f"{len(sorted_node_ids)}\n")  # Number of nodes
            
            # Write displacement magnitude for each node (in sorted order)
            for node_id in sorted_node_ids:
                magnitude = node_magnitudes.get(node_id, 0.0)
                f.write(f"{node_id} {magnitude:.15e}\n")
            
            f.write("$EndNodeData\n")
            
            # Also write vector displacement field
            f.write("$NodeData\n")
            f.write("1\n")  # Number of string tags
            f.write('"Displacement_Vector"\n')  # Field name
            f.write("1\n")  # Number of real tags
            f.write("0.0\n")  # Time value
            f.write("3\n")  # Number of integer tags
            f.write("0\n")  # Time step index
            f.write("3\n")  # Number of components (vector: x, y, z)
            f.write(f"{len(sorted_node_ids)}\n")  # Number of nodes
            
            # Write displacement vector for each node (in sorted order)
            for node_id in sorted_node_ids:
                ux, uy, uz = node_displacements.get(node_id, (0.0, 0.0, 0.0))
                f.write(f"{node_id} {ux:.15e} {uy:.15e} {uz:.15e}\n")
            
            f.write("$EndNodeData\n")
    
    print(f"✓ Successfully created: {output_file}")
    print(f"  Fields: {field_name} (scalar), Displacement_Vector (vector)")

def main():
    parser = argparse.ArgumentParser(
        description='Convert FEM solution vector to Gmsh post-processing format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert solution for bracket_3d with none method
  python3 gmsh_visualize.py --mesh bracket_3d --method none
  
  # Specify custom paths
  python3 gmsh_visualize.py --mesh bracket_3d --method rcm \\
      --mesh-dir CPU --results-dir results --output-dir gmsh_results
        """
    )
    
    parser.add_argument('--mesh', type=str, required=True,
                       help='Mesh base name (e.g., bracket_3d)')
    parser.add_argument('--method', type=str, required=True,
                       help='Reordering method (none, rcm, or amd)')
    parser.add_argument('--mesh-dir', type=str, default='CPU',
                       help='Directory containing mesh files (default: CPU)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing solution files (default: results)')
    parser.add_argument('--output-dir', type=str, default='gmsh_results',
                       help='Output directory for Gmsh files (default: gmsh_results)')
    parser.add_argument('--field-name', type=str, default='Displacement_Magnitude',
                       help='Name for scalar displacement field (default: Displacement_Magnitude)')
    
    args = parser.parse_args()
    
    # Construct file paths
    msh_file = Path(args.mesh_dir) / f"{args.mesh}.msh"
    solution_file = Path(args.results_dir) / args.method / f"{args.mesh}__{args.method}_solution.txt"
    output_dir = Path(args.output_dir)
    output_file = output_dir / f"{args.mesh}__{args.method}_with_displacement.msh"
    
    # Validate inputs
    if not msh_file.exists():
        print(f"ERROR: Mesh file not found: {msh_file}")
        sys.exit(1)
    
    if not solution_file.exists():
        print(f"ERROR: Solution file not found: {solution_file}")
        print(f"Make sure you've run the solver first:")
        print(f"  ./fem_assembly {msh_file} {args.method}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FEM Solution to Gmsh Converter")
    print("=" * 60)
    print()
    print(f"Mesh file: {msh_file}")
    print(f"Solution file: {solution_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Convert
    create_gmsh_postprocessing(msh_file, solution_file, output_file, args.field_name)
    
    print()
    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print()
    print(f"To visualize in Gmsh:")
    print(f"  gmsh {output_file}")
    print()
    print("In Gmsh GUI:")
    print("  1. Go to View → Options → Visibility")
    print("  2. Select 'Displacement_Magnitude' for scalar field")
    print("  3. Or select 'Displacement_Vector' for vector arrows")
    print("  4. Adjust colormap and scale as needed")
    print()

if __name__ == "__main__":
    main()

