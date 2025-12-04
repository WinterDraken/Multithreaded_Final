#!/usr/bin/env python3
"""
Convert FEM solver solution vector to Gmsh post-processing format
Creates a .msh file with displacement/stress data that can be visualized in Gmsh
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
            # v4 format: numEntityBlocks numNodes minNodeTag maxNodeTag
            parts = lines[i].strip().split()
            num_entity_blocks = int(parts[0])
            num_nodes = int(parts[1])
            i += 1
            
            node_counter = 0
            for _ in range(num_entity_blocks):
                # entityDim entityTag parametric numNodesInBlock
                parts = lines[i].strip().split()
                num_nodes_in_block = int(parts[3])
                i += 1
                
                # Read node tags
                node_tags = list(map(int, lines[i].strip().split()))
                i += 1
                
                # Read node coordinates (x y z for each node)
                for j in range(num_nodes_in_block):
                    coords = list(map(float, lines[i].strip().split()))
                    node_id = node_tags[j]
                    nodes[node_id] = (coords[0], coords[1], coords[2])
                    i += 1
            continue
        
        i += 1
    
    return nodes, version

def load_solution_vector(solution_file):
    """Load solution vector from file"""
    displacements = []
    nDOF = None
    nNodes = None
    ndof_per_node = None
    
    with open(solution_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Parse metadata
                if 'nDOF:' in line:
                    nDOF = int(line.split(':')[1].strip())
                elif 'nNodes:' in line:
                    nNodes = int(line.split(':')[1].strip())
                elif 'ndof_per_node:' in line:
                    ndof_per_node = int(line.split(':')[1].strip())
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                displacements.append(float(parts[1]))
    
    return np.array(displacements), nDOF, nNodes, ndof_per_node

def compute_node_displacements(displacements, nodes, ndof_per_node=3):
    """Compute displacement magnitude for each node"""
    node_displacements = {}
    node_components = {}  # For vector visualization
    
    for node_id in nodes.keys():
        # DOF indexing: dof = node_id * ndof_per_node + component
        idx_x = node_id * ndof_per_node + 0
        idx_y = node_id * ndof_per_node + 1
        idx_z = node_id * ndof_per_node + 2
        
        if idx_x < len(displacements) and idx_y < len(displacements) and idx_z < len(displacements):
            ux = displacements[idx_x]
            uy = displacements[idx_y]
            uz = displacements[idx_z]
            
            # Displacement magnitude
            magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
            node_displacements[node_id] = magnitude
            node_components[node_id] = (ux, uy, uz)
        else:
            # Node not in solution (shouldn't happen, but handle gracefully)
            node_displacements[node_id] = 0.0
            node_components[node_id] = (0.0, 0.0, 0.0)
    
    return node_displacements, node_components

def write_gmsh_postprocessing(msh_file, output_file, node_displacements, node_components, field_name="Displacement_Magnitude"):
    """Write Gmsh post-processing data to mesh file"""
    # Read original mesh file
    with open(msh_file, 'r') as f:
        mesh_content = f.read()
    
    # Write new mesh file with post-processing data
    with open(output_file, 'w') as f:
        # Write original mesh content
        f.write(mesh_content)
        
        # Append post-processing data
        f.write("\n")
        f.write("$NodeData\n")
        f.write("1\n")  # Number of string tags
        f.write(f'"{field_name}"\n')  # Field name
        f.write("1\n")  # Number of real tags
        f.write("0.0\n")  # Time value
        f.write("3\n")  # Number of integer tags
        f.write("0\n")  # Time step (0 = static)
        f.write("1\n")  # Number of components (1 = scalar, 3 = vector)
        f.write(f"{len(node_displacements)}\n")  # Number of nodes
        
        # Write node data (scalar: magnitude)
        for node_id in sorted(node_displacements.keys()):
            f.write(f"{node_id} {node_displacements[node_id]:.15e}\n")
        
        f.write("$EndNodeData\n")
        
        # Also write vector field (displacement components)
        f.write("\n")
        f.write("$NodeData\n")
        f.write("1\n")
        f.write('"Displacement_Vector"\n')
        f.write("1\n")
        f.write("0.0\n")
        f.write("3\n")
        f.write("0\n")
        f.write("3\n")  # 3 components (vector)
        f.write(f"{len(node_components)}\n")
        
        for node_id in sorted(node_components.keys()):
            ux, uy, uz = node_components[node_id]
            f.write(f"{node_id} {ux:.15e} {uy:.15e} {uz:.15e}\n")
        
        f.write("$EndNodeData\n")

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
                       help='Name of the scalar field (default: Displacement_Magnitude)')
    
    args = parser.parse_args()
    
    # Construct file paths
    msh_file = Path(args.mesh_dir) / f"{args.mesh}.msh"
    solution_file = Path(args.results_dir) / args.method / f"{args.mesh}__{args.method}_solution.txt"
    output_dir = Path(args.output_dir)
    output_file = output_dir / f"{args.mesh}__{args.method}_with_displacement.msh"
    
    # Check if files exist
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
    
    print("=" * 80)
    print("Gmsh Post-Processing Converter")
    print("=" * 80)
    print(f"Mesh file: {msh_file}")
    print(f"Solution file: {solution_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Parse mesh nodes
    print("Parsing mesh file...")
    nodes, version = parse_gmsh_nodes(msh_file)
    print(f"  Found {len(nodes)} nodes (Gmsh format v{version})")
    
    # Load solution vector
    print("Loading solution vector...")
    displacements, nDOF, nNodes, ndof_per_node = load_solution_vector(solution_file)
    print(f"  Solution vector length: {len(displacements)}")
    print(f"  nDOF: {nDOF}, nNodes: {nNodes}, ndof_per_node: {ndof_per_node}")
    
    # Compute node displacements
    print("Computing displacement magnitudes...")
    node_displacements, node_components = compute_node_displacements(
        displacements, nodes, ndof_per_node
    )
    
    # Statistics
    magnitudes = list(node_displacements.values())
    print(f"  Min displacement: {min(magnitudes):.6e}")
    print(f"  Max displacement: {max(magnitudes):.6e}")
    print(f"  Mean displacement: {np.mean(magnitudes):.6e}")
    
    # Write Gmsh post-processing file
    print("Writing Gmsh post-processing file...")
    write_gmsh_postprocessing(msh_file, output_file, node_displacements, 
                              node_components, args.field_name)
    
    print()
    print("=" * 80)
    print("Success!")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print()
    print("To visualize in Gmsh:")
    print(f"  1. Open Gmsh: gmsh {output_file}")
    print("  2. Go to: View -> Options -> Visibility")
    print("  3. Select 'Displacement_Magnitude' or 'Displacement_Vector'")
    print("  4. Adjust colormap and scale as needed")
    print()
    print("Or use command line:")
    print(f"  gmsh {output_file}")
    print()

if __name__ == "__main__":
    main()

