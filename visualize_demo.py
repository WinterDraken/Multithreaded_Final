#!/usr/bin/env python3
"""
FEM GPU Solver Visualization Script for Demo
Visualizes:
1. Sparse matrix sparsity patterns (before/after reordering)
2. Displacement magnitude on 3D mesh
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path

# Try to import pyvista for 3D mesh visualization, fallback to matplotlib
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: pyvista not available. 3D mesh visualization will use matplotlib.")

def load_matrix_pattern(filepath):
    """Load sparse matrix pattern from file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find nDOF and nnz
    nDOF = None
    nnz = None
    rowPtr_line_idx = None
    colIdx_line_idx = None
    
    for i, line in enumerate(lines):
        if line.startswith('# nDOF:'):
            nDOF = int(line.split(':')[1].strip())
        elif line.startswith('# nnz:'):
            nnz = int(line.split(':')[1].strip())
        elif '# rowPtr' in line:
            rowPtr_line_idx = i + 1
        elif '# colIdx' in line:
            colIdx_line_idx = i + 1
    
    if nDOF is None or nnz is None:
        raise ValueError(f"Could not parse matrix dimensions from {filepath}")
    
    # Read rowPtr
    rowPtr = np.array([int(x) for x in lines[rowPtr_line_idx].strip().split()])
    
    # Read colIdx
    colIdx = np.array([int(x) for x in lines[colIdx_line_idx].strip().split()])
    
    return nDOF, nnz, rowPtr, colIdx

def load_solution(filepath):
    """Load solution vector (displacements) from file"""
    displacements = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                displacements.append(float(parts[1]))
    return np.array(displacements)

def load_nodes(filepath):
    """Load node coordinates from file"""
    nodes = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(nodes)

def load_elements(filepath):
    """Load element connectivity from file"""
    elements = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                elements.append([int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
    return np.array(elements)

def create_sparse_matrix_plot(rowPtr, colIdx, nDOF, nnz, title, output_file):
    """Create sparsity pattern plot using matplotlib"""
    # Create sparse matrix representation
    row_indices = []
    col_indices = []
    
    for i in range(nDOF):
        start = rowPtr[i]
        end = rowPtr[i + 1]
        for j in range(start, end):
            row_indices.append(i)
            col_indices.append(colIdx[j])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # For large matrices, sample points for visualization
    if nDOF > 5000:
        # Sample every nth point
        sample_rate = max(1, nDOF // 5000)
        row_indices = np.array(row_indices)[::sample_rate]
        col_indices = np.array(col_indices)[::sample_rate]
        ax.scatter(col_indices, row_indices, s=0.1, c='black', marker='.', alpha=0.5)
    else:
        ax.scatter(col_indices, row_indices, s=0.5, c='black', marker='.', alpha=0.6)
    
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)
    ax.set_title(f'{title}\nMatrix Size: {nDOF}×{nDOF}, Non-zeros: {nnz}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match matrix convention
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved sparsity plot: {output_file}")
    plt.close()

def visualize_displacement_pyvista(nodes, elements, displacements, output_file):
    """Visualize displacement magnitude on 3D mesh using PyVista"""
    # Create PyVista unstructured grid
    cells = []
    cell_types = []
    
    for elem in elements:
        cells.append(4)  # Number of points in tetrahedron
        cells.extend(elem)
        cell_types.append(pv.CellType.TETRA)
    
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    
    # Compute displacement magnitude at nodes
    # displacements is per DOF, we need per node
    n_nodes = len(nodes)
    ndof_per_node = len(displacements) // n_nodes
    
    node_displacements = np.zeros(n_nodes)
    for i in range(n_nodes):
        idx_x = i * ndof_per_node
        idx_y = idx_x + 1
        idx_z = idx_x + 2
        ux = displacements[idx_x] if idx_x < len(displacements) else 0.0
        uy = displacements[idx_y] if idx_y < len(displacements) else 0.0
        uz = displacements[idx_z] if idx_z < len(displacements) else 0.0
        node_displacements[i] = np.sqrt(ux**2 + uy**2 + uz**2)
    
    grid.point_data['Displacement_Magnitude'] = node_displacements
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, scalars='Displacement_Magnitude', 
                     cmap='viridis', show_edges=False, opacity=0.9)
    plotter.add_scalar_bar(title='Displacement Magnitude', n_labels=5)
    plotter.camera_position = 'iso'
    plotter.background_color = 'white'
    
    plotter.screenshot(output_file)
    print(f"Saved 3D displacement visualization: {output_file}")
    plotter.close()

def visualize_displacement_matplotlib(nodes, elements, displacements, output_file):
    """Visualize displacement magnitude using matplotlib (fallback)"""
    # Compute displacement magnitude at nodes
    n_nodes = len(nodes)
    ndof_per_node = len(displacements) // n_nodes
    
    node_displacements = np.zeros(n_nodes)
    for i in range(n_nodes):
        idx_x = i * ndof_per_node
        idx_y = idx_x + 1
        idx_z = idx_x + 2
        ux = displacements[idx_x] if idx_x < len(displacements) else 0.0
        uy = displacements[idx_y] if idx_y < len(displacements) else 0.0
        uz = displacements[idx_z] if idx_z < len(displacements) else 0.0
        node_displacements[i] = np.sqrt(ux**2 + uy**2 + uz**2)
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], 
                        c=node_displacements, cmap='viridis', 
                        s=20, alpha=0.8)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Displacement Magnitude on Mesh', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Displacement Magnitude', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved 3D displacement visualization: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize FEM solver results')
    parser.add_argument('--mesh', type=str, default='bracket_3d',
                       help='Mesh base name (default: bracket_3d)')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['none', 'rcm', 'amd'],
                       help='Reordering methods to visualize (default: none rcm amd)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory (default: results)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations (default: visualizations)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("FEM GPU Solver Visualization")
    print("=" * 80)
    print(f"Mesh: {args.mesh}")
    print(f"Methods: {args.methods}")
    print()
    
    # Process each method
    for method in args.methods:
        print(f"Processing method: {method}")
        
        method_dir = Path(args.results_dir) / method
        if not method_dir.exists():
            print(f"  Warning: Directory {method_dir} does not exist, skipping...")
            continue
        
        # Find files for this mesh and method
        matrix_orig_file = method_dir / f"{args.mesh}__{method}_matrix_original.txt"
        matrix_reord_file = method_dir / f"{args.mesh}__{method}_matrix_reordered.txt"
        solution_file = method_dir / f"{args.mesh}__{method}_solution.txt"
        nodes_file = method_dir / f"{args.mesh}__{method}_nodes.txt"
        elements_file = method_dir / f"{args.mesh}__{method}_elements.txt"
        
        # Visualize sparse matrix patterns
        if matrix_orig_file.exists():
            try:
                nDOF, nnz, rowPtr, colIdx = load_matrix_pattern(matrix_orig_file)
                output_file = Path(args.output_dir) / f"{args.mesh}__{method}_sparsity_original.png"
                create_sparse_matrix_plot(rowPtr, colIdx, nDOF, nnz, 
                                         f'Sparsity Pattern: {method.upper()} (Original)', 
                                         str(output_file))
            except Exception as e:
                print(f"  Error visualizing original matrix: {e}")
        
        if matrix_reord_file.exists():
            try:
                nDOF, nnz, rowPtr, colIdx = load_matrix_pattern(matrix_reord_file)
                output_file = Path(args.output_dir) / f"{args.mesh}__{method}_sparsity_reordered.png"
                create_sparse_matrix_plot(rowPtr, colIdx, nDOF, nnz, 
                                         f'Sparsity Pattern: {method.upper()} (Reordered)', 
                                         str(output_file))
            except Exception as e:
                print(f"  Error visualizing reordered matrix: {e}")
        
        # Visualize displacement magnitude
        if solution_file.exists() and nodes_file.exists() and elements_file.exists():
            try:
                displacements = load_solution(solution_file)
                nodes = load_nodes(nodes_file)
                elements = load_elements(elements_file)
                
                output_file = Path(args.output_dir) / f"{args.mesh}__{method}_displacement.png"
                
                if HAS_PYVISTA:
                    visualize_displacement_pyvista(nodes, elements, displacements, str(output_file))
                else:
                    visualize_displacement_matplotlib(nodes, elements, displacements, str(output_file))
            except Exception as e:
                print(f"  Error visualizing displacement: {e}")
                import traceback
                traceback.print_exc()
        
        print()
    
    # Create comparison plot for sparsity patterns
    print("Creating comparison plot...")
    fig, axes = plt.subplots(1, len(args.methods), figsize=(6*len(args.methods), 6))
    if len(args.methods) == 1:
        axes = [axes]
    
    for idx, method in enumerate(args.methods):
        method_dir = Path(args.results_dir) / method
        matrix_reord_file = method_dir / f"{args.mesh}__{method}_matrix_reordered.txt"
        
        if matrix_reord_file.exists():
            try:
                nDOF, nnz, rowPtr, colIdx = load_matrix_pattern(matrix_reord_file)
                
                row_indices = []
                col_indices = []
                for i in range(nDOF):
                    start = rowPtr[i]
                    end = rowPtr[i + 1]
                    for j in range(start, end):
                        row_indices.append(i)
                        col_indices.append(colIdx[j])
                
                if nDOF > 5000:
                    sample_rate = max(1, nDOF // 5000)
                    row_indices = np.array(row_indices)[::sample_rate]
                    col_indices = np.array(col_indices)[::sample_rate]
                    axes[idx].scatter(col_indices, row_indices, s=0.1, c='black', marker='.', alpha=0.5)
                else:
                    axes[idx].scatter(col_indices, row_indices, s=0.5, c='black', marker='.', alpha=0.6)
                
                axes[idx].set_title(f'{method.upper()}\n({nDOF}×{nDOF}, {nnz} nnz)', 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Column Index', fontsize=10)
                if idx == 0:
                    axes[idx].set_ylabel('Row Index', fontsize=10)
                axes[idx].set_aspect('equal')
                axes[idx].invert_yaxis()
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error\n{str(e)}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{method.upper()}', fontsize=12)
    
    plt.suptitle(f'Sparsity Pattern Comparison: {args.mesh}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    comparison_file = Path(args.output_dir) / f"{args.mesh}_sparsity_comparison.png"
    plt.savefig(str(comparison_file), dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {comparison_file}")
    plt.close()
    
    print()
    print("=" * 80)
    print("Visualization complete!")
    print(f"Results saved in: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

