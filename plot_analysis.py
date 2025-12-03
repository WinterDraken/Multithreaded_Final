#!/usr/bin/env python3
"""
FEM GPU Solver Analysis and Visualization Script
Generates plots for memory access patterns, cache locality, and SpMV performance
"""

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

# Configuration
BENCHMARK_DIR = "benchmark_results"
PROFILE_DIR = "profile_results"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme for methods
METHOD_COLORS = {
    'none': '#1f77b4',  # blue
    'rcm': '#ff7f0e',   # orange
    'amd': '#2ca02c'    # green
}

METHOD_LABELS = {
    'none': 'No Reordering',
    'rcm': 'RCM',
    'amd': 'AMD'
}

def load_benchmark_data():
    """Load all benchmark CSV files and aggregate statistics"""
    data = defaultdict(lambda: defaultdict(list))
    
    for csv_file in Path(BENCHMARK_DIR).glob("*.csv"):
        # Parse filename: mesh_base__method.csv
        match = re.match(r'(.+)__(.+)\.csv', csv_file.stem)
        if not match:
            continue
        
        mesh_base, method = match.groups()
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Store mean and std for each metric
        for col in ['cpu_assembly_ms', 'reordering_ms', 'h2d_ms', 'gpu_solve_ms', 'total_ms']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                data[mesh_base][f'{col}_mean'] = mean_val
                data[mesh_base][f'{col}_std'] = std_val
                data[mesh_base][f'{col}_method'] = method
    
    return data

def extract_mesh_size(mesh_base):
    """Extract approximate mesh size from name"""
    size_map = {
        'bracket_3d': 1,
        'bracket_3d_large': 2,
        'bracket_3d_xlarge': 3,
        'bracket_3d_xxlarge': 4,
        'bracket_3d_xxxlarge': 5,
        'bracket_3d_xxxxlarge': 6
    }
    return size_map.get(mesh_base, 0)

def plot_timing_comparison(data):
    """Plot 1: Timing comparison across methods and mesh sizes"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: Reordering Methods', fontsize=16, fontweight='bold')
    
    methods = ['none', 'rcm', 'amd']
    mesh_sizes = sorted(set(extract_mesh_size(m) for m in data.keys()))
    mesh_names = sorted([m for m in data.keys()], key=extract_mesh_size)
    
    # Plot 1: Total Time
    ax = axes[0, 0]
    x = np.arange(len(mesh_names))
    width = 0.25
    
    for i, method in enumerate(methods):
        means = []
        stds = []
        for mesh in mesh_names:
            key = f'total_ms_mean'
            if key in data[mesh] and data[mesh].get(f'total_ms_method') == method:
                means.append(data[mesh][key])
                stds.append(data[mesh][f'total_ms_std'])
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i*width, means, width, label=METHOD_LABELS[method], 
               color=METHOD_COLORS[method], alpha=0.8, yerr=stds, capsize=3)
    
    ax.set_xlabel('Mesh Size')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title('Total Execution Time')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('bracket_3d_', '').replace('bracket_3d', 'small') for m in mesh_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: GPU Solve Time (SpMV performance)
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        means = []
        stds = []
        for mesh in mesh_names:
            key = f'gpu_solve_ms_mean'
            if key in data[mesh] and data[mesh].get(f'gpu_solve_ms_method') == method:
                means.append(data[mesh][key])
                stds.append(data[mesh][f'gpu_solve_ms_std'])
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i*width, means, width, label=METHOD_LABELS[method], 
               color=METHOD_COLORS[method], alpha=0.8, yerr=stds, capsize=3)
    
    ax.set_xlabel('Mesh Size')
    ax.set_ylabel('GPU Solve Time (ms)')
    ax.set_title('SpMV Performance (GPU Solve Phase)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('bracket_3d_', '').replace('bracket_3d', 'small') for m in mesh_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Assembly Time
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        means = []
        stds = []
        for mesh in mesh_names:
            key = f'cpu_assembly_ms_mean'
            if key in data[mesh] and data[mesh].get(f'cpu_assembly_ms_method') == method:
                means.append(data[mesh][key])
                stds.append(data[mesh][f'cpu_assembly_ms_std'])
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i*width, means, width, label=METHOD_LABELS[method], 
               color=METHOD_COLORS[method], alpha=0.8, yerr=stds, capsize=3)
    
    ax.set_xlabel('Mesh Size')
    ax.set_ylabel('CPU Assembly Time (ms)')
    ax.set_title('CSR Pattern Building (CPU)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('bracket_3d_', '').replace('bracket_3d', 'small') for m in mesh_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speedup relative to 'none'
    ax = axes[1, 1]
    for method in ['rcm', 'amd']:
        speedups = []
        for mesh in mesh_names:
            none_key = f'total_ms_mean'
            method_key = f'total_ms_mean'
            if (none_key in data[mesh] and data[mesh].get(f'total_ms_method') == 'none' and
                method_key in data[mesh] and data[mesh].get(f'total_ms_method') == method):
                speedup = data[mesh][none_key] / data[mesh][method_key]
                speedups.append(speedup)
            else:
                speedups.append(1.0)
        
        ax.plot(range(len(speedups)), speedups, marker='o', label=METHOD_LABELS[method],
                color=METHOD_COLORS[method], linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (none)')
    ax.set_xlabel('Mesh Size Index')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs. No Reordering')
    ax.set_xticks(range(len(mesh_names)))
    ax.set_xticklabels([m.replace('bracket_3d_', '').replace('bracket_3d', 'small') for m in mesh_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/timing_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/timing_comparison.png")

def plot_spmv_analysis(data):
    """Plot 2: Detailed SpMV performance analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SpMV Performance Analysis', fontsize=16, fontweight='bold')
    
    methods = ['none', 'rcm', 'amd']
    mesh_names = sorted([m for m in data.keys()], key=extract_mesh_size)
    x = np.arange(len(mesh_names))
    width = 0.25
    
    # Plot 1: Solve time breakdown
    ax = axes[0]
    for i, method in enumerate(methods):
        solve_times = []
        for mesh in mesh_names:
            key = f'gpu_solve_ms_mean'
            if key in data[mesh] and data[mesh].get(f'gpu_solve_ms_method') == method:
                solve_times.append(data[mesh][key])
            else:
                solve_times.append(0)
        
        ax.bar(x + i*width, solve_times, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], alpha=0.8)
    
    ax.set_xlabel('Mesh Size')
    ax.set_ylabel('GPU Solve Time (ms)')
    ax.set_title('SpMV Execution Time (Lower is Better)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('bracket_3d_', '').replace('bracket_3d', 'small') for m in mesh_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Relative performance
    ax = axes[1]
    baseline_times = []
    for mesh in mesh_names:
        key = f'gpu_solve_ms_mean'
        if key in data[mesh] and data[mesh].get(f'gpu_solve_ms_method') == 'none':
            baseline_times.append(data[mesh][key])
        else:
            baseline_times.append(1.0)
    
    for method in ['rcm', 'amd']:
        relative_perf = []
        for i, mesh in enumerate(mesh_names):
            key = f'gpu_solve_ms_mean'
            if key in data[mesh] and data[mesh].get(f'gpu_solve_ms_method') == method:
                if baseline_times[i] > 0:
                    relative_perf.append(baseline_times[i] / data[mesh][key])
                else:
                    relative_perf.append(1.0)
            else:
                relative_perf.append(1.0)
        
        ax.plot(range(len(relative_perf)), relative_perf, marker='o', 
                label=METHOD_LABELS[method], color=METHOD_COLORS[method], 
                linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (none)')
    ax.set_xlabel('Mesh Size Index')
    ax.set_ylabel('Relative Performance (Higher is Better)')
    ax.set_title('SpMV Speedup vs. No Reordering')
    ax.set_xticks(range(len(mesh_names)))
    ax.set_xticklabels([m.replace('bracket_3d_', '').replace('bracket_3d', 'small') for m in mesh_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/spmv_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/spmv_analysis.png")

def parse_profile_metrics():
    """Parse profiling metrics from nsys/nvprof output"""
    metrics = defaultdict(dict)
    
    # Try to parse nsys reports
    for rep_file in Path(PROFILE_DIR).glob("*.nsys-rep"):
        # Extract mesh and method from filename
        match = re.match(r'(.+)__(.+)\.nsys-rep', rep_file.stem)
        if match:
            mesh_base, method = match.groups()
            # Note: Detailed parsing would require nsys Python API
            # For now, we'll note that profiles exist
            metrics[mesh_base][method] = {'profile_exists': True}
    
    # Try to parse nvprof CSV files
    for csv_file in Path(PROFILE_DIR).glob("*.metrics.csv"):
        match = re.match(r'(.+)__(.+)\.metrics\.csv', csv_file.stem)
        if match:
            mesh_base, method = match.groups()
            try:
                # nvprof CSV format parsing
                with open(csv_file, 'r') as f:
                    content = f.read()
                    # Extract key metrics (this is simplified - adjust based on actual format)
                    if 'l2_cache_hit_rate' in content.lower():
                        # Parse actual values here
                        pass
            except:
                pass
    
    return metrics

def plot_scalability(data):
    """Plot 3: Scalability analysis"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = ['none', 'rcm', 'amd']
    mesh_names = sorted([m for m in data.keys()], key=extract_mesh_size)
    
    # Estimate problem size (use mesh index as proxy)
    problem_sizes = [extract_mesh_size(m) for m in mesh_names]
    
    for method in methods:
        solve_times = []
        for mesh in mesh_names:
            key = f'gpu_solve_ms_mean'
            if key in data[mesh] and data[mesh].get(f'gpu_solve_ms_method') == method:
                solve_times.append(data[mesh][key])
            else:
                solve_times.append(np.nan)
        
        ax.plot(problem_sizes, solve_times, marker='o', label=METHOD_LABELS[method],
                color=METHOD_COLORS[method], linewidth=2, markersize=8)
    
    ax.set_xlabel('Problem Size (Mesh Index)')
    ax.set_ylabel('GPU Solve Time (ms)')
    ax.set_title('SpMV Scalability: Performance vs. Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/scalability.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/scalability.png")

def generate_summary_report(data):
    """Generate a text summary report"""
    report_file = f'{OUTPUT_DIR}/summary_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEM GPU Solver Performance Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 80 + "\n\n")
        
        # Compare methods for largest mesh
        largest_mesh = max(data.keys(), key=extract_mesh_size)
        f.write(f"Analysis for largest mesh: {largest_mesh}\n\n")
        
        for method in ['none', 'rcm', 'amd']:
            f.write(f"{METHOD_LABELS[method]}:\n")
            if f'total_ms_mean' in data[largest_mesh] and data[largest_mesh].get(f'total_ms_method') == method:
                total = data[largest_mesh][f'total_ms_mean']
                solve = data[largest_mesh].get(f'gpu_solve_ms_mean', 0)
                f.write(f"  Total Time: {total:.2f} ms\n")
                f.write(f"  GPU Solve: {solve:.2f} ms ({solve/total*100:.1f}% of total)\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
    
    print(f"Saved: {report_file}")

def main():
    print("=" * 80)
    print("FEM GPU Solver Analysis and Visualization")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading benchmark data...")
    data = load_benchmark_data()
    if not data:
        print("ERROR: No benchmark data found. Run benchmark script first.")
        return
    
    print(f"Loaded data for {len(data)} meshes")
    print()
    
    # Generate plots
    print("Generating plots...")
    plot_timing_comparison(data)
    plot_spmv_analysis(data)
    plot_scalability(data)
    
    # Generate report
    print("\nGenerating summary report...")
    generate_summary_report(data)
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check the 'plots' directory for outputs.")
    print("=" * 80)

if __name__ == "__main__":
    main()