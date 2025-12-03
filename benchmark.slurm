#!/bin/bash
#SBATCH --job-name=fem_benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

# ============================================================================
# FEM GPU Solver Comprehensive Benchmarking Script
# Runs all mesh sizes 10 times each with all reordering methods
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver Benchmarking"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/10.1
module load gcc/6.3.0
module list
echo ""

# Check CUDA
echo "=== CUDA Check ==="
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo ""

# Navigate to working directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Build project
echo "=== Building Project ==="
make clean
make
if [ ! -f "./fem_assembly" ]; then
    echo "ERROR: Build failed"
    exit 1
fi
echo "Build successful!"
echo ""

# Define meshes and methods
MESHES=(
    "CPU/bracket_3d.msh"
    "CPU/bracket_3d_large.msh"
    "CPU/bracket_3d_xlarge.msh"
    "CPU/bracket_3d_xxlarge.msh"
    "CPU/bracket_3d_xxxlarge.msh"
    "CPU/bracket_3d_xxxxlarge.msh"
)

METHODS=("none" "rcm" "amd")
NUM_RUNS=10

# Create output directories
BENCHMARK_DIR="benchmark_results"
PROFILE_DIR="profile_results"
mkdir -p $BENCHMARK_DIR
mkdir -p $PROFILE_DIR

echo "=========================================="
echo "  Running Benchmarks"
echo "  Meshes: ${#MESHES[@]}"
echo "  Methods: ${METHODS[@]}"
echo "  Runs per configuration: $NUM_RUNS"
echo "=========================================="
echo ""

# Function to extract mesh base name
get_mesh_base() {
    local mesh_file=$1
    local basename=$(basename "$mesh_file" .msh)
    echo "$basename"
}

# Run benchmarks
for mesh_file in "${MESHES[@]}"; do
    if [ ! -f "$mesh_file" ]; then
        echo "WARNING: Mesh file not found: $mesh_file, skipping..."
        continue
    fi
    
    mesh_base=$(get_mesh_base "$mesh_file")
    echo "----------------------------------------"
    echo "Processing mesh: $mesh_base"
    echo "----------------------------------------"
    
    for method in "${METHODS[@]}"; do
        echo "  Method: $method"
        
        # Create output file for this configuration
        output_file="$BENCHMARK_DIR/${mesh_base}__${method}.csv"
        echo "run,cpu_assembly_ms,reordering_ms,h2d_ms,gpu_solve_ms,total_ms" > "$output_file"
        
        # Run multiple times
        for run in $(seq 1 $NUM_RUNS); do
            echo "    Run $run/$NUM_RUNS..."
            
            # Run and capture output
            output=$(./fem_assembly "$mesh_file" "$method" 2>&1)
            
            # Extract timing from results file
            result_file="results/$method/${mesh_base}__${method}_results.txt"
            if [ -f "$result_file" ]; then
                cpu=$(grep "CPU_Assembly_ms" "$result_file" | awk '{print $2}')
                reorder=$(grep "Reordering_ms" "$result_file" | awk '{print $2}')
                h2d=$(grep "HostToDevice_ms" "$result_file" | awk '{print $2}')
                solve=$(grep "GPU_Solve_ms" "$result_file" | awk '{print $2}')
                
                # Calculate total
                total=$(echo "$cpu + $reorder + $h2d + $solve" | bc -l)
                
                # Append to CSV
                echo "$run,$cpu,$reorder,$h2d,$solve,$total" >> "$output_file"
            else
                echo "    WARNING: Results file not found: $result_file"
            fi
        done
        
        echo "    Results saved to: $output_file"
    done
    echo ""
done

echo "=========================================="
echo "  Benchmarking Complete!"
echo "=========================================="
echo ""
echo "Results saved in: $BENCHMARK_DIR/"
echo ""
echo "Summary of collected data:"
ls -lh $BENCHMARK_DIR/*.csv 2>/dev/null | wc -l
echo " files generated"