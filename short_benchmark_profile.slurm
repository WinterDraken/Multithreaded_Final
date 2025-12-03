#!/bin/bash
#SBATCH --job-name=short_benchmark_profile
#SBATCH --output=short_benchmark_profile_%j.out
#SBATCH --error=short_benchmark_profile_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

# ============================================================================
# FEM GPU Solver Combined Benchmarking and Profiling Script
# Runs benchmarks first, then profiling for selected meshes
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: Benchmark + Profile"
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

# Define meshes and methods
ALL_MESHES=(
    "CPU/bracket_3d.msh"
    "CPU/bracket_3d_large.msh"
    "CPU/bracket_3d_xlarge.msh"
    "CPU/bracket_3d_xxlarge.msh"
    "CPU/bracket_3d_xxxlarge.msh"
    "CPU/bracket_3d_xxxxlarge.msh"
)

# For profiling, use subset (profiling is expensive)
PROFILE_MESHES=(
    "CPU/bracket_3d.msh"
    "CPU/bracket_3d_large.msh"
    "CPU/bracket_3d_xlarge.msh"
    "CPU/bracket_3d_xxlarge.msh"
)

METHODS=("none" "rcm" "amd")
NUM_RUNS=1

# Create output directories
BENCHMARK_DIR="short_benchmark_results"
PROFILE_DIR="short_profile_results"
mkdir -p $BENCHMARK_DIR
mkdir -p $PROFILE_DIR

# Function to extract mesh base name
get_mesh_base() {
    local mesh_file=$1
    local basename=$(basename "$mesh_file" .msh)
    echo "$basename"
}

# ============================================================================
# PHASE 1: BENCHMARKING
# ============================================================================

echo "=========================================="
echo "  PHASE 1: Running Benchmarks"
echo "  Meshes: ${#ALL_MESHES[@]}"
echo "  Methods: ${METHODS[@]}"
echo "  Runs per configuration: $NUM_RUNS"
echo "=========================================="
echo ""

# Run benchmarks for all meshes
for mesh_file in "${ALL_MESHES[@]}"; do
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
echo "  Phase 1 Complete: Benchmarking Done!"
echo "=========================================="
echo ""
echo "Results saved in: $BENCHMARK_DIR/"
echo "Summary: $(ls -1 $BENCHMARK_DIR/*.csv 2>/dev/null | wc -l) CSV files generated"
echo ""

# ============================================================================
# PHASE 2: PROFILING
# ============================================================================

echo "=========================================="
echo "  PHASE 2: Running Profiling"
echo "  Meshes for profiling: ${#PROFILE_MESHES[@]}"
echo "  Methods: ${METHODS[@]}"
echo "  (Profiling is expensive, using subset of meshes)"
echo "=========================================="
echo ""

for mesh_file in "${PROFILE_MESHES[@]}"; do
    if [ ! -f "$mesh_file" ]; then
        echo "WARNING: Mesh file not found: $mesh_file, skipping..."
        continue
    fi
    
    mesh_base=$(get_mesh_base "$mesh_file")
    echo "----------------------------------------"
    echo "Profiling mesh: $mesh_base"
    echo "----------------------------------------"
    
    for method in "${METHODS[@]}"; do
        echo "  Method: $method"
        
        profile_prefix="$PROFILE_DIR/${mesh_base}__${method}"
        
        # Use nsys for comprehensive profiling
        echo "    Running nsys profile..."
        nsys profile \
            --output="$profile_prefix" \
            --stats=true \
            --trace=cuda,nvtx \
            ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1
        
        # Extract key metrics using nvprof (if available)
        echo "    Collecting detailed metrics..."
        
        if command -v nvprof &> /dev/null; then
            nvprof --csv --log-file="$profile_prefix.metrics.csv" \
                --metrics=sm_efficiency,warp_efficiency,shared_efficiency,achieved_occupancy, \
                l2_cache_hit_rate,l2_read_throughput,l2_write_throughput, \
                dram_read_throughput,dram_write_throughput, \
                gld_throughput,gst_throughput, \
                atomic_transactions,atomic_transaction_per_request \
                ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1 || true
        fi
        
        # Also collect basic timing breakdown
        result_file="results/$method/${mesh_base}__${method}_results.txt"
        if [ -f "$result_file" ]; then
            cp "$result_file" "$profile_prefix.timing.txt"
        fi
        
        echo "    Profile saved: $profile_prefix"
    done
    echo ""
done

echo "=========================================="
echo "  Phase 2 Complete: Profiling Done!"
echo "=========================================="
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "  ALL PHASES COMPLETE!"
echo "=========================================="
echo ""
echo "Benchmark results: $BENCHMARK_DIR/"
echo "  - $(ls -1 $BENCHMARK_DIR/*.csv 2>/dev/null | wc -l) CSV files"
echo ""
echo "Profile results: $PROFILE_DIR/"
echo "  - $(ls -1 $PROFILE_DIR/*.nsys-rep 2>/dev/null | wc -l) nsys reports"
echo "  - $(ls -1 $PROFILE_DIR/*.metrics.csv 2>/dev/null | wc -l) metrics files"
echo ""
echo "Next steps:"
echo "  1. Run plotting script: python3 plot_analysis.py"
echo "  2. View profiles: nsys-ui $PROFILE_DIR/*.nsys-rep"
echo ""