#!/bin/bash
#SBATCH --job-name=fem_xlarge_profile
#SBATCH --output=xlarge_profile_%j.out
#SBATCH --error=xlarge_profile_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A TRA25006

# ============================================================================
# FEM GPU Solver: xlarge Mesh Run & Profile
# Runs solver first to generate results, then profiles with nsys + ncu
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: xlarge Run & Profile"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/12.2  # Module with nsys/ncu tools
module load gcc/9.1.0
module list
echo ""

# Set LD_LIBRARY_PATH if needed
if [ -n "$CUDA_HOME" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH updated: $LD_LIBRARY_PATH"
fi
echo ""

# Check CUDA and GPU details
echo "=== CUDA Check ==="
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo ""

# Navigate to working directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Check if executable exists
if [ ! -f "./fem_assembly" ]; then
    echo "ERROR: Executable not found. Building..."
    make clean
    make
    if [ ! -f "./fem_assembly" ]; then
        echo "ERROR: Build failed - executable not found"
        exit 1
    fi
    echo "Build successful!"
    echo ""
fi

# Define mesh - only xlarge
MESH_FILE="CPU/bracket_3d_xlarge.msh"
MESH_BASE="bracket_3d_xlarge"

METHODS=("none" "rcm" "amd")
BENCHMARK_DIR="xlarge_benchmark_results"
PROFILE_DIR="xlarge_profile_results"
mkdir -p $BENCHMARK_DIR
mkdir -p $PROFILE_DIR

# Function to extract mesh base name
get_mesh_base() {
    local mesh_file=$1
    local basename=$(basename "$mesh_file" .msh)
    echo "$basename"
}

# Check tool availability
echo "=== Checking Profiling Tools ==="
NSYS_EXISTS=true
command -v nsys >/dev/null 2>&1 || NSYS_EXISTS=false

NCU_EXISTS=true
command -v ncu >/dev/null 2>&1 || NCU_EXISTS=false

if $NSYS_EXISTS; then
    echo "✓ nsys found: $(which nsys)"
    nsys --version
else
    echo "✗ nsys not found - system-level profiling will be skipped"
fi

if $NCU_EXISTS; then
    echo "✓ ncu found: $(which ncu)"
    ncu --version
else
    echo "✗ ncu not found - kernel-level metric profiling will be skipped"
fi
echo ""

# ============================================================================
# PHASE 1: RUN SOLVER (Generate Results)
# ============================================================================

if [ ! -f "$MESH_FILE" ]; then
    echo "ERROR: Mesh file not found: $MESH_FILE"
    exit 1
fi

echo "=========================================="
echo "  PHASE 1: Running Solver"
echo "  Mesh: $MESH_BASE"
echo "  Methods: ${METHODS[@]}"
echo "=========================================="
echo ""

for method in "${METHODS[@]}"; do
    echo "----------------------------------------"
    echo "Running mesh: $MESH_BASE, method: $method"
    echo "----------------------------------------"
    
    # Run the solver
    echo "  Executing solver..."
    ./fem_assembly "$MESH_FILE" "$method" 2>&1
    
    # Extract timing and save to benchmark CSV
    result_file="results/$method/${MESH_BASE}__${method}_results.txt"
    if [ -f "$result_file" ]; then
        cpu=$(grep "CPU_Assembly_ms" "$result_file" | awk '{print $2}')
        reorder=$(grep "Reordering_ms" "$result_file" | awk '{print $2}')
        h2d=$(grep "HostToDevice_ms" "$result_file" | awk '{print $2}')
        solve=$(grep "GPU_Solve_ms" "$result_file" | awk '{print $2}')
        total=$(echo "$cpu + $reorder + $h2d + $solve" | bc -l)
        
        # Save to CSV
        output_file="$BENCHMARK_DIR/${MESH_BASE}__${method}.csv"
        echo "run,cpu_assembly_ms,reordering_ms,h2d_ms,gpu_solve_ms,total_ms" > "$output_file"
        echo "1,$cpu,$reorder,$h2d,$solve,$total" >> "$output_file"
        echo "  ✓ Results saved to: $output_file"
    else
        echo "  ⚠ WARNING: Results file not found: $result_file"
    fi
    echo ""
done

echo "=========================================="
echo "  Phase 1 Complete: Solver Runs Done!"
echo "=========================================="
echo ""

# ============================================================================
# PHASE 2: PROFILING
# ============================================================================

echo "=========================================="
echo "  PHASE 2: Running Profiling"
echo "  Mesh: $MESH_BASE"
echo "  Methods: ${METHODS[@]}"
echo "  Tools: nsys=$NSYS_EXISTS, ncu=$NCU_EXISTS"
echo "=========================================="
echo ""

for method in "${METHODS[@]}"; do
    echo "----------------------------------------"
    echo "Profiling mesh: $MESH_BASE, method: $method"
    echo "----------------------------------------"
    
    profile_prefix="$PROFILE_DIR/${MESH_BASE}__${method}"
    
    # 1. Use nsys for comprehensive system-level profiling and timeline
    if $NSYS_EXISTS; then
        echo "  Running nsys profile..."
        nsys profile \
            --output="$profile_prefix" \
            --stats=true \
            --trace=cuda,nvtx \
            ./fem_assembly "$MESH_FILE" "$method" > /dev/null 2>&1
        
        if [ -f "${profile_prefix}.nsys-rep" ]; then
            echo "    ✓ nsys report generated: ${profile_prefix}.nsys-rep"
        else
            echo "    ✗ nsys report not found (may have failed)"
        fi
    else
        echo "  Skipping nsys profiling (tool not available)"
    fi
    
    # 2. Use ncu for detailed kernel-level metric profiling
    if $NCU_EXISTS; then
        echo "  Collecting detailed metrics with ncu..."
        # Use --set default instead of full to avoid resource exhaustion
        # Add --target-processes all to handle child processes
        # --csv outputs to stdout, redirect to file
        # --log-file captures ncu's own log messages
        # Capture stderr separately to see actual errors
        ncu --set default \
            --target-processes all \
            --csv --log-file="${profile_prefix}.ncu.log" \
            --force-overwrite \
            ./fem_assembly "$MESH_FILE" "$method" > "${profile_prefix}.ncu_metrics.csv" 2> "${profile_prefix}.ncu_stderr.log" || true
        
        if [ -f "${profile_prefix}.ncu_metrics.csv" ]; then
            echo "    ✓ ncu metrics generated: ${profile_prefix}.ncu_metrics.csv"
        else
            echo "    ✗ ncu metrics not found (may have failed or hit resource limits)"
            if [ -f "${profile_prefix}.ncu_stderr.log" ]; then
                echo "    Error output:"
                tail -20 "${profile_prefix}.ncu_stderr.log" | sed 's/^/      /'
            fi
        fi
    else
        echo "  Skipping ncu profiling (tool not available)"
    fi
    
    # 3. Copy timing data
    result_file="results/$method/${MESH_BASE}__${method}_results.txt"
    if [ -f "$result_file" ]; then
        cp "$result_file" "${profile_prefix}.timing.txt"
        echo "    ✓ Timing data copied: ${profile_prefix}.timing.txt"
    else
        echo "    ⚠ Timing file not found: $result_file"
    fi
    
    echo ""
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "  All Phases Complete!"
echo "=========================================="
echo ""
echo "Benchmark results saved in: $BENCHMARK_DIR/"
echo "Profile results saved in: $PROFILE_DIR/"
echo ""

# Count generated files
benchmark_count=$(ls -1 $BENCHMARK_DIR/*.csv 2>/dev/null | wc -l)
nsys_count=$(ls -1 $PROFILE_DIR/*.nsys-rep 2>/dev/null | wc -l)
ncu_count=$(ls -1 $PROFILE_DIR/*.ncu_metrics.csv 2>/dev/null | wc -l)
timing_count=$(ls -1 $PROFILE_DIR/*.timing.txt 2>/dev/null | wc -l)

echo "Summary:"
echo "  - Benchmark CSV files: $benchmark_count"
echo "  - nsys reports: $nsys_count"
echo "  - ncu metrics files: $ncu_count"
echo "  - timing files: $timing_count"
echo ""
echo "To view profiles:"
echo "  nsys-ui $PROFILE_DIR/*.nsys-rep"
echo "  (or download .nsys-rep files and open with Nsight Systems)"
echo ""
echo "To analyze ncu metrics:"
echo "  cat $PROFILE_DIR/*.ncu_metrics.csv"
echo ""
echo "To view benchmark results:"
echo "  cat $BENCHMARK_DIR/*.csv"
echo ""