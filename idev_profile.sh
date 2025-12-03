#!/bin/bash
# ============================================================================
# FEM GPU Solver Interactive Benchmarking and Profiling Script
# Designed to run directly after calling 'idev -p rtx-dev' on TACC Frontera.
#
# CRITICAL FIX APPLIED: Removed 'srun' command from executables as the script
# is already running on a compute node via 'idev'.
# ============================================================================

# Check for profile-only flag
SKIP_BENCHMARKING=false
if [[ "$1" == "--profile-only" ]]; then
    SKIP_BENCHMARKING=true
    echo "=========================================="
    echo "  PROFILING ONLY MODE ACTIVATED"
    echo "  Skipping Phase 1 (Benchmarking)."
fi

set -e

# --- Interactive Session Variables ---
JOB_ID="Interactive-$$"
NODE_INFO=$(hostname)
WORKING_DIR=$(pwd)
# -----------------------------------

echo "=========================================="
echo "  FEM GPU Solver: Benchmark + Profile"
echo "  Job ID: $JOB_ID"
echo "  Node: $NODE_INFO"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge

# --- Module Load Logic (Adjusted for Clarity) ---
PROFILE_MODULE="cuda/11.3" # Define the module containing nsys/ncu

if $SKIP_BENCHMARKING; then
    # Load environment for profiling tools
    echo "Loading profiling environment ($PROFILE_MODULE) for tools..."
    module load $PROFILE_MODULE
else
    # Load working environment for the executable (cuda/10.1 / gcc/6.3.0)
    echo "Loading stable execution environment (cuda/10.1 / gcc/6.3.0)..."
    module load gcc/6.3.0
    module load cuda/10.1
fi

module list
echo ""

# *** CRITICAL FIX: EXPLICITLY SET LD_LIBRARY_PATH ***
# Retaining this to resolve "stub library" error.
if [ -n "$CUDA_HOME" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "Diagnostic: LD_LIBRARY_PATH updated successfully."
else
    echo "WARNING: CUDA_HOME environment variable not set. Assuming module load failed."
fi
# ----------------------------------------------------

# Check CUDA and GPU details
echo "=== CUDA Check ==="
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo ""

# Navigate to working directory (assuming current directory is the submission directory)
cd $WORKING_DIR
echo "Working directory: $WORKING_DIR"
echo ""

# Define meshes and methods
ALL_MESHES=(
    "CPU/bracket_3d.msh"
    "CPU/bracket_3d_large.msh"
)

# For profiling, use subset (profiling is expensive)
PROFILE_MESHES=(
    "CPU/bracket_3d.msh"
    "CPU/bracket_3d_large.msh"
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
# PHASE 1: BENCHMARKING (Conditional)
# ============================================================================

if ! $SKIP_BENCHMARKING; then
    echo "=========================================="
    echo "  PHASE 1: Running Benchmarks"
    # ... (omitted summary text for brevity)

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

                # Run and capture output - FIXED: NO srun, run directly
                # Old: output=$(srun --cpu-bind=none ./fem_assembly "$mesh_file" "$method" 2>&1)
                output=$(./fem_assembly "$mesh_file" "$method" 2>&1)

                # Extract timing from results file
                # ... (rest of Phase 1 logic remains the same)
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
else
    echo "=========================================="
    echo "  Phase 1 Skipped: Moving to Profiling"
    echo "=========================================="
fi

# ============================================================================
# PHASE 2: PROFILING
# ============================================================================

# ... (omitted summary text for brevity)

# --- TOOL AVAILABILITY CHECK ---
# (Logic remains the same)
# -------------------------------

for mesh_file in "${PROFILE_MESHES[@]}"; do
    if [ ! -f "$mesh_file" ]; then
        echo "WARNING: Mesh file not found: $mesh_file, skipping..."
        continue
    fi

    mesh_base=$(get_mesh_base "$mesh_file")
    # ... (omitted text for brevity)

    for method in "${METHODS[@]}"; do
        echo "  Method: $method"

        profile_prefix="$PROFILE_DIR/${mesh_base}__${method}"

        # 1. Use nsys for comprehensive system-level profiling and timeline
        echo "    Running nsys profile..."
        # FIXED: NO srun, run nsys directly
        # Old: srun --cpu-bind=none nsys profile \
        nsys profile \
            --output="$profile_prefix" \
            --stats=true \
            --trace=cuda,nvtx \
            ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1

        # 2. Use ncu for detailed kernel-level metric profiling (replaces nvprof)
        echo "    Collecting detailed metrics with ncu..."
        # FIXED: NO srun, run ncu directly
        # Old: srun --cpu-bind=none ncu --set full --csv --log-file="$profile_prefix.ncu_metrics.csv" \
        ncu --set full --csv --log-file="$profile_prefix.ncu_metrics.csv" \
            --force-overwrite \
            ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1 || true

        # 3. Also collect basic timing breakdown
        # ... (rest of Phase 2 logic remains the same)
        result_file="results/$method/${mesh_base}__${method}_results.txt"
        if [ -f "$result_file" ]; then
            cp "$result_file" "$profile_prefix.timing.txt"
        fi

        echo "    Profile saved: $profile_prefix.nsys-rep and $profile_prefix.ncu_metrics.csv"
    done
    echo ""
done

# ... (rest of the script's summary remains the same)