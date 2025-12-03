#!/bin/bash
#SBATCH --job-name=fem_large_profile
#SBATCH --output=large_profile_%j.out
#SBATCH --error=large_profile_%j.err
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A TRA25006

# ============================================================================
# FEM GPU Solver: large Mesh Run & Profile
# Runs solver firstto generate results, then profiles with nsys + ncu
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: large Run & Profile"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/12.2  # Module with nsys/ncu tools
module load gcc/9.4.0
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

# Define mesh - only large
MESH_FILE="CPU/bracket_3d_large.msh"
MESH_BASE="bracket_3d_large"

METHODS=("none" "rcm" "amd")
BENCHMARK_DIR="large_benchmark_results_a100"
PROFILE_DIR="large_profile_results_a100"
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
    
    # Debug: Verify method is being used
    echo "  DEBUG: Method parameter = '$method'"
    echo "  DEBUG: Expected results file = results/$method/${MESH_BASE}__${method}_results.txt"
    
    # Arrays to store results from 10 runs
    declare -a cpu_values
    declare -a reorder_values
    declare -a h2d_values
    declare -a solve_values
    declare -a total_values
    
    # Run the solver 10 times
    NUM_RUNS=10
    echo "  Running $NUM_RUNS benchmark iterations..."
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "    Run $run/$NUM_RUNS..."
        
        # Debug: Show what command is being run (only for first run to avoid spam)
        if [ $run -eq 1 ]; then
            echo "      Executing: ./fem_assembly \"$MESH_FILE\" \"$method\""
        fi
        
        ./fem_assembly "$MESH_FILE" "$method" > /dev/null 2>&1
        
        # Extract timing from results file
        result_file="results/$method/${MESH_BASE}__${method}_results.txt"
        
        # Wait a moment to ensure file is written
        sleep 0.1
        
        # Verify file exists and show its contents (only for first run)
        if [ -f "$result_file" ]; then
            if [ $run -eq 1 ]; then
                echo "      Results file found: $result_file"
                echo "      File contents:"
                cat "$result_file" | sed 's/^/        /'
            fi
            
            cpu=$(grep "CPU_Assembly_ms" "$result_file" | awk '{print $2}')
            reorder=$(grep "Reordering_ms" "$result_file" | awk '{print $2}')
            h2d=$(grep "HostToDevice_ms" "$result_file" | awk '{print $2}')
            solve=$(grep "GPU_Solve_ms" "$result_file" | awk '{print $2}')
            total=$(echo "$cpu + $reorder + $h2d + $solve" | bc -l)
            
            # Store values in arrays
            cpu_values+=("$cpu")
            reorder_values+=("$reorder")
            h2d_values+=("$h2d")
            solve_values+=("$solve")
            total_values+=("$total")
            
            # Debug: Print what we read (only for first run)
            if [ $run -eq 1 ]; then
                echo "      Read values: cpu=$cpu, reorder=$reorder, h2d=$h2d, solve=$solve, total=$total"
            fi
        else
            echo "    ⚠ WARNING: Results file not found for run $run: $result_file"
            echo "    Available files in results/$method/:"
            ls -la "results/$method/" 2>&1 | sed 's/^/      /' || echo "      Directory does not exist"
        fi
    done
    
    # Calculate averages
    cpu_sum=0
    reorder_sum=0
    h2d_sum=0
    solve_sum=0
    total_sum=0
    
    for i in $(seq 0 $((NUM_RUNS - 1))); do
        cpu_sum=$(echo "$cpu_sum + ${cpu_values[$i]}" | bc -l)
        reorder_sum=$(echo "$reorder_sum + ${reorder_values[$i]}" | bc -l)
        h2d_sum=$(echo "$h2d_sum + ${h2d_values[$i]}" | bc -l)
        solve_sum=$(echo "$solve_sum + ${solve_values[$i]}" | bc -l)
        total_sum=$(echo "$total_sum + ${total_values[$i]}" | bc -l)
    done
    
    cpu_avg=$(echo "scale=6; $cpu_sum / $NUM_RUNS" | bc -l)
    reorder_avg=$(echo "scale=6; $reorder_sum / $NUM_RUNS" | bc -l)
    h2d_avg=$(echo "scale=6; $h2d_sum / $NUM_RUNS" | bc -l)
    solve_avg=$(echo "scale=6; $solve_sum / $NUM_RUNS" | bc -l)
    total_avg=$(echo "scale=6; $total_sum / $NUM_RUNS" | bc -l)
    
    # Save to CSV with all runs and average
    output_file="$BENCHMARK_DIR/${MESH_BASE}__${method}.csv"
    echo "run,cpu_assembly_ms,reordering_ms,h2d_ms,gpu_solve_ms,total_ms" > "$output_file"
    
    # Write all 10 runs
    for i in $(seq 0 $((NUM_RUNS - 1))); do
        run_num=$((i + 1))
        echo "$run_num,${cpu_values[$i]},${reorder_values[$i]},${h2d_values[$i]},${solve_values[$i]},${total_values[$i]}" >> "$output_file"
    done
    
    # Write average row
    echo "average,$cpu_avg,$reorder_avg,$h2d_avg,$solve_avg,$total_avg" >> "$output_file"
    
    echo "  ✓ Results saved to: $output_file (10 runs + average)"
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